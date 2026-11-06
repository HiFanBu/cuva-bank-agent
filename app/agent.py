"""The hand-written agent loop.

Per the spec this is intentionally NOT built on an agent framework -- it is
a plain loop so every control point (step budget, retry budget, which tools
require confirmation, what counts as a must-handoff signal) is visible and
testable in one place.

Flow for one `/chat` turn:
  1. input guardrail: injection detection (logged, never trusted) + risk
     routing. A must-handoff signal (fraud/dispute/complaint/human
     request/vulnerable customer) short-circuits straight to a human case
     without ever calling the model, matching "先用简单路由" and keeping
     complaint/dispute recall independent of model behavior.
  2. otherwise, run the loop: ask the model for one AgentStep, validate it,
     execute at most one tool per step, feed the result back as the next
     observation, for at most `settings.max_agent_steps` steps.
  3. a `schedule_payment` tool call is only ever validated here, turned into
     a `pending_actions` row, and returned as `needs_confirmation` --
     money moves only in `confirm_payment` below.
  4. a `create_case` tool call always ends the turn as a handoff.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.audit import load_recent_messages, log_conversation_message, log_event
from app.config import settings
from app.guardrails.input_guard import redact_pii, run_input_guard
from app.guardrails.output_guard import check_final_answer
from app.llm import LLMProvider
from app.schemas import (
    ChatResponse,
    CreateCaseArgs,
    FinalStep,
    HandoffReason,
    HandoffStep,
    ToolCallStep,
)
from app.tools.base import execute_tool
from app.tools.complaint import create_case
from app.tools.payment import execute_scheduled_payment


class AgentError(Exception):
    pass


def _parse_agent_step(raw: str) -> ToolCallStep | FinalStep | HandoffStep:
    data = json.loads(raw)  # may raise json.JSONDecodeError
    step_type = data.get("type")
    if step_type == "tool_call":
        return ToolCallStep.model_validate(data)
    if step_type == "final":
        return FinalStep.model_validate(data)
    if step_type == "handoff":
        return HandoffStep.model_validate(data)
    raise ValueError(f"unknown AgentStep type: {step_type!r}")


def _get_valid_step(
    llm: LLMProvider, history: list[dict[str, Any]], trace_id: str
) -> tuple[ToolCallStep | FinalStep | HandoffStep | None, str | None]:
    local_history = list(history)
    attempts = 0
    while attempts <= settings.max_json_repair_retries:
        raw = llm.generate_step(history=local_history, trace_id=trace_id)
        try:
            return _parse_agent_step(raw), None
        except Exception as e:  # json decode or pydantic validation error
            attempts += 1
            local_history.append(
                {
                    "role": "system_error",
                    "content": f"Your previous output was invalid ({e}). "
                    "Respond with exactly one JSON object matching the AgentStep schema.",
                }
            )
    return None, "invalid_json_after_retries"


def _handoff_response(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    trace_id: str,
    reason: HandoffReason,
    summary: str,
    file_case: bool = True,
) -> ChatResponse:
    if file_case:
        result = create_case(
            user_id,
            CreateCaseArgs(category=reason, summary=redact_pii(summary)),
            conn,
        )
        log_event(
            conn,
            trace_id=trace_id,
            user_id=user_id,
            event="handoff",
            tool="create_case",
            detail={"reason": reason, "case_id": result.data.get("case_id")},
        )
    return ChatResponse(
        status="handoff",
        message=(
            "I'm connecting you with a specialist who can help with this. "
            "A summary of your request has been shared with them."
        ),
        citations=[],
        action_id=None,
        trace_id=trace_id,
    )


def run_turn(
    conn: sqlite3.Connection,
    llm: LLMProvider,
    *,
    user_id: str,
    conversation_id: str,
    message: str,
    trace_id: str,
) -> ChatResponse:
    # Earlier turns give the model enough context for multi-turn requests
    # ("I want to pay my bill" -> "how much?" -> "$500 on Friday"). Loaded
    # before logging the current message so it isn't duplicated.
    prior_turns = load_recent_messages(
        conn, conversation_id=conversation_id, user_id=user_id
    )
    log_conversation_message(
        conn, conversation_id=conversation_id, user_id=user_id, role="user", content=message
    )

    guard = run_input_guard(message)
    if guard.injection_detected:
        log_event(
            conn,
            trace_id=trace_id,
            user_id=user_id,
            event="prompt_injection_detected",
            detail={"message_excerpt": message[:200]},
        )
    if guard.must_handoff_reason is not None:
        log_event(
            conn,
            trace_id=trace_id,
            user_id=user_id,
            event="input_guard_routed_handoff",
            detail={"reason": guard.must_handoff_reason},
        )
        resp = _handoff_response(
            conn,
            user_id=user_id,
            trace_id=trace_id,
            reason=guard.must_handoff_reason,
            summary=message,
        )
        log_conversation_message(
            conn, conversation_id=conversation_id, user_id=user_id, role="assistant", content=resp.message
        )
        return resp

    history: list[dict[str, Any]] = [*prior_turns, {"role": "user", "content": message}]
    used_policy_tool = False

    for _ in range(settings.max_agent_steps):
        step, error = _get_valid_step(llm, history, trace_id)
        if error:
            log_event(conn, trace_id=trace_id, user_id=user_id, event="agent_step_failed", detail={"error": error})
            return _handoff_response(
                conn,
                user_id=user_id,
                trace_id=trace_id,
                reason="invalid_model_output",
                summary=f"Automated handling failed for: {message}",
            )

        if isinstance(step, FinalStep):
            guard_result = check_final_answer(step.message, step.citations, used_policy_tool)
            if not guard_result.passed:
                log_event(
                    conn,
                    trace_id=trace_id,
                    user_id=user_id,
                    event="output_guard_blocked",
                    detail={"reason": guard_result.reason},
                )
                return _handoff_response(
                    conn,
                    user_id=user_id,
                    trace_id=trace_id,
                    reason="low_confidence",
                    summary=f"Could not safely answer: {message}",
                )
            log_event(conn, trace_id=trace_id, user_id=user_id, event="final_answer")
            log_conversation_message(
                conn, conversation_id=conversation_id, user_id=user_id, role="assistant", content=step.message
            )
            return ChatResponse(
                status="answered",
                message=step.message,
                citations=step.citations,
                action_id=None,
                trace_id=trace_id,
            )

        if isinstance(step, HandoffStep):
            return _handoff_response(
                conn, user_id=user_id, trace_id=trace_id, reason=step.reason, summary=step.summary
            )

        # ToolCallStep
        if step.tool == "create_case":
            execute_tool("create_case", step.args, user_id=user_id, conn=conn)
            return _handoff_response(
                conn,
                user_id=user_id,
                trace_id=trace_id,
                reason="complaint",
                summary=step.args.get("summary", message),
                file_case=False,  # already filed by the execute_tool call above
            )

        if step.tool == "schedule_payment":
            result = execute_tool("schedule_payment", step.args, user_id=user_id, conn=conn)
            if not result.ok:
                history.append({"role": "observation", "tool": "schedule_payment", "content": {"error": result.error}})
                log_event(conn, trace_id=trace_id, user_id=user_id, event="tool_error", tool="schedule_payment", detail={"error": result.error})
                continue
            action_id = f"act_{uuid.uuid4().hex[:16]}"
            now = datetime.now(timezone.utc)
            expires_at = now + timedelta(seconds=settings.pending_action_ttl_seconds)
            conn.execute(
                """
                INSERT INTO pending_actions (action_id, user_id, conversation_id, tool, args, created_at, expires_at, status)
                VALUES (?, ?, ?, 'schedule_payment', ?, ?, ?, 'pending')
                """,
                (
                    action_id,
                    user_id,
                    conversation_id,
                    json.dumps(result.data),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            conn.commit()
            log_event(
                conn,
                trace_id=trace_id,
                user_id=user_id,
                event="pending_action_created",
                tool="schedule_payment",
                detail={"action_id": action_id, **result.data},
            )
            amount = result.data["amount_cents"] / 100
            return ChatResponse(
                status="needs_confirmation",
                message=f"Confirm scheduling a payment of ${amount:.2f} on {result.data['pay_date']}?",
                citations=[],
                action_id=action_id,
                trace_id=trace_id,
            )

        # Read-only / low-risk tools execute immediately.
        result = execute_tool(step.tool, step.args, user_id=user_id, conn=conn)
        if step.tool == "search_policy":
            used_policy_tool = True
        if result.ok:
            history.append({"role": "observation", "tool": step.tool, "content": result.data})
            log_event(conn, trace_id=trace_id, user_id=user_id, event="tool_call", tool=step.tool, detail=result.data)
        else:
            history.append({"role": "observation", "tool": step.tool, "content": {"error": result.error}})
            log_event(conn, trace_id=trace_id, user_id=user_id, event="tool_error", tool=step.tool, detail={"error": result.error})

    log_event(conn, trace_id=trace_id, user_id=user_id, event="max_steps_exceeded")
    return _handoff_response(
        conn,
        user_id=user_id,
        trace_id=trace_id,
        reason="max_steps_exceeded",
        summary=f"Could not resolve within the step budget: {message}",
    )


def confirm_payment(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    conversation_id: str,
    action_id: str,
    trace_id: str,
) -> ChatResponse:
    row = conn.execute(
        """
        SELECT action_id, user_id, tool, args, expires_at, status
        FROM pending_actions WHERE action_id = ?
        """,
        (action_id,),
    ).fetchone()

    if row is None:
        raise AgentError("action_not_found")
    if row["user_id"] != user_id:
        log_event(conn, trace_id=trace_id, user_id=user_id, event="confirm_rejected", detail={"reason": "not_owner", "action_id": action_id})
        raise AgentError("not_owner")
    if row["tool"] != "schedule_payment":
        raise AgentError("unsupported_action")

    expires_at = datetime.fromisoformat(row["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if row["status"] == "pending" and datetime.now(timezone.utc) > expires_at:
        log_event(conn, trace_id=trace_id, user_id=user_id, event="confirm_rejected", detail={"reason": "expired", "action_id": action_id})
        raise AgentError("expired")
    if row["status"] not in ("pending", "executed"):
        raise AgentError("invalid_status")

    args = json.loads(row["args"])
    result = execute_scheduled_payment(
        user_id=user_id,
        action_id=action_id,
        amount_cents=args["amount_cents"],
        pay_date=date.fromisoformat(args["pay_date"]),
        conn=conn,
    )
    if not result.ok:
        log_event(conn, trace_id=trace_id, user_id=user_id, event="payment_execution_failed", detail={"error": result.error})
        raise AgentError(result.error or "execution_failed")

    conn.execute(
        "UPDATE pending_actions SET status = 'executed' WHERE action_id = ?", (action_id,)
    )
    conn.commit()
    log_event(
        conn,
        trace_id=trace_id,
        user_id=user_id,
        event="payment_confirmed",
        tool="schedule_payment",
        detail=result.data,
    )
    amount = result.data["amount_cents"] / 100
    message = f"Your payment of ${amount:.2f} on {result.data['pay_date']} has been confirmed and scheduled."
    log_conversation_message(conn, conversation_id=conversation_id, user_id=user_id, role="assistant", content=message)
    return ChatResponse(
        status="answered",
        message=message,
        citations=[],
        action_id=action_id,
        trace_id=trace_id,
    )
