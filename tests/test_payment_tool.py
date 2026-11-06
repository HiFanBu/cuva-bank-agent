from __future__ import annotations

from datetime import date, timedelta

from app.agent import AgentError, confirm_payment, run_turn
from app.llm import FakeLLM
import pytest


def _future_date(days: int = 7) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def test_balance_and_transactions_are_scoped_to_caller(conn):
    llm = FakeLLM()
    resp1 = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="What's my balance?", trace_id="t1")
    resp2 = run_turn(conn, llm, user_id="user_2", conversation_id="c2", message="What's my balance?", trace_id="t2")
    assert resp1.status == "answered"
    assert resp2.status == "answered"
    # user_1's seeded balance ($1284.75) differs from user_2's ($542.30);
    # each call only ever sees its own authenticated user_id's account.
    assert "1284.75" in resp1.message or "1,284.75" in resp1.message
    assert "542.30" in resp2.message


def test_schedule_payment_requires_confirmation(conn):
    script = [
        {
            "type": "tool_call",
            "tool": "schedule_payment",
            "args": {"amount_cents": 50000, "pay_date": _future_date()},
        }
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="pay $500", trace_id="t1")
    assert resp.status == "needs_confirmation"
    assert resp.action_id is not None


def test_confirm_executes_payment_exactly_once_on_duplicate_submit(conn):
    script = [
        {
            "type": "tool_call",
            "tool": "schedule_payment",
            "args": {"amount_cents": 50000, "pay_date": _future_date()},
        }
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="pay $500", trace_id="t1")
    action_id = resp.action_id

    first = confirm_payment(conn, user_id="user_1", conversation_id="c1", action_id=action_id, trace_id="t2")
    second = confirm_payment(conn, user_id="user_1", conversation_id="c1", action_id=action_id, trace_id="t3")

    assert first.status == "answered"
    assert second.status == "answered"
    rows = conn.execute("SELECT COUNT(*) AS n FROM payments WHERE idempotency_key = ?", (action_id,)).fetchall()
    assert rows[0]["n"] == 1


def test_confirm_rejected_for_non_owner(conn):
    script = [
        {
            "type": "tool_call",
            "tool": "schedule_payment",
            "args": {"amount_cents": 50000, "pay_date": _future_date()},
        }
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="pay $500", trace_id="t1")

    with pytest.raises(AgentError, match="not_owner"):
        confirm_payment(conn, user_id="user_2", conversation_id="c1", action_id=resp.action_id, trace_id="t2")


def test_confirm_rejected_when_expired(conn):
    script = [
        {
            "type": "tool_call",
            "tool": "schedule_payment",
            "args": {"amount_cents": 50000, "pay_date": _future_date()},
        }
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="pay $500", trace_id="t1")

    conn.execute(
        "UPDATE pending_actions SET expires_at = '2000-01-01T00:00:00+00:00' WHERE action_id = ?",
        (resp.action_id,),
    )
    conn.commit()

    with pytest.raises(AgentError, match="expired"):
        confirm_payment(conn, user_id="user_1", conversation_id="c1", action_id=resp.action_id, trace_id="t2")


def test_invalid_payment_amount_rejected_without_executing(conn):
    # amount_cents <= 0 fails pydantic validation inside execute_tool, so the
    # agent feeds the error back as an observation (spec: "不执行,把错误作为
    # 观察结果返回给模型") instead of creating a pending_action.
    script = [
        {
            "type": "tool_call",
            "tool": "schedule_payment",
            "args": {"amount_cents": -100, "pay_date": _future_date()},
        },
        {
            "type": "handoff",
            "reason": "low_confidence",
            "summary": "Could not schedule an invalid payment amount.",
        },
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="pay -$1", trace_id="t1")
    assert resp.status == "handoff"
    rows = conn.execute("SELECT COUNT(*) AS n FROM pending_actions").fetchall()
    assert rows[0]["n"] == 0
    rows = conn.execute("SELECT COUNT(*) AS n FROM payments").fetchall()
    assert rows[0]["n"] == 0
