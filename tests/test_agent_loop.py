from __future__ import annotations

from app.config import settings
from app.agent import run_turn
from app.llm import FakeLLM


def test_loop_stops_after_max_steps_and_hands_off(conn):
    # 7 tool_call steps in a row, each targeting a harmless read-only tool,
    # so the only way the loop can end is by hitting the step budget.
    script = [{"type": "tool_call", "tool": "get_balance", "args": {}} for _ in range(7)]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="hello", trace_id="t1")
    assert resp.status == "handoff"
    assert llm._call_count == settings.max_agent_steps


def test_exactly_max_steps_budget_is_respected(conn):
    assert settings.max_agent_steps == 6
