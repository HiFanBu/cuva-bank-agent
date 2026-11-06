from __future__ import annotations

from app.agent import run_turn
from app.llm import FakeLLM


def test_dispute_routes_to_handoff_without_calling_the_model(conn):
    # The input guard must catch this before the model is ever consulted --
    # a FakeLLM with an empty script would raise IndexError if the agent
    # loop tried to call generate_step, proving the short-circuit happened.
    llm = FakeLLM(script=[])
    resp = run_turn(
        conn,
        llm,
        user_id="user_1",
        conversation_id="c1",
        message="I was charged twice for the same purchase, this is a dispute",
        trace_id="t1",
    )
    assert resp.status == "handoff"
    rows = conn.execute("SELECT category, user_id FROM cases WHERE user_id = 'user_1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["category"] == "billing_dispute"


def test_fraud_report_routes_to_handoff(conn):
    llm = FakeLLM(script=[])
    resp = run_turn(
        conn,
        llm,
        user_id="user_1",
        conversation_id="c1",
        message="There's an unauthorized charge on my account, I think it's fraud",
        trace_id="t1",
    )
    assert resp.status == "handoff"
    rows = conn.execute("SELECT category FROM cases WHERE user_id = 'user_1'").fetchall()
    assert rows[0]["category"] == "fraud_or_unauthorized"


def test_explicit_human_request_routes_to_handoff(conn):
    llm = FakeLLM(script=[])
    resp = run_turn(
        conn, llm, user_id="user_1", conversation_id="c1", message="I want to talk to a human agent", trace_id="t1"
    )
    assert resp.status == "handoff"


def test_model_emitted_handoff_also_files_a_case(conn):
    script = [{"type": "handoff", "reason": "complaint", "summary": "Customer unhappy with service quality."}]
    llm = FakeLLM(script=script)
    resp = run_turn(conn, llm, user_id="user_1", conversation_id="c1", message="general feedback", trace_id="t1")
    assert resp.status == "handoff"
    rows = conn.execute("SELECT category FROM cases WHERE user_id = 'user_1'").fetchall()
    assert rows[0]["category"] == "complaint"


def test_higher_risk_intent_wins_when_multiple_signals_present(conn):
    # Spec 3: "一条消息同时包含多个意图时，优先处理风险更高的投诉或争议。"
    llm = FakeLLM(script=[])
    resp = run_turn(
        conn,
        llm,
        user_id="user_1",
        conversation_id="c1",
        message="What's my balance? Also there's an unauthorized charge, possible fraud.",
        trace_id="t1",
    )
    assert resp.status == "handoff"
    rows = conn.execute("SELECT category FROM cases WHERE user_id = 'user_1'").fetchall()
    assert rows[0]["category"] == "fraud_or_unauthorized"
