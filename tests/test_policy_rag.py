from __future__ import annotations

from app.agent import run_turn
from app.llm import FakeLLM


def test_policy_question_returns_answer_with_citation(conn):
    llm = FakeLLM()  # heuristic mode
    resp = run_turn(
        conn,
        llm,
        user_id="user_1",
        conversation_id="c1",
        message="How is the late fee charged?",
        trace_id="t1",
    )
    assert resp.status == "answered"
    assert len(resp.citations) >= 1
    assert resp.citations[0].doc_id == "sample_tos"


def test_no_evidence_refuses_instead_of_fabricating(conn):
    # search_policy finds nothing for this nonsense query; if the model still
    # tries to answer without citations, the output guard must convert that
    # into a handoff rather than let a fabricated answer through.
    script = [
        {"type": "tool_call", "tool": "search_policy", "args": {"query": "quantum teleportation refund policy"}},
        {"type": "final", "message": "Sure, teleportation refunds are allowed under section 12.", "citations": []},
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(
        conn,
        llm,
        user_id="user_1",
        conversation_id="c2",
        message="Are teleportation refunds covered?",
        trace_id="t2",
    )
    assert resp.status == "handoff"


def test_output_guard_rejects_uncited_policy_claim(conn):
    script = [
        {
            "type": "final",
            "message": "Your APR and late fee policy says this.",
            "citations": [],
        }
    ]
    llm = FakeLLM(script=script)
    resp = run_turn(
        conn,
        llm,
        user_id="user_1",
        conversation_id="c3",
        message="Tell me about late fee and APR",
        trace_id="t3",
    )
    assert resp.status == "handoff"
