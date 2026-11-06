"""Multi-turn conversations: prior turns reach the model, scoped to the
caller, and the demo FakeLLM can assemble a payment across several messages."""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.llm import FakeLLM, _extract_payment_slots
from app.main import app, get_db, get_llm

ALICE = {"Authorization": "Bearer token-alice"}
BOB = {"Authorization": "Bearer token-bob"}


def _client(conn) -> TestClient:
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    return TestClient(app)


def _say(client, text, headers=ALICE, conv="c_multi"):
    resp = client.post("/chat", headers=headers, json={"conversation_id": conv, "message": text})
    assert resp.status_code == 200
    return resp.json()


def test_vague_payment_request_asks_for_details_instead_of_handoff(conn):
    client = _client(conn)
    body = _say(client, "I can pay the bill")
    assert body["status"] == "answered"
    assert "how much" in body["message"].lower()
    assert conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 0


def test_payment_slots_collected_across_turns(conn):
    client = _client(conn)
    _say(client, "I want to make a payment")
    body = _say(client, "$250")
    assert body["status"] == "answered"
    assert "what date" in body["message"].lower()
    body = _say(client, "tomorrow")
    assert body["status"] == "needs_confirmation"
    row = conn.execute(
        "SELECT args FROM pending_actions WHERE action_id = ?", (body["action_id"],)
    ).fetchone()
    assert '"amount_cents": 25000' in row[0]
    assert (date.today() + timedelta(days=1)).isoformat() in row[0]


def test_changing_topic_mid_payment_flow_is_respected(conn):
    client = _client(conn)
    _say(client, "pay my card")
    body = _say(client, "actually what's my balance?")
    assert body["status"] == "answered"
    assert "balance" in body["message"].lower()


def test_other_users_conversation_history_is_not_used(conn):
    client = _client(conn)
    _say(client, "I want to pay $300", headers=ALICE, conv="shared_id")
    # Bob reuses Alice's conversation_id; Alice's $300 must not leak into his turn.
    body = _say(client, "tomorrow", headers=BOB, conv="shared_id")
    assert body["status"] != "needs_confirmation"
    assert conn.execute("SELECT COUNT(*) FROM pending_actions").fetchone()[0] == 0


def test_greeting_does_not_create_a_case(conn):
    client = _client(conn)
    body = _say(client, "hi")
    assert body["status"] == "answered"
    assert conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0] == 0


def test_policy_question_mentioning_payment_still_goes_to_policy(conn):
    client = _client(conn)
    body = _say(client, "How is my minimum payment calculated?")
    assert body["status"] == "answered"
    assert body["citations"]


def test_slot_extraction_does_not_read_dates_as_amounts():
    amount, pay_date = _extract_payment_slots("pay 500 on Dec 20")
    assert amount == 50000
    assert pay_date is not None and pay_date.endswith("-12-20")
    amount, pay_date = _extract_payment_slots("pay on 2026-12-15")
    assert amount is None
    assert pay_date == "2026-12-15"
