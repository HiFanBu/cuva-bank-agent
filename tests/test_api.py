from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.llm import FakeLLM
from app.main import app, get_db, get_llm


def _client(conn, llm) -> TestClient:
    app.dependency_overrides[get_db] = lambda: conn
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app)


def test_health(conn):
    client = _client(conn, FakeLLM())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_requires_auth(conn):
    client = _client(conn, FakeLLM())
    resp = client.post("/chat", json={"conversation_id": "c1", "message": "hi"})
    assert resp.status_code == 401


def test_chat_policy_question_end_to_end(conn):
    client = _client(conn, FakeLLM())
    resp = client.post(
        "/chat",
        json={"conversation_id": "c1", "message": "How is the late fee charged?"},
        headers={"Authorization": "Bearer token-alice"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert len(body["citations"]) >= 1
    assert body["trace_id"].startswith("t_")


def test_chat_then_confirm_schedules_payment(conn):
    future = (date.today() + timedelta(days=5)).isoformat()
    script = [{"type": "tool_call", "tool": "schedule_payment", "args": {"amount_cents": 50000, "pay_date": future}}]
    client = _client(conn, FakeLLM(script=script))
    chat_resp = client.post(
        "/chat",
        json={"conversation_id": "c1", "message": "pay $500"},
        headers={"Authorization": "Bearer token-alice"},
    )
    assert chat_resp.json()["status"] == "needs_confirmation"
    action_id = chat_resp.json()["action_id"]

    confirm_resp = client.post(
        "/chat/confirm",
        json={"conversation_id": "c1", "action_id": action_id},
        headers={"Authorization": "Bearer token-alice"},
    )
    assert confirm_resp.status_code == 200
    assert confirm_resp.json()["status"] == "answered"


def test_confirm_by_different_user_is_rejected(conn):
    future = (date.today() + timedelta(days=5)).isoformat()
    script = [{"type": "tool_call", "tool": "schedule_payment", "args": {"amount_cents": 50000, "pay_date": future}}]
    client = _client(conn, FakeLLM(script=script))
    chat_resp = client.post(
        "/chat",
        json={"conversation_id": "c1", "message": "pay $500"},
        headers={"Authorization": "Bearer token-alice"},
    )
    action_id = chat_resp.json()["action_id"]

    confirm_resp = client.post(
        "/chat/confirm",
        json={"conversation_id": "c1", "action_id": action_id},
        headers={"Authorization": "Bearer token-bob"},
    )
    assert confirm_resp.status_code == 400


def test_metrics_endpoint_reports_after_requests(conn):
    client = _client(conn, FakeLLM())
    client.post(
        "/chat",
        json={"conversation_id": "c1", "message": "How is the late fee charged?"},
        headers={"Authorization": "Bearer token-alice"},
    )
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] >= 1
