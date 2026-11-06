from __future__ import annotations

import sqlite3
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from app.agent import AgentError, confirm_payment, run_turn
from app.auth import get_current_user_id
from app.data.seed import seed_accounts
from app.db import get_connection, init_db
from app.llm import LLMProvider, get_llm_provider
from app.metrics import Timer, metrics
from app.rag.ingest import seed_documents
from app.schemas import ChatRequest, ChatResponse, ConfirmRequest

_connection: sqlite3.Connection | None = None
_llm: LLMProvider | None = None


def get_db() -> sqlite3.Connection:
    assert _connection is not None, "DB not initialized"
    return _connection


def get_llm() -> LLMProvider:
    assert _llm is not None, "LLM not initialized"
    return _llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _connection, _llm
    _connection = get_connection()
    init_db(_connection)
    seed_accounts(_connection)
    seed_documents(_connection)
    _llm = get_llm_provider()
    yield


app = FastAPI(title="CuVA-style Credit Card Agent (Phase 1)", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics() -> dict:
    return metrics.snapshot()


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    conn: sqlite3.Connection = Depends(get_db),
    llm: LLMProvider = Depends(get_llm),
) -> ChatResponse:
    trace_id = f"t_{uuid.uuid4().hex[:12]}"
    with Timer() as t:
        response = run_turn(
            conn,
            llm,
            user_id=user_id,
            conversation_id=req.conversation_id,
            message=req.message,
            trace_id=trace_id,
        )
    metrics.record(latency_ms=t.elapsed_ms, status=response.status)
    return response


@app.post("/chat/confirm", response_model=ChatResponse)
def chat_confirm(
    req: ConfirmRequest,
    user_id: str = Depends(get_current_user_id),
    conn: sqlite3.Connection = Depends(get_db),
) -> ChatResponse:
    trace_id = f"t_{uuid.uuid4().hex[:12]}"
    error: AgentError | None = None
    response: ChatResponse | None = None
    with Timer() as t:
        try:
            response = confirm_payment(
                conn,
                user_id=user_id,
                conversation_id=req.conversation_id,
                action_id=req.action_id,
                trace_id=trace_id,
            )
        except AgentError as e:
            error = e
    if error is not None:
        metrics.record(latency_ms=t.elapsed_ms, status="failed")
        raise HTTPException(status_code=400, detail=str(error)) from error
    metrics.record(latency_ms=t.elapsed_ms, status=response.status)
    return response
