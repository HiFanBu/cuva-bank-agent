from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.guardrails.input_guard import redact_pii


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(
    conn: sqlite3.Connection,
    *,
    trace_id: str,
    user_id: str | None,
    event: str,
    tool: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Writes one audit row. `detail` is JSON-encoded and PII-redacted first
    so logs and any downstream model input never carry raw PII (spec 5)."""
    safe_detail = None
    if detail is not None:
        raw = json.dumps(detail, default=str)
        safe_detail = redact_pii(raw)
    conn.execute(
        """
        INSERT INTO audit_log (trace_id, user_id, event, tool, detail, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (trace_id, user_id, event, tool, safe_detail, _now()),
    )
    conn.commit()


def log_conversation_message(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
) -> None:
    conn.execute(
        """
        INSERT INTO conversation_messages (conversation_id, user_id, role, content, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (conversation_id, user_id, role, redact_pii(content), _now()),
    )
    conn.commit()
