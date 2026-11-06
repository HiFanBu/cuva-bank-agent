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


def load_recent_messages(
    conn: sqlite3.Connection,
    *,
    conversation_id: str,
    user_id: str,
    limit: int = 10,
) -> list[dict[str, str]]:
    """Prior turns of this conversation, oldest first, for model context.

    Scoped by `user_id` as well as `conversation_id`, so a caller who
    guesses or reuses another customer's conversation_id gets nothing back.
    Content is already PII-redacted at write time (see
    `log_conversation_message`), so it is safe to hand to the model.
    """
    rows = conn.execute(
        """
        SELECT role, content FROM conversation_messages
        WHERE conversation_id = ? AND user_id = ?
        ORDER BY id DESC LIMIT ?
        """,
        (conversation_id, user_id, limit),
    ).fetchall()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
