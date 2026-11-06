from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from app.schemas import CreateCaseArgs, ToolResult


def create_case(user_id: str, args: CreateCaseArgs, conn: sqlite3.Connection) -> ToolResult:
    """Per spec: create_case never decides the outcome of a complaint/dispute
    -- it only files it for a human. The agent loop always converts a
    create_case tool call into a `handoff` response; this tool just persists
    the case record."""
    case_id = f"case_{uuid.uuid4().hex[:12]}"
    conn.execute(
        """
        INSERT INTO cases (id, user_id, category, summary, status, created_at)
        VALUES (?, ?, ?, ?, 'open', ?)
        """,
        (case_id, user_id, args.category, args.summary, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    return ToolResult(ok=True, data={"case_id": case_id})
