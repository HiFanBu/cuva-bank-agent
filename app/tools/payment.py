"""Payment-related tools.

`schedule_payment_stub` only *validates* a proposed payment inside the agent
loop -- it never moves money. The agent loop turns a valid call into a
`pending_actions` row and returns `needs_confirmation`; money only ever
moves in `execute_scheduled_payment`, which is invoked exclusively from the
`/chat/confirm` route after the confirmation/ownership/expiry checks in
agent.py. This split is what makes "未确认就动钱" structurally impossible
rather than merely discouraged by a prompt.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from app.config import settings
from app.schemas import GetTransactionsArgs, NoArgs, SchedulePaymentArgs, ToolResult


def get_balance(user_id: str, args: NoArgs, conn: sqlite3.Connection) -> ToolResult:
    row = conn.execute(
        "SELECT balance_cents, due_date FROM accounts WHERE user_id = ?", (user_id,)
    ).fetchone()
    if row is None:
        return ToolResult(ok=False, error="account_not_found")
    return ToolResult(
        ok=True,
        data={
            "balance_cents": row["balance_cents"],
            "due_date": row["due_date"],
            "as_of": datetime.now(timezone.utc).isoformat(),
        },
    )


def get_transactions(
    user_id: str, args: GetTransactionsArgs, conn: sqlite3.Connection
) -> ToolResult:
    rows = conn.execute(
        """
        SELECT id, amount_cents, merchant, posted_at FROM transactions
        WHERE user_id = ? ORDER BY posted_at DESC LIMIT ?
        """,
        (user_id, args.limit),
    ).fetchall()
    return ToolResult(
        ok=True,
        data={
            "transactions": [
                {
                    "id": r["id"],
                    "amount_cents": r["amount_cents"],
                    "merchant": r["merchant"],
                    "posted_at": r["posted_at"],
                }
                for r in rows
            ],
            "as_of": datetime.now(timezone.utc).isoformat(),
        },
    )


def schedule_payment_stub(
    user_id: str, args: SchedulePaymentArgs, conn: sqlite3.Connection
) -> ToolResult:
    if args.amount_cents > settings.max_payment_amount_cents:
        return ToolResult(ok=False, error="amount_exceeds_limit")
    account = conn.execute(
        "SELECT user_id FROM accounts WHERE user_id = ?", (user_id,)
    ).fetchone()
    if account is None:
        return ToolResult(ok=False, error="account_not_found")
    return ToolResult(
        ok=True,
        data={"amount_cents": args.amount_cents, "pay_date": args.pay_date.isoformat()},
    )


def execute_scheduled_payment(
    *, user_id: str, action_id: str, amount_cents: int, pay_date: date, conn: sqlite3.Connection
) -> ToolResult:
    """The only place money actually moves. Idempotent on `action_id`."""
    existing = conn.execute(
        "SELECT id, amount_cents, pay_date FROM payments WHERE idempotency_key = ?",
        (action_id,),
    ).fetchone()
    if existing is not None:
        return ToolResult(
            ok=True,
            data={
                "payment_id": existing["id"],
                "amount_cents": existing["amount_cents"],
                "pay_date": existing["pay_date"],
                "idempotent_replay": True,
            },
        )

    if amount_cents > settings.max_payment_amount_cents:
        return ToolResult(ok=False, error="amount_exceeds_limit")
    if pay_date < date.today():
        return ToolResult(ok=False, error="pay_date_in_past")

    payment_id = f"p_{action_id}"
    try:
        conn.execute(
            """
            INSERT INTO payments (id, user_id, amount_cents, pay_date, idempotency_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payment_id,
                user_id,
                amount_cents,
                pay_date.isoformat(),
                action_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Concurrent confirm race: another request already inserted this
        # idempotency key. Treat as a successful idempotent replay.
        conn.rollback()
        row = conn.execute(
            "SELECT id, amount_cents, pay_date FROM payments WHERE idempotency_key = ?",
            (action_id,),
        ).fetchone()
        return ToolResult(
            ok=True,
            data={
                "payment_id": row["id"],
                "amount_cents": row["amount_cents"],
                "pay_date": row["pay_date"],
                "idempotent_replay": True,
            },
        )

    return ToolResult(
        ok=True,
        data={
            "payment_id": payment_id,
            "amount_cents": amount_cents,
            "pay_date": pay_date.isoformat(),
            "idempotent_replay": False,
        },
    )
