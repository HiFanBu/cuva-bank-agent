"""Seeds demo accounts/transactions for the two demo users (see app/auth.py)."""

from __future__ import annotations

import sqlite3

DEMO_ACCOUNTS = [
    {"user_id": "user_1", "balance_cents": 128475, "due_date": "2026-11-20"},
    {"user_id": "user_2", "balance_cents": 54230, "due_date": "2026-11-15"},
]

DEMO_TRANSACTIONS = [
    {"id": "t_1001", "user_id": "user_1", "amount_cents": 4599, "merchant": "Blue Bottle Coffee", "posted_at": "2026-10-28"},
    {"id": "t_1002", "user_id": "user_1", "amount_cents": 12000, "merchant": "Delta Airlines", "posted_at": "2026-10-25"},
    {"id": "t_1003", "user_id": "user_1", "amount_cents": 8999, "merchant": "Whole Foods", "posted_at": "2026-10-20"},
    {"id": "t_2001", "user_id": "user_2", "amount_cents": 2500, "merchant": "Netflix", "posted_at": "2026-10-27"},
    {"id": "t_2002", "user_id": "user_2", "amount_cents": 15000, "merchant": "Home Depot", "posted_at": "2026-10-22"},
]


def seed_accounts(conn: sqlite3.Connection) -> None:
    for a in DEMO_ACCOUNTS:
        conn.execute(
            """
            INSERT INTO accounts (user_id, balance_cents, due_date) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                balance_cents=excluded.balance_cents, due_date=excluded.due_date
            """,
            (a["user_id"], a["balance_cents"], a["due_date"]),
        )
    for t in DEMO_TRANSACTIONS:
        conn.execute(
            """
            INSERT INTO transactions (id, user_id, amount_cents, merchant, posted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                amount_cents=excluded.amount_cents, merchant=excluded.merchant, posted_at=excluded.posted_at
            """,
            (t["id"], t["user_id"], t["amount_cents"], t["merchant"], t["posted_at"]),
        )
    conn.commit()
