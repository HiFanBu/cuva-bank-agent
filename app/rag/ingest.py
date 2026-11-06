"""Seeds the `documents` table with sample credit card policy text.

This text is written from scratch for this demo -- generic, publicly known
credit-card terms (late fees, APR, grace period, disputes, etc.) phrased in
our own words. It is NOT copied from Barclays, Capital One, or any other
issuer's actual cardholder agreement; a real deployment would replace this
with the issuer's approved, legally-reviewed documents.

A tf-idf "embedding" is stored per chunk mainly so the `documents.embedding`
column (as specified) is populated and inspectable; PolicyRetriever
recomputes vectors in-memory from `text` at query time (see retriever.py).
"""

from __future__ import annotations

import json
import sqlite3

from app.rag.retriever import tokenize

POLICY_CHUNKS: list[dict[str, str]] = [
    {
        "doc_id": "sample_tos",
        "chunk_id": "late_fees",
        "section": "Late Fees",
        "text": (
            "If we do not receive at least your minimum payment by the due date, "
            "a late fee may be charged on your next statement. The fee is up to "
            "$29 for a first late payment and up to $40 for a second late payment "
            "within the following six billing cycles. The late fee will never "
            "exceed your minimum payment amount."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "apr_interest",
        "section": "APR and Interest Charges",
        "text": (
            "Your Annual Percentage Rate (APR) for purchases is a variable rate "
            "tied to the Prime Rate and is shown on your account summary. Interest "
            "is calculated using the average daily balance method and is charged "
            "unless you pay your full statement balance by the due date each month."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "grace_period",
        "section": "Grace Period",
        "text": (
            "You will not be charged interest on new purchases if you pay your "
            "entire statement balance, including any previous balance, in full "
            "by the due date every billing period. There is no grace period on "
            "cash advances or balance transfers; interest accrues from the "
            "transaction date."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "minimum_payment",
        "section": "Minimum Payment",
        "text": (
            "Your minimum payment is the greater of $25 or 1% of your new balance "
            "plus billed interest and fees, rounded to the nearest dollar. Paying "
            "only the minimum payment will increase the amount of interest you "
            "pay and the time it takes to repay your balance."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "credit_limit",
        "section": "Credit Limit",
        "text": (
            "Your credit limit is shown on your monthly statement and in the "
            "mobile app. Charges that would put your balance over your credit "
            "limit may be declined. We may periodically review your account and "
            "adjust your credit limit based on your payment history and "
            "creditworthiness."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "annual_fee",
        "section": "Annual Fee",
        "text": (
            "Some card products charge an annual fee, disclosed at account "
            "opening and shown on your account summary. The annual fee, if any, "
            "is billed once per cardmember year and is non-refundable except as "
            "required by applicable law."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "rewards",
        "section": "Rewards Program",
        "text": (
            "Eligible purchases earn rewards points at the rate shown in your "
            "rewards program terms. Points are forfeited if your account is "
            "closed for any reason or becomes more than 60 days past due. "
            "Returns and refunds reverse the rewards earned on that purchase."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "billing_disputes",
        "section": "Billing Disputes and Errors",
        "text": (
            "If you believe there is an error on your statement, including an "
            "unauthorized or duplicate charge, contact us in writing within 60 "
            "days of the statement date. We will investigate and you will not "
            "have to pay the disputed amount while we investigate. Disputes are "
            "reviewed by our billing team, not resolved automatically."
        ),
    },
    {
        "doc_id": "sample_tos",
        "chunk_id": "lost_stolen_liability",
        "section": "Lost or Stolen Card Liability",
        "text": (
            "Report a lost or stolen card immediately. You will not be held "
            "responsible for unauthorized charges made after you report your "
            "card lost or stolen, consistent with applicable zero-liability "
            "protections."
        ),
    },
]


def seed_documents(conn: sqlite3.Connection, chunks: list[dict[str, str]] | None = None) -> None:
    chunks = chunks if chunks is not None else POLICY_CHUNKS
    for c in chunks:
        tokens = tokenize(c["text"])
        fake_embedding = json.dumps(sorted(set(tokens)))
        conn.execute(
            """
            INSERT INTO documents (doc_id, chunk_id, section, text, embedding)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id, chunk_id) DO UPDATE SET
                section=excluded.section, text=excluded.text, embedding=excluded.embedding
            """,
            (c["doc_id"], c["chunk_id"], c["section"], c["text"], fake_embedding),
        )
    conn.commit()
