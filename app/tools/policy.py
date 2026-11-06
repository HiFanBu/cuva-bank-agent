from __future__ import annotations

import sqlite3

from app.rag.retriever import PolicyRetriever
from app.schemas import SearchPolicyArgs, ToolResult

# Only these doc_ids are approved for retrieval (spec: "只用批准文档，按权限过滤").
# All Phase 1 content is public T&C sample text; a real deployment would
# scope this per product/jurisdiction.
APPROVED_DOC_IDS = {"sample_tos"}


def search_policy(user_id: str, args: SearchPolicyArgs, conn: sqlite3.Connection) -> ToolResult:
    retriever = PolicyRetriever(conn, approved_doc_ids=APPROVED_DOC_IDS)
    results = retriever.search(args.query)
    chunks = [
        {
            "doc_id": r.doc_id,
            "section": r.section,
            "text": r.text,
            "score": r.score,
        }
        for r in results
    ]
    return ToolResult(ok=True, data={"chunks": chunks})
