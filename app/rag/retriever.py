"""Minimal, dependency-free retrieval over the `documents` table.

This implements TF-IDF + cosine similarity by hand (no numpy/sklearn) so
Phase 1 has zero external embedding-API dependency. It is intentionally
simple -- swapping in a real embedding model later only means changing
`PolicyRetriever` and the `documents.embedding` column's contents, not any
caller.

Retrieved text is always treated as *data* returned to the model, never as
instructions (see guardrails/input_guard.py and the prompt-injection tests) --
this module has no way to execute anything it retrieves.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    doc_id: str
    chunk_id: str
    section: str
    text: str
    score: float


class PolicyRetriever:
    def __init__(self, conn: sqlite3.Connection, approved_doc_ids: set[str] | None = None):
        rows = conn.execute(
            "SELECT doc_id, chunk_id, section, text FROM documents"
        ).fetchall()
        if approved_doc_ids is not None:
            rows = [r for r in rows if r["doc_id"] in approved_doc_ids]

        self._docs = [
            {
                "doc_id": r["doc_id"],
                "chunk_id": r["chunk_id"],
                "section": r["section"],
                "text": r["text"],
                "tokens": tokenize(r["text"]),
            }
            for r in rows
        ]
        self._idf = self._build_idf(self._docs)
        self._doc_vectors = [self._vectorize(d["tokens"]) for d in self._docs]

    @staticmethod
    def _build_idf(docs: list[dict]) -> dict[str, float]:
        n = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d["tokens"]))
        return {tok: math.log((n + 1) / (count + 1)) + 1.0 for tok, count in df.items()}

    def _vectorize(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {tok: freq * self._idf.get(tok, 1.0) for tok, freq in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {tok: w / norm for tok, w in vec.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        return sum(a[t] * b[t] for t in common)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.12) -> list[ScoredChunk]:
        if not self._docs:
            return []
        q_vec = self._vectorize(tokenize(query))
        scored = [
            ScoredChunk(
                doc_id=d["doc_id"],
                chunk_id=d["chunk_id"],
                section=d["section"],
                text=d["text"],
                score=self._cosine(q_vec, dv),
            )
            for d, dv in zip(self._docs, self._doc_vectors)
        ]
        scored = [s for s in scored if s.score >= min_score]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
