"""In-memory metrics collector backing GET /metrics.

Phase 1 keeps this in-process (no Prometheus/StatsD dependency) since the
spec only asks for p95 latency, token/cost tracking, and error/handoff rate
to be visible, not a full observability stack.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Record:
    latency_ms: float
    status: str
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int


class MetricsCollector:
    def __init__(self, max_records: int = 2000):
        self._lock = threading.Lock()
        self._records: list[_Record] = []
        self._max_records = max_records

    def record(
        self,
        *,
        latency_ms: float,
        status: str,
        cost_usd: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._records.append(
                _Record(latency_ms, status, cost_usd, prompt_tokens, completion_tokens)
            )
            if len(self._records) > self._max_records:
                self._records.pop(0)

    def snapshot(self) -> dict:
        with self._lock:
            records = list(self._records)
        if not records:
            return {
                "total_requests": 0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "error_rate": 0.0,
                "handoff_rate": 0.0,
                "total_cost_usd": 0.0,
                "total_tokens": 0,
            }
        latencies = sorted(r.latency_ms for r in records)
        total = len(records)
        errors = sum(1 for r in records if r.status == "failed")
        handoffs = sum(1 for r in records if r.status == "handoff")
        return {
            "total_requests": total,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "error_rate": errors / total,
            "handoff_rate": handoffs / total,
            "total_cost_usd": sum(r.cost_usd for r in records),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in records),
        }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, math.ceil(pct * len(sorted_values)) - 1)
    return sorted_values[max(idx, 0)]


class Timer:
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


metrics = MetricsCollector()
