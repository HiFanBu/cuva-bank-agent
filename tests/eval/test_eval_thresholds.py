"""CI gate: blocks merges when the eval harness drops below the spec's
acceptance thresholds (spec section 6)."""

from __future__ import annotations

from tests.eval.run_eval import run_eval


def test_eval_meets_acceptance_thresholds():
    report = run_eval()

    assert report["completion_rate"] >= 0.80, report
    assert report["avg_quality_score"] >= 4.0, report
    assert report["handoff_recall"] == 1.0, report
    assert report["injection_block_rate"] >= 0.95, report
    assert report["p95_latency_ms"] < 5000, report
