"""Output-layer guardrails: citation enforcement, banned phrases, leak checks.

Runs on every `FinalStep` before it is returned to the user. A failure here
never lets the message through -- it is converted into a handoff so a human
reviews it, per the "无证据拒答" / no-fabrication requirement in the spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas import Citation

_BANNED_PATTERNS = re.compile(
    r"\b(i guarantee|guaranteed approval|we will waive|i('ll| will) waive the fee|"
    r"this (dispute|complaint) will (definitely|certainly) (win|be resolved in your favor)|"
    r"your (card|account) is definitely not (compromised|at risk))\b",
    re.IGNORECASE,
)

_POLICY_KEYWORDS = re.compile(
    r"\b(late fee|apr|interest rate|grace period|minimum payment|credit limit|annual fee|rewards program|liability)\b",
    re.IGNORECASE,
)


@dataclass
class OutputGuardResult:
    passed: bool
    reason: str | None = None


def check_final_answer(message: str, citations: list[Citation], used_policy_tool: bool) -> OutputGuardResult:
    if _BANNED_PATTERNS.search(message):
        return OutputGuardResult(passed=False, reason="banned_phrase")

    if used_policy_tool and not citations:
        return OutputGuardResult(passed=False, reason="missing_citation")

    if not used_policy_tool and _POLICY_KEYWORDS.search(message) and not citations:
        return OutputGuardResult(passed=False, reason="uncited_policy_claim")

    return OutputGuardResult(passed=True)
