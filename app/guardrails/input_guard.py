"""Input-layer guardrails: injection detection, risk routing, PII redaction.

These run BEFORE the agent loop and BEFORE anything is sent to the model.
Per the spec, user text and retrieved documents are always data, never
instructions -- this module never grants extra tool permissions, it only
decides whether to (a) refuse outright, (b) route straight to human handoff,
or (c) let the agent loop proceed normally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# --- Prompt injection ------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) instructions",
    r"disregard (all |any )?(previous|prior|above) (instructions|rules)",
    r"you are now (in )?(admin|developer|root|unrestricted) mode",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (if you (have|had) )?no (restrictions|rules|filters)",
    r"pretend (you are|to be) (an? )?(unfiltered|unrestricted|jailbroken)",
    r"system\s*:\s*",
    r"\boverride\b.*\b(permission|authorization|user_?id)\b",
    r"user_id\s*[=:]\s*['\"]?user_?\d+",
    r"transfer (all|my) (money|funds|balance) to",
    r"set (my )?user_?id to",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


# --- High-risk intent routing (must-handoff list from the spec) -----------

HandoffReasonLiteral = Literal[
    "user_request",
    "fraud_or_unauthorized",
    "billing_dispute",
    "complaint",
    "vulnerable_customer",
]

_FRAUD_PATTERNS = re.compile(
    r"\b(fraud(ulent)?|unauthorized (charge|transaction)|stolen card|someone used my card|didn'?t make this charge)\b",
    re.IGNORECASE,
)
_DISPUTE_PATTERNS = re.compile(
    r"\b(dispute|disputed|charged twice|duplicate charge|double charged|wrong amount|charged.*incorrect(ly)?)\b",
    re.IGNORECASE,
)
_COMPLAINT_PATTERNS = re.compile(
    r"\b(complain|complaint|this is unacceptable|terrible service|file a complaint|lawsuit|sue you|regulator|cfpb)\b",
    re.IGNORECASE,
)
_HUMAN_REQUEST_PATTERNS = re.compile(
    r"\b(talk to a human|speak to a (person|human|representative|agent)|human agent|real person|customer service rep)\b",
    re.IGNORECASE,
)
_VULNERABLE_PATTERNS = re.compile(
    r"\b(i can'?t afford|can'?t pay (my )?rent|losing my home|on a fixed income|i'?m (elderly|disabled)|"
    r"my (caregiver|guardian|power of attorney)|i'?m (scared|desperate|suicidal)|financial hardship)\b",
    re.IGNORECASE,
)

_RISK_ORDER: list[tuple[re.Pattern, HandoffReasonLiteral]] = [
    (_FRAUD_PATTERNS, "fraud_or_unauthorized"),
    (_DISPUTE_PATTERNS, "billing_dispute"),
    (_COMPLAINT_PATTERNS, "complaint"),
    (_VULNERABLE_PATTERNS, "vulnerable_customer"),
    (_HUMAN_REQUEST_PATTERNS, "user_request"),
]


def classify_must_handoff(text: str) -> HandoffReasonLiteral | None:
    """Highest-risk match wins when multiple intents are present (spec 3)."""
    for pattern, reason in _RISK_ORDER:
        if pattern.search(text):
            return reason
    return None


# --- PII redaction (applied before logging / before sending to the model) -

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD_NUMBER]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> str:
    redacted = text
    for pattern, replacement in _PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


@dataclass
class InputGuardResult:
    injection_detected: bool
    must_handoff_reason: HandoffReasonLiteral | None
    sanitized_message: str


def run_input_guard(message: str) -> InputGuardResult:
    return InputGuardResult(
        injection_detected=detect_prompt_injection(message),
        must_handoff_reason=classify_must_handoff(message),
        sanitized_message=message,
    )
