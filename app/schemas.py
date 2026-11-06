from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Public API: /chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    conversation_id: str
    message: str = Field(min_length=1, max_length=4000)


class Citation(BaseModel):
    doc_id: str
    section: str


ChatStatus = Literal["answered", "needs_confirmation", "handoff", "failed"]


class ChatResponse(BaseModel):
    status: ChatStatus
    message: str
    citations: list[Citation] = Field(default_factory=list)
    action_id: Optional[str] = None
    trace_id: str


class ConfirmRequest(BaseModel):
    conversation_id: str
    action_id: str


# ---------------------------------------------------------------------------
# Model output contract: AgentStep (what the LLM is allowed to emit)
# ---------------------------------------------------------------------------


class ToolCallStep(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class FinalStep(BaseModel):
    type: Literal["final"] = "final"
    message: str
    citations: list[Citation] = Field(default_factory=list)


HandoffReason = Literal[
    "user_request",
    "fraud_or_unauthorized",
    "billing_dispute",
    "complaint",
    "low_confidence",
    "repeated_failure",
    "vulnerable_customer",
    "max_steps_exceeded",
    "invalid_model_output",
]


class HandoffStep(BaseModel):
    type: Literal["handoff"] = "handoff"
    reason: HandoffReason
    summary: str


AgentStep = ToolCallStep | FinalStep | HandoffStep

# ---------------------------------------------------------------------------
# Tool argument schemas (server validates these independent of the model)
#
# extra="forbid" on every one of these is load-bearing: it is what makes a
# model-supplied `user_id` (or any other smuggled field) a validation error
# instead of a silently-dropped field. Tool handlers only ever receive
# user_id as a function parameter from the authenticated caller, never from
# these models.
# ---------------------------------------------------------------------------


class NoArgs(BaseModel):
    model_config = {"extra": "forbid"}


class SearchPolicyArgs(BaseModel):
    model_config = {"extra": "forbid"}

    query: str = Field(min_length=1, max_length=500)


class GetTransactionsArgs(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(default=10, ge=1, le=50)


class SchedulePaymentArgs(BaseModel):
    model_config = {"extra": "forbid"}

    amount_cents: int = Field(gt=0)
    pay_date: date

    @field_validator("pay_date")
    @classmethod
    def pay_date_not_in_past(cls, v: date) -> date:
        if v < date.today():
            raise ValueError("pay_date cannot be before today")
        return v


class CreateCaseArgs(BaseModel):
    model_config = {"extra": "forbid"}

    category: str
    summary: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# Internal: tool execution result fed back to the agent loop as an observation
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
