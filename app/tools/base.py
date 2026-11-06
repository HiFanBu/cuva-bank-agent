"""Tool registry and the single server-side execution gate.

Every tool function's signature is `(user_id: str, args: <ArgsModel>, conn)`.
`user_id` always comes from the authenticated caller via the agent loop --
never from `args`, and `args` schemas use `extra="forbid"` so a model that
tries to smuggle a `user_id` field (or anything else) into tool args gets a
validation error instead of a silently-ignored field. This is what makes
"model never decides user_id" a code-level guarantee rather than a prompt
instruction.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.schemas import (
    CreateCaseArgs,
    GetTransactionsArgs,
    NoArgs,
    SchedulePaymentArgs,
    SearchPolicyArgs,
    ToolResult,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: type[BaseModel]
    risk: str
    requires_confirmation: bool
    handler: Callable[[str, BaseModel, sqlite3.Connection], ToolResult]


_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(spec: ToolSpec) -> None:
    _REGISTRY[spec.name] = spec


def get_tool(name: str) -> ToolSpec | None:
    return _REGISTRY.get(name)


def execute_tool(
    name: str, raw_args: dict[str, Any], *, user_id: str, conn: sqlite3.Connection
) -> ToolResult:
    spec = get_tool(name)
    if spec is None:
        return ToolResult(ok=False, error=f"unknown_tool:{name}")
    try:
        validated_args = spec.args_model.model_validate(raw_args)
    except ValidationError as e:
        return ToolResult(ok=False, error=f"invalid_args:{e.errors()[0]['msg'] if e.errors() else str(e)}")
    try:
        return spec.handler(user_id, validated_args, conn)
    except Exception as e:  # tool-level failure -> observation, never a crash
        return ToolResult(ok=False, error=f"tool_exception:{e}")


def _register_defaults() -> None:
    from app.tools.complaint import create_case
    from app.tools.payment import get_balance, get_transactions, schedule_payment_stub
    from app.tools.policy import search_policy

    register_tool(
        ToolSpec("search_policy", SearchPolicyArgs, "low", False, search_policy)
    )
    register_tool(
        ToolSpec("get_balance", NoArgs, "medium", False, get_balance)
    )
    register_tool(
        ToolSpec(
            "get_transactions", GetTransactionsArgs, "medium", False, get_transactions
        )
    )
    register_tool(
        ToolSpec(
            "schedule_payment",
            SchedulePaymentArgs,
            "high",
            True,
            schedule_payment_stub,
        )
    )
    register_tool(
        ToolSpec("create_case", CreateCaseArgs, "high", False, create_case)
    )


_register_defaults()
