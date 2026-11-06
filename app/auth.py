"""Server-side identity resolution.

The spec requires that `user_id` is derived from authentication on the server
and never taken from the request body or produced by the model. This module
is a deliberately simple stand-in for a real session/JWT layer: a fixed map
from bearer token to user_id. Swapping this for real auth (JWT/OIDC) does not
change any call site, because every tool and route only ever sees `user_id`
through `get_current_user_id`.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

# Demo-only token -> user_id map. Never source of truth in production.
_DEMO_TOKENS: dict[str, str] = {
    "token-alice": "user_1",
    "token-bob": "user_2",
}


def get_current_user_id(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    user_id = _DEMO_TOKENS.get(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id
