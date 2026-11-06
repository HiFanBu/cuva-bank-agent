# CuVA-style Credit Card AI Assistant (Phase 1)

A from-scratch recreation of a Barclays-CuVA-style credit card support agent,
built with public knowledge and simulated data only. No Barclays code, data,
or internal details are used anywhere in this repo.

Goal: let an already-authenticated credit card customer self-serve common
requests (policy questions, balance/transactions, scheduling a payment,
filing a complaint) through a hand-written agent loop, with safety rules
enforced in code rather than in a prompt.

## Status

**Phase 1** (this repo): a single Python service -- FastAPI + Pydantic +
SQLite, a hand-written agent loop, TF-IDF retrieval over sample policy text,
and a pytest-based eval harness with a fixed fake model. This is the phase
described in the spec as needing to be solid before any service-splitting.

Phases 2 (MCP server/client) and 3 (Spring Boot account service) are
designed but not implemented here -- see section 9 of the spec for the
target architecture.

## Architecture

```
app/
  main.py          FastAPI app: POST /chat, POST /chat/confirm, GET /health, GET /metrics
  agent.py         Hand-written agent loop (<= 6 steps), confirmation flow
  llm.py           Model access layer -- FakeLLM (tests/eval), Ollama, Bedrock (stub)
  schemas.py       ChatRequest/Response, AgentStep, per-tool arg schemas (extra="forbid")
  auth.py          Server-side user_id resolution from a bearer token
  db.py            SQLite schema: accounts, transactions, payments, pending_actions,
                    cases, audit_log, documents, conversation_messages
  audit.py         Audit log + conversation log writer (PII-redacted before writing)
  metrics.py       In-memory latency/cost/error metrics for GET /metrics
  tools/           search_policy, get_balance, get_transactions, schedule_payment, create_case
  rag/             TF-IDF retriever + policy text ingestion (no external embedding API)
  guardrails/      input_guard.py (injection + risk routing + PII redaction),
                    output_guard.py (citation + banned-phrase checks)
  data/            Seed data (demo accounts/transactions)
tests/
  test_*.py        Unit/integration tests for each spec requirement
  eval/            cases.jsonl + run_eval.py + test_eval_thresholds.py (CI gate)
```

## Why these security choices are structural, not prompted

- **`user_id` never comes from the model or the request body.** `auth.py`
  resolves it from the bearer token; every tool function takes `user_id` as
  a plain Python parameter supplied by the agent loop, never read out of
  the model's tool-call `args`.
- **Every tool-args schema uses `extra="forbid"`.** If a model (or an
  injected instruction) tries to smuggle `user_id` or any other field into
  a tool call, Pydantic validation fails and the loop gets back an error
  observation -- it never silently executes with the wrong identity.
- **Money only moves in one place.** `schedule_payment` inside the agent
  loop only validates the request and creates a `pending_actions` row
  (`needs_confirmation`). `execute_scheduled_payment` -- the only code path
  that writes to `payments` -- is called exclusively from `/chat/confirm`,
  after ownership and expiry checks, and is idempotent on `action_id`.
- **Retrieved documents and user text are always data.** Nothing in this
  codebase parses tool output or retrieved chunks for instructions; see
  `tests/test_guardrails.py::test_retrieved_document_content_is_never_treated_as_instructions`.
- **Complaints/disputes/fraud are routed before the model is ever called.**
  `agent.run_turn` checks `guardrails/input_guard.py`'s risk classifier
  first; a must-handoff signal short-circuits straight to a filed case, so
  recall on these categories doesn't depend on model behavior at all.

## Running it

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn app.main:app --reload
```

Demo auth tokens (see `app/auth.py`): `Bearer token-alice` -> `user_1`,
`Bearer token-bob` -> `user_2`.

```bash
curl -X POST localhost:8000/chat \
  -H "Authorization: Bearer token-alice" -H "Content-Type: application/json" \
  -d '{"conversation_id":"c1","message":"How is the late fee charged?"}'
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Testing & eval

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pytest
```

This runs the full unit/integration suite plus `tests/eval/test_eval_thresholds.py`,
which fails the build if the eval harness drops below the spec's thresholds
(task completion >= 80%, dispute/complaint handoff recall == 100%,
prompt-injection block rate >= 95%, p95 latency < 5s). Run the harness
directly for a metrics printout:

```bash
./.venv/bin/python -m tests.eval.run_eval
```

## Swapping the model

Everything goes through `app/llm.py::get_llm_provider`, selected by the
`LLM_PROVIDER` env var:

- `fake` (default): deterministic `FakeLLM`, used in tests/eval and safe for
  local exploration without any model dependency.
- `ollama`: local open-source model via Ollama's HTTP API (`OLLAMA_BASE_URL`,
  `OLLAMA_MODEL`).
- `bedrock`: AWS Bedrock Converse API -- stubbed as a placeholder
  (`BedrockProvider.generate_step` raises `NotImplementedError`); wire up
  `boto3`'s `bedrock-runtime` client here for production use.

## Known Phase 1 simplifications

- Retrieval is hand-written TF-IDF/cosine similarity, not a neural
  embedding model -- adequate for a small, approved policy-doc corpus and
  keeps Phase 1 free of any external embedding API dependency.
- "Answer quality" in the eval harness is a cheap keyword+citation proxy,
  not a human or LLM-judge score.
- Auth is a fixed demo token map, standing in for real session/JWT auth.
- SQLite + an in-memory metrics collector are fine for a single-process
  demo; the Payment high-concurrency design in spec section 8 (Redis-backed
  pending actions, outbox pattern, read replicas, rate limiting/circuit
  breaking) is designed but not built here.
