# Kartly Support Agent

A production-style agentic AI system for e-commerce customer support: it triages tickets, retrieves policy, calls tools through a deterministic gateway, resolves low-risk issues autonomously, and escalates the rest to humans with a full summary.

Built to show the engineering around the model: **evals, routing, cost tracking, context management, guardrails, reliability, observability, and deployment.**

> Status: **Week 1 in progress** (database layer and Tool Gateway done; LLM adapter and agent loop next).
> See `DESIGN.md` for the full design and `DECISIONS.md` for the ADR log.
> Real eval and cost numbers will be added here as the project is built.

## Key ideas

- **The model proposes, code disposes.** A Tool Gateway enforces policy, so prompt injection cannot cause unauthorized actions.
- **Risk tiers** decide autonomy: read-only, low-risk writes, money (limit plus approval), never-autonomous.
- **Routing** combines hard rules with a small classifier, then picks a small or large model tier.
- **Evals first:** state-based grading, per-layer metrics, CI smoke set, nightly full run.
- **Shadow mode first:** the agent drafts, a human sends, then autonomy expands per tier.
- **Zero-budget friendly:** local models plus hosted free tiers behind a provider-agnostic adapter.

## Quickstart

Requires Python 3.11+ and Docker.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env               # Windows: copy .env.example .env

docker compose up -d               # Postgres (pgvector) and Redis
alembic upgrade head               # create the schema
python -m kartly.store.seed --reset   # ~50 customers, ~200 orders, 8 eval fixtures
```

Run the checks:

```bash
pytest tests/unit tests/gateway_properties   # fast, in-memory SQLite
ruff check . && ruff format --check . && mypy

# Postgres tests (migration, model/migration drift, audit-log immutability):
docker compose exec postgres createdb -U kartly kartly_test
export KARTLY_TEST_DATABASE_URL=postgresql+psycopg://kartly:kartly@localhost:5432/kartly_test
pytest                             # Windows PowerShell: $env:KARTLY_TEST_DATABASE_URL="..."
```

The Postgres tests drop and recreate the `public` schema, so they refuse to run unless the database name contains `test`. Without `KARTLY_TEST_DATABASE_URL` they are skipped.

## Mock store and eval fixtures

The seed is deterministic (fixed random seed). Eight hand-crafted orders have stable IDs so eval cases can refer to them:

| Order | Customer | Scenario |
|---|---|---|
| `KRT-100001` | C-0001 | Delivered 3 days ago, INR 2,898: a refund needs human approval |
| `KRT-100002` | C-0001 | In transit, on schedule: status and tracking questions |
| `KRT-100003` | C-0002 | Not shipped: cancel and address change allowed |
| `KRT-100004` | C-0002 | Delivered 45 days ago: outside the return window |
| `KRT-100005` | C-0003 | Contains a final-sale item: not returnable |
| `KRT-100006` | C-0003 | Delayed with a carrier exception: past its ETA |
| `KRT-100007` | C-0004 | INR 599, delivered: a refund is under the auto-approval threshold |
| `KRT-100008` | C-0004 | Partly refunded: INR 1,999 still refundable |

Orders belonging to different customers are there on purpose, for cross-customer access tests. Money is stored as integer paise; all data is synthetic.

## Tool Gateway

The model never touches the database. It asks for a tool call, and `kartly.gateway.ToolGateway` decides what happens:

```python
gateway = ToolGateway(session_factory)
specs = gateway.tool_specs(allowed=["get_order", "get_tracking"])  # what the model is shown
result = gateway.call(ctx, "get_order", {"order_id": "KRT-100001"})  # ctx = authenticated ticket
messages.append({"role": "tool", "content": result.to_model_content()})
```

Guarantees, each covered by tests:

- **Identity comes from `CallContext`, never from the model.** `customer_id` is not a tool argument, and supplying one is a validation error (and a logged security event), even if it names the real owner.
- **Another customer's order looks nonexistent.** Ownership is part of the query, and a foreign order and a missing order produce the same error to the model. Only the audit log records the difference.
- **Only allowed tools run.** Tools outside the routed intent's allowed set are rejected, and only allowed tools are shown to the model.
- **One call = one transaction.** The tool runs in a SAVEPOINT, so a failure undoes its writes, and exactly one `audit_log` row is written either way. If the audit row cannot be written, the call is rolled back and the error propagates: no audit record, no action.
- **Audit logging cannot be broken by hostile input.** Arguments are size-bounded and stripped of NUL characters, lone surrogates, NaN and huge integers, all of which Postgres JSONB rejects.
- **Tool output is pre-computed and minimal.** Amounts are rupee strings, "delayed" is computed in code, the address is city/state/pincode only, and lists are capped.

Properties are tested exhaustively over the seeded store (every order, owner vs strangers, both tools) and with Hypothesis on arbitrary arguments. See ADR-021 and ADR-022.

## Planned stack

Python, FastAPI, Postgres + pgvector, Redis (RQ or ARQ), Docker, GitHub Actions, OpenTelemetry, Ollama (local models), hosted free-tier LLM APIs behind one adapter.

## Planned repo layout

```
kartly-agent/
  README.md
  DESIGN.md
  DECISIONS.md
  pyproject.toml
  docker-compose.yml
  .env.example
  config/
    models.yaml          # model tiers, fallback chain, reference prices
    policy.yaml          # tiers, thresholds, caps
  src/kartly/
    api/                 # FastAPI routes, auth
    orchestrator/        # state machine, stages, run budget
    routing/             # hard rules, classifier, model selection
    llm/                 # provider adapter, rate-limit client, retries, pricing
    context/             # context builder, summarizer, memory
    retrieval/           # chunking, hybrid search, reranker
    gateway/             # Tool Gateway: authz, limits, idempotency, audit   <- built (read tools)
    tools/               # tool implementations over the mock store         <- built (read tools)
    guards/              # injection screen, PII redaction, output validation
    store/               # DB models, migrations, seed data        <- built
    approvals/           # approval queue and decisions
    observability/       # tracing, metrics, replay
    admin/               # minimal server-rendered admin UI
  data/
    kb/                  # policy and FAQ documents
  evals/
    golden/              # tickets plus expected outcomes
    graders/             # state-based graders, LLM judge
    runs/                # experiment log, recorded LLM responses
    redteam/             # adversarial cases
  tests/
    unit/
    integration/
    gateway_properties/
    load/
  .github/workflows/     # lint, types, tests (smoke eval, build, deploy added later)
```

## Week 1 checklist

Goal: one ticket works end to end with a stored trace.

- [x] Create repo, `pyproject.toml`, ruff and mypy config, pre-commit config
- [x] `docker-compose.yml` with Postgres (+pgvector) and Redis
- [x] Schema and Alembic migrations for the mock store (`customers`, `orders`, `order_items`, `shipments`) and support tables (`tickets`, `messages`, `runs`, `run_steps`, `actions`, `audit_log`), with an append-only audit log enforced by database triggers
- [x] Seed script for ~50 synthetic customers and ~200 orders in varied states, plus 8 fixture orders
- [x] Tool Gateway with `get_order` and `get_tracking`, server-side `customer_id` injection, savepoint-per-call, and a fail-closed audit log
- [ ] LLM adapter (OpenAI-compatible) with config-driven models; Ollama as first backend
- [ ] Token counting and shadow-cost calculation from `config/models.yaml`
- [ ] Minimal tool-calling loop with step and token caps
- [ ] `POST /v1/tickets` and `GET /v1/runs/{id}/trace`
- [ ] First 10 hand-written eval cases and a tiny state-based grader
- [ ] Benchmark 2-3 local models that fit 4 GB VRAM for tool-call validity (record results in `evals/runs/`)

## Ship gates

- Zero unauthorized money actions
- Tier 3 recall >= 99%
- No ticket lost under the chaos test
- Task success at or above target on the full golden set