# Kartly Support Agent

A production-style agentic AI system for e-commerce customer support: it triages tickets, retrieves policy, calls tools through a deterministic gateway, resolves low-risk issues autonomously, and escalates the rest to humans with a full summary.

Built to show the engineering around the model: **evals, routing, cost tracking, context management, guardrails, reliability, observability, and deployment.**

> Status: **Week 1 in progress** (database layer, Tool Gateway, LLM adapter and the bounded agent loop done; the HTTP API and the first evals are next).
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

## LLM providers

Models are configured in `config/models.yaml`: providers (base URL, API-key variable, quirks), models (provider model ID, reference price), and roles (`small`, `large`) that list a fallback order. Both providers speak the OpenAI chat-completions protocol, so one client covers them.

**Groq (free tier):** put your key in `.env` (never in a file you commit):

```bash
cp .env.example .env     # then edit .env and set GROQ_API_KEY=...
```

**Ollama (local):** pull a small tool-calling model, then create the `kartly-small` alias, which sets an 8k context window (Ollama's default is too small for our prompts):

```bash
ollama pull qwen2.5:3b                                  # or any model from `ollama list`
ollama create kartly-small -f ollama/Modelfile.small    # edit FROM in the file to change the base model
ollama ps                                               # PROCESSOR should say 100% GPU; CPU offload means it is too big for 4 GB
```

Check that each model really works before building on it. The smoke test makes three real calls (plain reply, a tool call whose arguments must pass the real tool schema, and a tool-result round trip) and prints the tokens and shadow cost of each:

```bash
python -m kartly.llm.smoke --list
python -m kartly.llm.smoke local_small
python -m kartly.llm.smoke groq_small
```

Exit code 0 means all three steps passed, 1 means a step failed, 2 means a configuration problem (missing key, unknown model).

The first call to a local model loads it into memory and can take a minute or more (about 94 s on an RTX 3050 laptop); later calls take seconds, so run the smoke test twice before judging local latency.

Hosted lineups change: Groq retired `llama-3.1-8b-instant` for free accounts in August 2026 and the small tier is now `openai/gpt-oss-20b`, a reasoning model. If a model returns HTTP 404, check the provider's deprecations page and edit the `model:` line in `config/models.yaml` (ADR-025).

The client makes one request per call and never retries (ADR-023); every failure is a typed error saying whether it is retryable. Costs are **shadow costs** at the reference prices in `models.yaml`, each with a note on its source (ADR-013, ADR-024); the real spend is $0.

## Agent loop

`kartly.orchestrator` runs the one place an LLM makes decisions. `AgentLoop` talks to a model, lets it ask for tool calls, hands every call to the gateway, and always stops with a reason. `TicketRunner` wraps it with the database: it opens a run, stores each step as it happens, and closes the run.

Limits (`RunBudget`, checked before every model call): 6 model calls, 8 tool calls, 20,000 tokens, USD 0.30 shadow cost, 120 seconds. One repair re-prompt is allowed for a broken tool call or an empty reply. A repeated identical tool call is refused and a second repeat ends the run. Any stop other than an answer (`step_limit`, `tool_limit`, `token_limit`, `cost_limit`, `timeout`, `no_progress`, `malformed_output`, `truncated_reply`, `llm_error`, `gateway_error`) escalates the ticket to a human. See ADR-026.

Try it on a real ticket (needs the Quickstart database and one of the models above):

```bash
alembic upgrade head              # once after updating: adds wider cost columns (migration 0002)
python -m kartly.orchestrator.cli --model groq_small "Where is my order KRT-100002?"
python -m kartly.orchestrator.cli --model local_small --customer C-0002 "Where is KRT-100002?"
```

The second command asks customer C-0002 about an order that belongs to C-0001. The reply should say the order was not found, and the printed trace shows the gateway refusing the call. The command prints the stored trace (each model call and tool call with tokens, time and shadow cost) and the draft reply. Exit code 0 means a draft was produced, 1 means the ticket was escalated, 2 means a setup problem.

In shadow mode nothing is sent: the draft is stored as an agent message and the ticket becomes `drafted`.

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
    orchestrator/        # bounded loop, run budget, ticket runner, trace  <- built; stages come later
    routing/             # hard rules, classifier, model selection
    llm/                 # provider adapter, pricing, smoke test  <- built; retries and fallback in Week 5
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
- [x] LLM adapter (OpenAI-compatible) with config-driven models; Ollama as first backend, Groq second
- [x] Token counting and shadow-cost calculation from `config/models.yaml`
- [x] Minimal tool-calling loop with step and token caps
- [ ] `POST /v1/tickets` and `GET /v1/runs/{id}/trace`
- [ ] First 10 hand-written eval cases and a tiny state-based grader
- [ ] Benchmark 2-3 local models that fit 4 GB VRAM for tool-call validity (record results in `evals/runs/`)

## Ship gates

- Zero unauthorized money actions
- Tier 3 recall >= 99%
- No ticket lost under the chaos test
- Task success at or above target on the full golden set