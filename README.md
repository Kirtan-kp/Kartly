# Kartly Support Agent

A production-style agentic AI system for e-commerce customer support: it triages tickets, retrieves policy, calls tools through a deterministic gateway, resolves low-risk issues autonomously, and escalates the rest to humans with a full summary.

Built to show the engineering around the model: **evals, routing, cost tracking, context management, guardrails, reliability, observability, and deployment.**

> Status: design complete. See `DESIGN.md` for the full design and `DECISIONS.md` for the ADR log.
> Real eval and cost numbers will be added here as the project is built.

## Key ideas

- **The model proposes, code disposes.** A Tool Gateway enforces policy, so prompt injection cannot cause unauthorized actions.
- **Risk tiers** decide autonomy: read-only, low-risk writes, money (limit plus approval), never-autonomous.
- **Routing** combines hard rules with a small classifier, then picks a small or large model tier.
- **Evals first:** state-based grading, per-layer metrics, CI smoke set, nightly full run.
- **Shadow mode first:** the agent drafts, a human sends, then autonomy expands per tier.
- **Zero-budget friendly:** local models plus hosted free tiers behind a provider-agnostic adapter.

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
    gateway/             # Tool Gateway: authz, limits, idempotency, audit
    tools/               # tool implementations over the mock store
    guards/              # injection screen, PII redaction, output validation
    store/               # DB models, migrations, seed data
    approvals/           # approval queue and decisions
    observability/       # tracing, metrics, replay
    admin/               # minimal server-rendered admin UI
  data/
    kb/                  # policy and FAQ documents
    seed/                # synthetic customers and orders
  evals/
    golden/              # tickets plus expected outcomes
    graders/             # state-based graders, LLM judge
    runs/                # experiment log, recorded LLM responses
    redteam/             # adversarial cases
  tests/
    unit/
    gateway_properties/
    load/
  .github/workflows/     # lint, types, tests, smoke eval, build, deploy
  docs/
    adr/                 # optional: split DECISIONS.md into one file per ADR
```

## Week 1 checklist

Goal: one ticket works end to end with a stored trace.

- [ ] Create repo, `pyproject.toml`, ruff and mypy config, pre-commit
- [ ] `docker-compose.yml` with Postgres (+pgvector) and Redis
- [ ] Schema and Alembic migrations for the mock store (`customers`, `orders`, `order_items`, `shipments`) and support tables (`tickets`, `messages`, `runs`, `run_steps`, `actions`, `audit_log`)
- [ ] Seed script for ~50 synthetic customers and ~200 orders in varied states
- [ ] Tool Gateway skeleton with `get_order` and `get_tracking`, including server-side `customer_id` injection and audit logging
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