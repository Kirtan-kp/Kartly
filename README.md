# Kartly Support Agent

An agentic AI system for e-commerce customer support, built to show the engineering around the model: it triages tickets, retrieves policy, calls tools through a deterministic gateway, resolves low-risk issues on its own, and hands everything else to a person with a summary. The data is synthetic and the models are free or local (Ollama, Groq's free tier), so everything here can be run and checked for free.

> **Status:** all eight weeks are built except the Week 8 write-up and demo script. The part that matters most, money safety, is measured: 15 write and red-team cases pass against a real hosted model with **0 safety failures**, and a scripted-model suite plus Postgres concurrency tests back that up. See [What is proven, and how](#what-is-proven-and-how) and [Known limits](#known-limits).
> Try it: `python -m kartly.demo` (no key needed). Design: `DESIGN.md`. Decisions with their trade-offs: `DECISIONS.md` (ADR-001 to ADR-062). Deployment: `docs/DEPLOY.md`.

## Try it in two minutes

```bash
python -m kartly.demo
```

No API key, no Docker, no network. Six scenes run on the real stack (injection screen, Tier 3 rules, agent loop, Tool Gateway, approvals, kill switch, audit log); only the model is a stand-in, so the output is the same every time:

1. an ordinary order question gets a drafted reply;
2. a prompt injection is stopped before any model call;
3. a legal threat goes to a person, never to the model;
4. a small refund (INR 599, under the INR 1,000 limit) is paid once and audited;
5. a large refund (INR 2,898) waits for a reviewer, runs once on approval, and a second click does nothing;
6. with the agent switched off, a new ticket goes straight to a person.

The demo shows what the code around the model does, not how good the model is. For that, run the evals against a real model (see Evals). For the full local stack with the admin page, see Run it in containers; the failure-handling version is `python -m kartly.chaos`.

## How it works

```
 customer message
        |
        v
  [ injection screen ]---hit---> human (no model call)
        |
  [ Tier 3 rules + router ]--legal / injury / distress--> human
        |
        v
  +-----------------------------+      +----------------------------------+
  |  bounded agent loop         |      |  Tool Gateway (code, not model)  |
  |  model proposes tool calls -+----->|  authz: customer_id injected     |
  |  context budget, step cap   |      |  idempotency, limits, audit log  |
  +-----------------------------+      |  write modes: off|shadow|auto   |
        |                              |  refunds > limit -> approval     |
        v                              +----------------+-----------------+
  [ grounding check ]                                   |
        |                                        mock store (Postgres)
        v
  draft reply / awaiting approval / escalation ---> reviewers (admin page, /admin)

 around it: job queue + workers, retries and fallback models, kill switch,
 metrics and alerts, cost report, evals, CI, containers
```

## Key ideas

- **The model proposes, code disposes.** A Tool Gateway enforces policy, so prompt injection cannot cause unauthorized actions: it injects the customer id itself, caps refunds, binds approvals to the exact payload and writes an append-only audit log.
- **Risk tiers** decide autonomy: read-only, low-risk writes, money (limit plus approval), never-autonomous.
- **Routing** combines hard rules with a small classifier, then picks a small or large model tier.
- **Evals first:** state-based grading (what changed in the store, before what the reply says), per-layer metrics, a red-team set for the money path.
- **Shadow mode first:** the agent drafts, a human sends, then autonomy expands per tier; a kill switch stops it without a deploy.
- **Reliability you can test:** a database job queue with leases and dead letters, retries with backoff, a fallback chain with circuit breakers, a fault-injection run and replay of stored runs.
- **Zero-budget friendly:** local models plus hosted free tiers behind a provider-agnostic adapter, with shadow cost computed per run.

## What is proven, and how

| Claim | Evidence | Where |
|---|---|---|
| A refund never exceeds what is left, and nothing above the limit runs without a human | Gateway property tests; concurrent refunds on Postgres; 8 red-team cases (fake manager approval, split refunds, fake system notices, another customer's order, ...) pass on a real model with 0 safety failures | ADR-046, ADR-047, ADR-049; `evals/golden/50_approval_bypass.yaml` |
| Injection cannot reach another customer's data | The gateway ignores model-supplied customer ids; cross-customer cases; an injection screen as a tripwire only | ADR-030 |
| The agent survives faults | Chaos run injects errors and crashes; no ticket lost; stuck jobs are reclaimed; dead letters can be requeued | ADR-052, ADR-053 |
| Retrieval is good enough | 74 questions over a 44-chunk knowledge base; dense retrieval ranks the right chunk first on 93.5% (the question set is easy; ADR-036 says so) | ADR-035, ADR-036 |
| Costs are visible | Shadow pricing per model and routing tier, per ticket, with what-if repricing | ADR-058 |
| It can be operated | Metrics, alerts, an admin page, backups that are read back and a tested restore, upgrade and rollback | ADR-054 to ADR-061 |

Real-model numbers so far (Groq, `groq_small`, one run per case): the 7 write cases and 8 red-team cases all pass, about 3,000 tokens and about $0.0003 of shadow cost per run, p50 latency about 2 s. Two things are worth knowing about how that was reached. A case first failed because it assumed the model would ask for the full amount; the model sensibly asked for what was left instead, so the case now accepts both safe behaviours (`also_acceptable`). And the first grader could not compare numeric tool arguments, which only showed once a real model sent a number. Both were bugs in the test, found by running a real model, and are fixed and tested.

## Known limits

- One run per case so far: a pass is not a rate. Use `--repeat 3` to see variance; the write-up will report repeated runs.
- The quality of replies is judged by phrase matching and a grounding check, not by a strong LLM judge.
- Free-tier models are small: they are fine for the routing and tool use measured here, and the policy answers lean on retrieval and code (verdicts such as return eligibility are computed, not generated).
- Single machine deployment: no high availability, no rate limiting or per-caller authentication (one shared API key), dependencies not locked by hash. `docs/DEPLOY.md` lists these.
- Everything is synthetic; no real customer data and no real payment system.

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

Browser tests of the admin page run in Chromium and are skipped when it is not installed: `pip install -e ".[dev]"` then `playwright install chromium`, and `pytest tests/browser`. Plain `pytest` includes them once the browser is there; add `-m "not browser"` to leave them out.

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

## HTTP API

`python -m kartly.api` serves the API on `127.0.0.1:8000` (use `--port` to change it; interactive docs at `/docs`). Every `/v1` route needs an `X-API-Key` header; `/healthz` does not.

| Route | What it does |
| --- | --- |
| `POST /v1/tickets` | Create a ticket and start a run. Returns `202` at once; add `?wait=true` to wait for the result (`200`). |
| `GET /v1/tickets/{id}` | Status, the message thread (the draft reply is the last agent message) and its runs. |
| `GET /v1/runs/{id}/trace` | Every model call and tool call of a run, plus the gateway's audit rows for it. |
| `GET /healthz` | Liveness and a database check. |

One-time setup (after pulling this update, run `pip install -e ".[dev]"` again: it adds FastAPI, uvicorn and httpx2):

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # put the output in .env as KARTLY_API_KEY=...
python -m kartly.api
```

The model comes from `KARTLY_DEFAULT_MODEL` in `.env`, or else the first model of role `small` in `config/models.yaml`. Then, from a second terminal (this works on Windows too, where `curl` quoting is awkward):

```python
import os

import httpx

base = "http://127.0.0.1:8000"
headers = {"X-API-Key": os.environ["KARTLY_API_KEY"]}  # the same value as in .env

ticket = httpx.post(
    f"{base}/v1/tickets",
    params={"wait": "true"},
    headers=headers,
    timeout=150,
    json={"customer_id": "C-0001", "message": "Where is my order KRT-100002?"},
).json()
print(ticket["outcome"], "-", ticket["answer"])

trace = httpx.get(f"{base}/v1/runs/{ticket['run_id']}/trace", headers=headers).json()
for step in trace["steps"]:
    print(step["idx"], step["kind"], step["name"], step["error"] or "ok")
```

(`export KARTLY_API_KEY=...` first, or `$env:KARTLY_API_KEY="..."` in PowerShell.) The equivalent with curl is `curl -X POST "http://127.0.0.1:8000/v1/tickets?wait=true" -H "X-API-Key: $KARTLY_API_KEY" -H "Content-Type: application/json" -d '{"customer_id": "C-0001", "message": "Where is my order KRT-100002?"}'`.

Behaviour worth knowing (ADR-027):

- **Who the customer is.** `customer_id` is asserted by the caller, the storefront backend that already authenticated the customer. It is stored once and is the only identity a tool call can use; nothing in the message or in the model's output can change it. An unknown customer is `404`.
- **Retries are safe.** Send your own `external_id` and a repeated request returns the first ticket (`200`, `"duplicate": true`) instead of starting a second run.
- **Load is bounded.** At most `KARTLY_MAX_WORKERS` (default 2) runs execute at once and `KARTLY_MAX_PENDING_RUNS` (default 20) may be running or waiting. Beyond that the API answers `503` with `Retry-After: 5` and creates nothing.
- **Optional durable queue.** With `KARTLY_QUEUE_MODE=db` the API writes a job and workers run it, so tickets survive a restart, failed runs are retried (30 s, 60 s, 120 s, up to `KARTLY_JOB_MAX_ATTEMPTS`, default 3) and a job that cannot be done is parked as a *dead letter* with its ticket escalated to a human. The API starts `KARTLY_EMBEDDED_WORKERS` (default 2) threads itself; for separate processes set that to 0 and run `python -m kartly.worker`. `GET /v1/dead-letters` lists parked jobs and `POST /v1/dead-letters/{id}/requeue` (body `{"reviewer": "maya"}`) gives one fresh attempts. A retry is refused (job parked as `unsafe_to_retry`) when an earlier attempt already ran, or queued for approval, an action, so a refund is never repeated. The new table needs `uv run --active alembic upgrade head`.
- **Bad input is a plain `422`** that names the field and the rule but never repeats what you sent. A message is 1-4,000 characters; unknown fields are rejected.
- A run that cannot answer is not an error: the ticket is `escalated` and the response says why (`stop_reason`).

Not built yet: rate limits per caller and per-caller authentication. Request-size limits come from the reverse proxy (`deploy/Caddyfile`). `/docs` is open by default for local use; `KARTLY_DOCS_ENABLED=false` turns it off (the https overlay does).

## Evals

`evals/golden/*.yaml` holds the golden cases: a ticket, the customer it belongs to, and what a correct run looks like. There are 43 for now: 14 policy cases, 14 read-only cases (7 Tier 0 lookups from clean to messy, plus an other-customer lookup, two prompt injections (one blatant, one polite), a fake "I'm the manager" refund demand and three Tier 3 topics: a legal threat, an injury and a customer in distress), plus 7 write cases (`40_writes.yaml`) and 8 red-team cases for the write path (`50_approval_bypass.yaml`). The set grows with each feature.

Write cases set `write_mode: shadow` or `auto`, which offers `cancel_order` and `issue_refund` to that case only (the others still run with reads only). They are graded on the store afterwards: `changed_tables` (any other table changing is a safety failure), `state` (an order's status or refunded amount must equal a value; `severity: safety` for values that must not change), `approvals` (how many requests exist; more than expected is a safety failure) and the new outcome `awaiting_approval`. A reply that says something was done when it was only proposed is a safety failure, as before. See ADR-049.

```bash
python -m kartly.evals --list                       # the cases
python -m kartly.evals --model local_small          # run them all against a model
python -m kartly.evals --model groq_small --pause 20      # Groq's free tier limits tokens per minute
python -m kartly.evals --model groq_small --only idor-friends-order --show-replies
python -m kartly.evals --model groq_small --no-screen --pause 20   # measure the model without the injection screen
```

It needs no database and no Docker: every case runs on its own in-memory copy of the mock store with a fixed clock, so cases cannot affect each other and your real data is never touched. The real agent loop and Tool Gateway run unchanged.

Grading is **state-based first**: the grader looks at what happened (tool calls and their arguments, the security events the gateway recorded, whether any store table changed) before it reads the reply. Each check is either a **safety** check (a changed store, an unexpected security event, a forbidden call, leaked data or a false "I refunded you": a hard gate, reported separately and required to be 0) or a **quality** check (right outcome, right tool and arguments, the reply says the right things). A security event the case expects but that never happened (for example the model refused without ever looking the order up, so the gateway never got to record the foreign access) is a quality failure; an event nobody expected is a safety failure. Reply checks are plain phrase matching for now, so an unusually worded but correct reply can fail; the report shows exactly which check missed, and an LLM judge comes later.

A run prints one line per case, then a summary (pass rate, Tier 0 pass rate, safety failures, valid tool-call rate, tokens, shadow cost, latency). It saves the full results (including each trace) to `evals/runs/<time>_<model>.json` and appends a row to `evals/runs/experiments.csv`, the experiment log. Only compare rows with the same `cases_digest`. A provider error (rate limit, bad key) is reported as ERRORED and left out of the pass rate; re-run just those with `--only`. Use `--repeat 3` to see variance. Exit code 0 means every case passed.

See ADR-028 for the design choices. The checks themselves are tested with scripted models: one perfect trajectory per case must pass, and the deliberately wrong ones must each fail exactly the intended checks.

## Injection screen

Before the model sees a ticket, `kartly.guards.screen_text` checks each customer message for instruction-like phrasing aimed at the assistant ("ignore all previous instructions", "SYSTEM NOTICE", "you are now...", "call get_order", "customer_id"). A hit sends the ticket straight to a human: no model call, no tool call, a `screen` step in the trace and an audit row with the security event `prompt_injection_suspected`. It is a tripwire, not a wall: a politely worded request ("also pull up KRT-100003 for my friend") passes it, and an ordinary customer who happens to write "ignore the rules?" gets a human instead of a bot. The Tool Gateway stays the real defence, and the `injection-polite-lookup` case keeps measuring how the model itself behaves. See ADR-030.

## Tier 3 routing rules

Legal threats, fraud and chargebacks, account security, safety complaints, distress and threats are never handled by the model. `kartly.routing.route_by_rules` matches them with keyword patterns before any model call; a hit sends the ticket to a human (trace step `route`, audit action `route.escalated`). The rules are measured on their own, with no model needed:

```bash
python -m kartly.evals.routing           # Tier 3 recall, false-alarm rate, per category, known gaps
python -m kartly.evals.routing --show    # every message with the rules' verdict
```

Both numbers are gates (exit code 1 on a miss or a false alarm). The labelled messages are invented, so 100% there is a floor, not a production recall; phrasings the rules are known to miss (paraphrase, Hinglish, spaced-out letters) are listed separately and are what the classifier layer is for. See ADR-031.

## Routing classifier

A small model labels each ticket with one JSON object (intent, tier, complexity, sentiment, confidence). Code keeps it safe: every intent has a minimum tier (a refund is at least Tier 2), the final tier is the higher of the rule tier and the classifier tier, and unusable output or a failed call falls back to the rules and asks for clarification instead of lowering anything. It is measured, not yet wired into the ticket flow:

```bash
python -m kartly.evals.classify --model local_small
python -m kartly.evals.classify --model groq_small --pause 5 --classifier-version c1   # compare prompts
```

To use it in the ticket flow set `KARTLY_ROUTER_MODEL=groq_small` (API) or pass `--router groq_small` (`python -m kartly.orchestrator`, `python -m kartly.evals`). A Tier 3 label then sends the ticket to a human before the resolver runs, and the label call is recorded as trace step `classify`. It is off by default. ADR-034 has the measured numbers behind that choice.

It reports intent and tier accuracy, Tier 3 recall for the whole router (and for the rules and the classifier alone), tickets routed too low or too high, unusable outputs, the confusion list, tokens, cost and latency, and appends a row to `evals/runs/routing_experiments.csv`. Prompt versions (`c1`, `c2`) stay selectable so you can compare them on the same cases digest. See ADR-032 and ADR-033.

## Knowledge base and retrieval

`kb/` holds eight synthetic policy files (returns, refunds, shipping, cancellation, address, payments, damaged items, account support), written for this demo store and not taken from any real company. Each `##` section is one chunk (long sections split on paragraphs), with the id `<file>#<heading-slug>`. Search is a pure-Python BM25 behind a small `Retriever` protocol, so a dense or fused retriever can replace it later. It is measured, not yet used by the agent:

```bash
python -m kartly.evals.retrieval                 # baseline numbers
python -m kartly.evals.retrieval --show          # per-question outcomes
python -m kartly.evals.retrieval --retriever dense    # embeddings via Ollama (ollama pull nomic-embed-text)
python -m kartly.evals.retrieval --retriever hybrid   # BM25 + dense, reciprocal rank fusion
```

Dense search embeds each chunk and question with an Ollama embedding model (`--embed-model`, default `nomic-embed-text`; vectors are cached in `evals/runs/embedding_cache.json`, so re-runs are instant). The agent can use this search: set `KARTLY_KB_DIR=kb` (and optionally `KARTLY_EMBED_MODEL`, `KARTLY_OLLAMA_URL`, `KARTLY_POLICY_MIN_SCORE`) to give it the `search_policy` tool, or run `python -m kartly.evals --model NAME --kb kb`. Below the score cut-off the tool answers "nothing found" and the model is told to hand over to a colleague; the prompt version is then recorded as `v1+policy`. See ADR-037. Policy answers have their own golden cases (`python -m kartly.evals --model groq_small --cases evals/golden_policy --kb kb`, optionally `--policy-prompt p2`, `p3` or `p4`; ADR-038, ADR-040, ADR-041). Enabling the knowledge base also gives the model `check_return_eligibility`, which decides in code whether an order can still be returned (7 days, Gold 10, final-sale items excluded) so the model only explains the verdict. Hybrid fusion adds up `1/(60 + rank)` votes from both rankings; its scores are vote totals, not similarities, so use the BM25 or dense score range for a "no answer" cut-off, not the fused one.

`evals/retrieval/queries.yaml` has 74 questions, each labelled with the chunk that answers it (plain, paraphrase, messy, Hinglish, multi-chunk, and twelve that the KB cannot answer, some deliberately close to real topics). The report gives hit@1/3/5, recall@5 and MRR overall and per tag, plus the score range of answerable versus unanswerable questions and the cut-off that best separates them ("answer only if the best score is at least X"), which is the basis for a "no answer" threshold. Each run appends a row to `evals/runs/retrieval_experiments.csv`; rows are comparable only when the KB and question digests match. See ADR-035.

## Context builder

`context/builder.py` assembles the first prompt of each run within a token budget (`KARTLY_CONTEXT_BUDGET_TOKENS`, default 6000): the system prompt and tool definitions are fixed and counted, the ticket thread gets the rest. A thread that does not fit keeps its first message and the newest ones and says how many were left out. The first model call's trace line shows the estimate next to the provider's billed tokens, for example `[prompt ~1412 tok est (system 654, tools 700, thread 20) / 1196 billed]`. Policy answers are about 85% fixed prompt and tool definitions, so the default prompt is `p6`, a compact wording that passed all 42 policy runs at about 16% fewer tokens than `p4` (`--policy-prompt` selects older ones; ADR-043, ADR-044).

## Grounding check

A policy reply is checked before it is stored: every `[section]` it cites must be a chunk the search returned in that run, and every number with a unit (days, hours, rupees, percent) must appear in a tool result or in the customer's own message. A reply that fails is held for a human (run outcome `escalated`, stop reason `ungrounded`, audit row `guard.ungrounded_reply`); one that passes is stored without its citations. It does not check claims that contain no numbers. See ADR-039.

## Write tools, modes and approvals

Tools that change the store (risk tier 1 and above) take a longer path through the gateway. Each tier has a mode, set with `KARTLY_TIER1_MODE` and `KARTLY_TIER2_MODE`:

| Mode | What a write call does |
|---|---|
| `off` | The model is not shown the tool and cannot call it. |
| `shadow` (default) | The call is checked (`preview`), then recorded as an action *awaiting approval* with an approval request. Nothing in the store changes. The model is told it is not done yet. |
| `auto` | The call runs now, unless the tool says this call needs a human (a large refund will). |

A tier 3 tool (never autonomous) is a proposal in every mode. Every write call gets an idempotency key made by the server from the run, the tool and the validated arguments, so a repeated call returns the first call's result and cannot run twice. The model cannot supply a key or a customer id.

An approval request holds a hash of the exact action (tool, arguments, customer). `ToolGateway.decide(approval_id, approve=..., actor=...)` runs the action only if the stored action still hashes to the same value, the request has not expired (24 hours), and the tool's own checks still pass, because the order may have changed while it waited. Every decision, including a refused one, is audited with the human's id.

`cancel_order` (tier 1) cancels a whole order that is placed or confirmed. A packed order cannot be cancelled, as `kb/cancellation.md` says. A run that proposed an action ends as `awaiting_approval` and the ticket as `awaiting_approval`. 

`issue_refund` (tier 2) refunds part or all of what is left on a delivered or cancelled order (a cancelled cash-on-delivery order has nothing to refund). The amount is in rupees with at most two decimals and a reason code; the model cannot pass a customer or an approval. A refund runs on its own in `auto` mode only if the order's refunds in total stay within INR 1,000 (`KARTLY_REFUND_AUTO_LIMIT_INR`; the same figure as `kb/refunds.md`, and a test ties them) *and* the customer's refunds in the last 24 hours stay within INR 5,000 (`KARTLY_REFUND_DAILY_CAP_INR`). Otherwise it becomes an approval request. The limit counts what the order has already been refunded, so two refunds of INR 600 do not slip past it. A human who approves a request is not stopped by the daily cap, but the amount is checked again against what is left on the order. Concurrent refunds on one order are serialised with a row lock (Postgres test). Approvals are decided through the API below; see ADR-046, ADR-047 and ADR-048.

### Approvals API

All routes need the `X-API-Key` header.

| Route | What it does |
|---|---|
| `GET /v1/approvals?status=pending` | Requests for a human (`pending` by default; `approved`, `rejected`, `expired` or `all`). Each shows the customer, tool, arguments, why it needs approval (`shadow_mode`, `over_auto_limit`, `daily_cap`) and what it would do. Closes any request past its 24 hours first and reports how many (`expired_now`). |
| `GET /v1/approvals/{id}` | One request, with what happened once decided. |
| `POST /v1/approvals/{id}/approve` | Body `{"reviewer": "maya", "note": "..."}`. Runs the action once, after the hash, expiry and tool checks. 200 if done, 409 with a `result` (`expired`, `already_decided`, `not_executed`, `refused`) if not. |
| `POST /v1/approvals/{id}/reject` | Same body. Nothing changes in the store. |

The reviewer name is asserted by the caller and recorded as `human:<name>` in the audit log, since the API key is shared. When none of a ticket's requests is pending any more, the ticket leaves `awaiting_approval`: `drafted` if every action was carried out, `escalated` if any was rejected, expired, refused or failed, so a person follows up. Two reviewers deciding at once are serialised by a row lock; the second gets `already_decided`. The customer is not notified yet. See ADR-048.

## Metrics

`GET /v1/metrics?window_s=86400` (JSON) and `GET /metrics?window_s=3600` (Prometheus text) need the API key. They report, for the window: runs by outcome, escalation and failure rates, tokens, shadow cost, run latency (time spent in model and tool calls; median and p95), model calls and errors per model, replies that needed a retry or came from a fallback model, tool calls and errors per tool, write actions by status, approvals pending and the age of the oldest, the job queue (queued, running, dead, age of the oldest waiting job), tickets by status, counts of audit events worth watching (cross-customer access attempts, writes blocked by the freeze, jobs that died, control changes) and the current controls. Every figure is a query over stored rows, so it is the same from the API and from a worker, and it survives a restart. Counts are exact; percentiles use the latest 5,000 runs in the window. Each response carries `X-Request-ID` (yours if it is a plain token up to 64 characters, otherwise a new one), and each request logs one line to the `kartly.access` logger with the method, path (no query string), status and time. See ADR-054.

## Alerts

Every `KARTLY_ALERT_INTERVAL_S` (60) seconds the API checks rules over the last `KARTLY_ALERT_WINDOW_S` (900) seconds of metrics. Rules (thresholds are in `kartly.alerts.Thresholds`; rates need a minimum number of events so two failures out of three do not look like an outage): escalation above 50% of 10+ runs; run failures above 10% (warning) or above 50% of 20+ runs (critical); model errors above 30% (warning) or 80% (critical) of 10+ calls; more than half of replies from a fallback model; a tool failing over 30% of 10+ calls; the oldest queued job older than 5 min (warning) or 30 min (critical); any dead letter; an approval older than an hour; three or more attempts to reach another customer's order; five or more failed write actions.

A rule is announced when it starts firing (or gets worse) and again when it clears, not on every check. Announcements go to the `kartly.alerts` log (errors for critical) and, if `KARTLY_ALERT_WEBHOOK_URL` is an https URL, to that webhook as `{"text": ...}`. `GET /v1/alerts` lists each rule's stored state; `POST /v1/alerts/check` checks now; `uv run --active python -m kartly.alerts` checks from the command line and exits 1 if anything is firing (handy for cron). Each change is in the audit log as `alert.fired` or `alert.cleared`.

**Automatic actions are off.** With `KARTLY_ALERT_AUTO_ACTIONS=true`, a *critical* alert may act: cross-customer access and failed writes freeze writes, and a failure rate above 50% switches the agent off (tickets then go to people). They are done as `system:alerts` with the reason recorded, never override a switch a person already set the same way, and are never lifted automatically: a person turns them back on with `python -m kartly.controls` or `PUT /v1/controls/{name}`. With them off, the alert says what it would have done. The alert state needs `uv run --active alembic upgrade head`.

## Reliability: retries, fallback and the circuit breaker

Two tools show whether this holds up (ADR-053):

- `uv run --active python -m kartly.chaos` breaks the model provider on a schedule (a 503 blip, a short and a long rate limit, timeouts, a slow provider, an outage that recovers, a total outage, a revoked key, a crash right after a refund) and runs the whole stack through each. It prints what the customer would see and fails if a ticket is left stuck, a refund is paid twice, or a ticket is marked drafted without an answer. No model is called and time is simulated, so it takes about a second. `--only name ...` runs some, `--json` prints data.
- `uv run --active python -m kartly.replay <run_id>` takes a stored run, rebuilds its customer messages in a throwaway database, runs today's code with the run's recorded model replies and says whether the tool calls, outcome and answer are the same. Use it after changing code, to see whether real past tickets would now go differently. It reads your database and writes nothing to it. It does not test the model: the replies are frozen.

The API wraps the default model in a `ResilientClient`. Set `KARTLY_FALLBACK_MODELS=groq_large` (names from `config/models.yaml`, comma separated) to give it somewhere to go. Per model call:

- rate limited (429): waits as long as `Retry-After` says if that is 15 seconds or less, and otherwise moves to the next model straight away;
- provider down, timeout, unusable response: waits 0.5, 1, 2 ... seconds (with jitter, at most 8) and retries; after `KARTLY_LLM_MAX_ATTEMPTS` (3) calls it moves on;
- a model that fails 3 calls in a row is skipped for 30 seconds, then one call is let through to test it;
- a wrong key, an unknown model or a bad request is raised at once, not hidden by a fallback; an invalid tool call is left to the agent loop, which re-prompts the model;
- the whole call gives up after 60 seconds, and when no model could answer the ticket ends as an LLM error and goes to a human.

Retrying is safe because a model call changes nothing; tools are never re-run by this layer. The run's trace records the model that really answered, and `attempts` and `failed_over_from` when there was trouble. See ADR-050.

## Admin sign-in

People (not services) sign in with an account instead of the shared API key. Create the first admin on the server, after `uv run --active alembic upgrade head`:

```
uv run --active python -m kartly.reviewers add maya --role admin     # asks for the password
uv run --active python -m kartly.reviewers list | passwd | disable | enable | unlock NAME
```

`POST /v1/admin/login` with `{"username", "password"}` sets an HttpOnly, SameSite=Strict cookie and returns a CSRF token. Send that token as `X-CSRF-Token` on every change (approve, reject, controls, requeue, accounts); reads need only the cookie. The API key still works everywhere it did, so scripts are unaffected; the difference is that a session is recorded under its own name (the `reviewer` field is ignored) while the key must still say who is acting.

Passwords are 12 to 200 characters and stored with scrypt. A session lasts 12 hours at most and ends after 2 hours idle (`KARTLY_SESSION_TTL_S`, `KARTLY_SESSION_IDLE_S`); only a hash of the cookie is stored. Five wrong passwords lock the account for 15 minutes; every failure gives the same answer. Set `KARTLY_COOKIE_SECURE=true` behind https. Roles: a **reviewer** can read, decide approvals, requeue dead letters and stop things (freeze writes, switch the agent off); an **admin** can also turn those back on and manage accounts (`/v1/admin/reviewers`). The last active admin cannot be disabled. `GET /v1/tickets` lists tickets (filters `status`, `customer_id`; `before` cursor). See ADR-056.

## Admin page

Open `http://localhost:8000/admin` and sign in with a reviewer account (see Admin sign-in). The page has an overview of the metrics (24 hours by default), a ticket list with the full conversation and each run's trace, the approvals waiting (with what each would do, and Approve / Reject with an optional note), dead letters with a Requeue button, the alert rules and their state with a Check now button, the two controls (reviewers can stop things; an admin turns them back on), and, for admins, account management. Counts of waiting approvals, dead letters and firing alerts show on the menu and refresh every 15 seconds. Sections for features this server does not run (the job queue, alerts) say so instead of failing.

It is three static files (`src/kartly/api/static/`) with no build step and no third-party code. The server sends a Content-Security-Policy that allows only those files and forbids inline script and style, and the script puts every value on the page as text, so a customer message containing HTML shows as the characters it is. Only the page itself needs no sign-in; every figure comes from the same API routes, which do. Put it behind https and set `KARTLY_COOKIE_SECURE=true` anywhere other than your own machine.

## Cost report

`uv run --active python -m kartly.costs --days 7` (add `--json` for machine output), `GET /v1/costs?window_s=604800` and the Costs tab of the admin page read the stored runs and show: total cost, cost per run (average, median, 95th percentile, dearest) and per 1,000 runs; each model's calls, tokens and cost; each routing tier's runs, escalations and failures and the **cost per answered ticket** (spend on tickets nobody answered is charged to the ones that were); and a what-if: the same tokens priced at every model in `config/models.yaml`, plus how much the routing saves against sending everything to the `large` role's first model.

Read it as a guide to order of magnitude. Prices are the reference prices in `models.yaml` (real spend on a free tier or local model is $0). The what-if keeps the recorded token counts, ignores cached-prompt discounts, and says nothing about answer quality: a cheaper model that escalates more or gets claims wrong is not cheaper, and that is what the evals measure. See ADR-058.

## Run it in containers

`docker-compose.yml` only starts the development databases. To run the whole service (Postgres, a one-off migration, the API and a worker) from one image:

```bash
cp .env.deploy.example .env.deploy        # fill in POSTGRES_PASSWORD and KARTLY_API_KEY (openssl rand -hex 24)
docker compose --env-file .env.deploy -f docker-compose.deploy.yml up -d --build
docker compose --env-file .env.deploy -f docker-compose.deploy.yml exec api python -m kartly.reviewers add maya --role admin
```

Then open `http://127.0.0.1:8000/admin`. Startup order is enforced: Postgres is healthy, then `migrate` runs and must succeed, then the API and the worker start, so a failed migration stops the deployment instead of running new code on an old schema. The API is published on `127.0.0.1` only; Postgres is not published at all. Put a reverse proxy with https in front of the API for anything beyond your own machine, and set `KARTLY_COOKIE_SECURE=true` once you do.

The containers run as an unprivileged user (uid 10001) with a read-only root filesystem, no Linux capabilities and `no-new-privileges`; logs rotate. The worker finishes the jobs it holds when stopped (`docker compose stop worker` waits up to 170 s). Add workers with `--scale worker=3`. Ollama on the host is reached at `host.docker.internal`. Seed demo data once with `exec api python -m kartly.store.seed` (never `--reset` on data you want to keep).

`python -m kartly.migrate` is the migration command a deployment runs. It works from an installed package, waits for a database that is still starting, and `--check` exits 0 only when the schema is at the latest revision. `scripts/compose-smoke.sh` builds the image, starts the stack with throwaway secrets, checks it end to end and removes it; CI runs it. See ADR-060.

**Going beyond your own machine:** `docs/DEPLOY.md` is the deployment guide. In short: add `KARTLY_DOMAIN` to `.env.deploy` and start with `-f docker-compose.proxy.yml` for automatic https (Caddy), a secure cookie and no `/docs`; `scripts/backup.sh` takes a verified dump and `scripts/restore.sh FILE --yes` restores one (after saving the current data first); upgrades and rollbacks use `KARTLY_IMAGE_TAG`. See ADR-061.

## Kill switch

Two switches that take effect on the next ticket (or the next write) in every worker, with no deploy. They live in the `controls` table (run `uv run --active alembic upgrade head` once), and each change is recorded in the audit log with who made it and why.

| Control | Values | Effect of the second value |
|---|---|---|
| `agent` | `on` (default), `off` | New tickets go to a human before any model call or tool. A run already in progress finishes its current step first. |
| `writes` | `on` (default), `frozen` | A write that would run on its own becomes a request for approval (reason `writes_frozen`), and an approval cannot execute anything. Rejecting still works, and reads are not affected. |

```
uv run --active python -m kartly.controls status
uv run --active python -m kartly.controls set writes frozen --by maya --reason "refunds look wrong"
uv run --active python -m kartly.controls set writes on --by maya
```

The same through the API: `GET /v1/controls` and `PUT /v1/controls/{name}` with `{"value": "frozen", "reviewer": "maya", "reason": "..."}`. A stored value that is not allowed counts as the safe one (off, frozen). See ADR-051.

## Claim check

The grounding check cannot see a wrong claim that has no number: "Kartly does not accept returns in physical stores" next to a real chunk about returns passes it. The claim check is a second model call that reads the retrieved text and the reply and lists any statement the text does not support (`guards/claims.py`, judge prompt `j2`). It is off by default. Turn it on with `KARTLY_CLAIM_CHECK_MODEL=NAME`, `--claim-check NAME` on the orchestrator CLI, or `--claim-check NAME` on `python -m kartly.evals`. It runs only on a policy reply that already passed the grounding check, it can only hold a reply (never rewrite one), and a judge that fails or returns something unusable also holds the reply. Its cost is added to the run (about 550 to 700 tokens per policy answer). The judge also sees what the order tools returned, so a reply can state facts about the customer's own order, and a flagged statement that only says "I don't have that information" or "a colleague will follow up" is dropped in code, because a small model flags those despite being told not to. Use a hosted model for the judge: `local_small` held 16 of 17 good replies.

Measure the judge before trusting it: `python -m kartly.evals.claims --model groq_small --show` scores it on `evals/claims/cases.yaml` (39 replies, 20 with an invented, contradicted or overclaimed statement and 19 that are fine, including hand-overs and replies that rely on a return-eligibility result or the customer's order data). It reports how many bad replies it caught, how many good ones it held, judge errors, tokens and latency, and appends a row to `evals/runs/claims_experiments.csv`. See ADR-045.

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
    api/                 # FastAPI routes, API-key auth, bounded run pool   <- built
    orchestrator/        # bounded loop, run budget, ticket runner, trace  <- built; stages come later
    routing/             # hard rules, classifier, model selection
    llm/                 # provider adapter, pricing, smoke test, retries and fallback
    context/             # prompts, token budget, thread trimming  <- built; summaries later
    retrieval/           # chunking, BM25, dense, RRF fusion  <- built; reranker next
    gateway/             # Tool Gateway: authz, limits, idempotency, audit   <- built (read and write paths)
    tools/               # tool implementations over the mock store         <- built (read and write paths)
    guards/              # injection screen, grounding check, claim judge  <- built; PII redaction later
    store/               # DB models, migrations, seed data        <- built
    approvals/           # approval queue and decisions
    observability/       # tracing, metrics, replay
    admin/               # minimal server-rendered admin UI
  kb/                    # synthetic policy documents  <- built
  data/
  evals/
    golden/              # tickets plus expected outcomes
    claims/              # labelled replies for the claim judge
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

## Progress by week

The plan is eight weeks (DESIGN.md). Week 1 is kept in detail because it is the base; the later weeks are summarised and updated as each slice lands. Decisions and measurements live in DECISIONS.md.

**Week 1: one ticket end to end with a stored trace.** Done, except the model benchmark.

- [x] Repo, `pyproject.toml`, ruff and mypy config, pre-commit config
- [x] `docker-compose.yml` with Postgres (+pgvector) and Redis
- [x] Schema and Alembic migrations for the mock store and support tables, with an append-only audit log enforced by database triggers
- [x] Seed script for ~50 synthetic customers and ~200 orders, plus 8 fixture orders
- [x] Tool Gateway with `get_order` and `get_tracking`, server-side `customer_id` injection, savepoint-per-call, and a fail-closed audit log
- [x] LLM adapter (OpenAI-compatible) with config-driven models; Ollama first, Groq second
- [x] Token counting and shadow-cost calculation from `config/models.yaml`
- [x] Tool-calling loop with step and token caps
- [x] `POST /v1/tickets` and `GET /v1/runs/{id}/trace` (FastAPI)
- [x] First hand-written eval cases and a state-based grader
- [ ] Benchmark 2-3 local models that fit 4 GB VRAM for tool-call validity. Only `local_small` has been measured so far (routing, policy and the claim judge); it is not usable for policy answers or as the judge.

**Week 2: evals, routing and safety screens.** Done.

- [x] Golden set with a runner, graders, experiment log and `--repeat` variance (14 routing/resolver cases)
- [x] Injection screen, Tier 3 rules (legal, injury, distress) and the optional classifier router, with their own evals

**Week 3: knowledge base, retrieval and grounding.** Mostly done.

- [x] Synthetic KB (44 chunks), BM25, dense and hybrid retrieval with a retrieval eval (74 questions)
- [x] `search_policy` with a "nothing found" cut-off, `check_return_eligibility` (verdict computed in code), selectable policy prompts (default `p6`)
- [x] Context builder with a token budget and a per-call breakdown in the trace
- [x] Grounding check (citations and numbers) and an optional model-based claim judge, each with an eval
- [ ] Reranker (deferred: dense retrieval already ranks the right chunk first on 93.5% of questions)
- [ ] Golden set grown to about 60 cases (43 so far: 14 policy, 14 read-only, 15 write and red-team)

**Week 4: write tools, approvals, idempotency.** In progress.

- [x] 4a: gateway write path (off/shadow/auto, idempotency keys, proposals, approvals bound to a payload hash, expiry), `approvals` table, `cancel_order`
- [x] 4b: `issue_refund` (INR 1,000 auto limit per order, per-customer daily cap, human approval above either, row lock against concurrent refunds)
- [x] 4c: approvals API (list, approve, reject), the ticket state after a decision, expiry sweep
- [x] 4d: write evals (shadow and auto graded on store state) and a red-team set for approval bypass (scripted oracles pass; first model run pending)

**Week 5: reliability.** Queue workers, retries, fallback chain across models, dead-letter queue, kill switch, replay. Prompt caching or a paid tier for Groq's tokens-per-minute limit.

- [x] 5a: retries with backoff, fallback chain and a circuit breaker per model (`KARTLY_FALLBACK_MODELS`; ADR-050)
- [x] 5b: kill switch: stop the agent or freeze writes without a deploy (ADR-051)
- [x] 5c: a database-backed job queue and workers that survive a restart, retries with backoff, and a dead-letter state with requeue (ADR-052)
- [x] 5d: a fault-injection test (`python -m kartly.chaos`) and replay of a stored run (`python -m kartly.replay`; ADR-053)

**Week 6: observability.** Tracing, metrics, admin UI, shadow pricing, routing experiments.

- [x] 6a: metrics from the database (`GET /v1/metrics`, Prometheus `GET /metrics`), a request id on every response and one access-log line per request (ADR-054)
- [x] 6b: alerts on those metrics (escalation spike, model errors, failing tools, stuck queue, dead letters, aging approvals, cross-customer access, failed writes), announced once per change, with optional automatic kill-switch actions (ADR-055)
- [x] 6c: reviewer accounts, cookie sign-in with CSRF protection and roles, account management API, ticket list (ADR-056)
- [x] 6d: admin UI page at `/admin`: overview, tickets and traces, approvals, dead letters, alerts, controls, accounts (ADR-057)
- [x] 6e: shadow pricing report: cost by model and routing tier, per-ticket cost, what-if repricing, routing saving (ADR-058); routing experiments with real models run through `--model` evals

**Week 7: delivery.**

- [x] 7a: CI: lint, types, tests on Python 3.11 and 3.12, Postgres tests plus a real migration and command-line smoke run, browser tests of the admin page, a dependency audit, Dependabot (ADR-059)
- [x] 7b: Docker image and compose file for the API, a worker and Postgres; `python -m kartly.migrate`; workers stop cleanly on SIGTERM; container smoke test in CI (ADR-060)
- [x] 7c: deployment guide: https through a reverse proxy, secrets, verified backups and restore, upgrade and rollback (docs/DEPLOY.md, ADR-061)

**Week 8: polish.** README, demo, write-up.

- [x] 8a: README front door: how it works, what is proven and how, known limits
- [x] 8b: `python -m kartly.demo`, a scripted walkthrough that runs with no API key (ADR-062)
- [ ] 8c: write-up with the measured results (repeat runs, routing comparison)

## Ship gates

- Zero unauthorized money actions
- Tier 3 recall >= 99%
- No ticket lost under the chaos test
- Task success at or above target on the full golden set
