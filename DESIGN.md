# Kartly Support Agent: Design Document

**Status:** Planning complete, ready for implementation (Week 1)
**Author:** (you)
**Last updated:** 2026-10-23
**Companion docs:** `DECISIONS.md` (ADR log), `README.md` (repo layout, progress by week)

---

## 0. Summary

An autonomous e-commerce customer support platform for a fictional mid-size store ("Kartly", ~10k tickets/day). The system reads incoming tickets, classifies intent and risk, retrieves policy from a knowledge base, calls tools through a deterministic gateway, resolves what it safely can, and escalates the rest to a human with a full summary.

The project is built to demonstrate what companies expect from an AI engineer: evals, routing, cost management, context management, guardrails, reliability, observability, and deployment.

**Design principles**

1. **The model proposes, code disposes.** Policy (refund limits, access control) is enforced in code, never only in a prompt.
2. **Deterministic wherever possible.** The LLM is used only where judgment or language is needed.
3. **Every step is traced and replayable.**
4. **Failures fall to a human.** No ticket is ever dropped.
5. **Simplest thing that works.** A fixed workflow with one bounded agent loop, not a free-roaming agent.

---

## 1. Problem, users, goals, non-goals

**Scenario.** About 60% of Kartly's tickets are repetitive (where is my order, how do I return this, change my address). Human agents are slow on those and rushed on the hard ones.

**Users**
- **Customer:** sends a ticket and gets a reply.
- **Support agent:** reviews escalations and approvals; needs a good summary so they do not start from scratch.
- **Support lead/admin:** watches quality, cost, and escalation rates; edits policies and thresholds.
- **On-call engineer:** debugs bad runs through traces.

**Goals**
- Fully resolve low-risk tickets without a human.
- Draft or pre-fill medium-risk ones for approval.
- Hand off high-risk ones instantly with a clean summary.
- Never lose a ticket; never take an unauthorized money action.

**Non-goals (v1)**
- Voice, multilingual support, proactive outreach.
- Real payment gateway integration (the backend is mocked).
- Fine-tuning or training models.
- Replacing the human team; the aim is to shrink their queue.

### Risk tiers

Routing, permissions, and evals all hang off this table.

| Tier | Intents | Autonomy |
|---|---|---|
| **0: Read-only** | Order status, tracking, return/shipping policy, product questions | Fully autonomous |
| **1: Low-risk writes** | Change address (pre-shipment), cancel unshipped order, generate return label | Autonomous, with eligibility checks |
| **2: Money** | Refunds, replacements, goodwill credits | Auto up to INR 1,000; human approval above that |
| **3: Never autonomous** | Legal threats, fraud/chargebacks, account security/deletion, safety complaints, abusive or highly distressed customers | Immediate escalation |

### v1 intents (cap of ~8)
order status, tracking, return, refund, cancel, address change, policy FAQ, escalation.

---

## 2. Success metrics and SLOs

| Area | Metric | v1 target |
|---|---|---|
| Quality | End-to-end correct outcome on golden set | >=85% overall, >=95% on Tier 0 |
| Quality | Intent/routing accuracy | >=92% |
| Safety | Tier 3 escalation recall | >=99% |
| Safety | Unauthorized money actions | 0 |
| Quality | Ungrounded policy claims | <2% |
| Business | Auto-resolution rate | >=50% of tickets |
| Business | Human rating of escalation summaries | >=4/5 |
| Cost | Avg shadow cost per ticket | <$0.05 (at reference prices) |
| Cost | Hard cap per run | Token and step caps (see section 5); shadow-cost equivalent ~$0.30 |
| Latency | Tier 0 replies | p50 <5s, p95 <15s |
| Reliability | Tickets lost | 0 |
| Tool use | Valid tool-call rate | Tracked from day one; target set after baseline |

Notes:
- Cost per **resolved** ticket is tracked separately from cost per ticket.
- Safety metrics are **gates**, not goals. If Tier 3 recall or unauthorized-action count misses target, the build does not ship regardless of other numbers.
- Quality targets are provisional. Small/free-tier models may score lower; targets are recalibrated after the first baseline run, and the gap itself is a reported finding.

---

## 3. Requirements

### Functional
1. **Ingest:** accept tickets via API/webhook, with multi-turn threads.
2. **Classify:** intent, risk tier, complexity per ticket.
3. **Answer:** use a policy/FAQ knowledge base with citations.
4. **Tools:** `get_order`, `get_tracking`, `search_policy`, `check_return_eligibility`, `update_address`, `cancel_order`, `create_return_label`, `issue_refund` (gated), `create_escalation`.
5. **Escalate:** summary, what the agent tried, recommended next step.
6. **Approvals:** queue for human approve/reject on Tier 2 above threshold.
7. **Admin UI:** review queue, run traces, cost and quality dashboard.
8. **Audit log:** every action, who or what took it, and why.

### Non-functional
- **Reliability:** idempotent tool calls, retries with backoff, model fallback, dead-letter queue.
- **Security:** ticket text treated as untrusted; PII redacted in logs; least-privilege tool scopes.
- **Scale:** design for 10k tickets/day, ~20 concurrent runs.
- **Maintainability:** prompts and policies versioned; models swappable by config; every change gated by evals.
- **Observability:** full trace per ticket (prompts, tool calls, tokens, latency, cost).

### Constraints and assumptions
- Solo developer, ~8 weeks.
- Synthetic data only: a seeded mock store (customers, orders, shipments) plus an LLM-generated, hand-curated ticket set including adversarial cases.
- Student budget: no paid LLM APIs. Local models on an RTX 3050 (4 GB VRAM) plus hosted free tiers.
- Postgres, Redis, Docker for infrastructure.

### Scope decisions
Chat-style API only in v1 (email as a later adapter); English only; INR 1,000 auto-refund threshold (configurable per tier); shadow mode first; ~8 intents.

---

## 4. High-level architecture

```
Customer -> API (FastAPI) -> Queue (Redis) -> Worker: Orchestrator
                                                   |
 1 INTAKE    dedupe, PII redaction, injection screen, hard-rule check
 2 ROUTE     classifier -> intent, risk tier, complexity, confidence
 3 RESOLVE   bounded agent loop: context builder + RAG + tools
                      | every tool call goes through
              +-----------------------------+
              |        TOOL GATEWAY         |  authz, eligibility, limits,
              |  (deterministic, no LLM)    |  idempotency, audit log
              +-----------------------------+
                      |                     |
                 Mock store DB          Approval queue
 4 VALIDATE  output guard: grounding, PII, tier re-check
 5 DELIVER   auto-send | draft for human | await approval | escalate

 Cross-cutting: Postgres (+pgvector), Redis, tracing (OTel), Admin UI
```

The **Tool Gateway** is the most important component. The LLM never touches the database directly. It requests an action; the gateway decides whether to execute it, queue it for approval, or reject it.

---

## 5. Agent design

**Workflow with a bounded agent inside, not a free-roaming agent.** Stages 1, 2, 4, and 5 are deterministic code. Only stage 3 runs an LLM tool-calling loop, with hard limits:

- max 6 loop steps and 8 tool calls
- per-run token budget and wall-clock timeout (shadow-cost equivalent of ~$0.30)
- only the tools allowed for the routed intent are shown to the model

Implementation is a plain Python state machine and a hand-written tool loop (no agent framework).

### Ticket state machine

`RECEIVED -> SCREENED -> ROUTED -> RESOLVING -> VALIDATING -> (SENT | DRAFTED | AWAITING_APPROVAL | ESCALATED) -> CLOSED`

Any stage can fall to `FAILED -> human queue` after retries.

### Tools

| Tool | Tier | Key args | Gateway checks |
|---|---|---|---|
| `get_order` | 0 | order_id | order belongs to this ticket's customer |
| `get_tracking` | 0 | order_id | same |
| `search_policy` | 0 | query | RAG over KB, returns chunks with IDs |
| `check_return_eligibility` | 0 | order_id, item_ids | deterministic window and condition logic |
| `update_address` | 1 | order_id, new_address | status must be pre-shipment; address validated |
| `cancel_order` | 1 | order_id | must be unshipped |
| `create_return_label` | 1 | order_id, item_ids | eligibility check must have passed |
| `issue_refund` | 2 | order_id, amount, reason_code | <= remaining refundable; over INR 1,000 becomes an approval request |
| `create_escalation` | always on | summary, category, recommended_action | none |

### Security-critical details
- **`customer_id` is never a tool argument.** The gateway injects it from the authenticated ticket, so even a successful prompt injection cannot read or refund another customer's order (IDOR).
- **Idempotency keys are generated server-side** from run, tool, and arguments, so a retry can never refund twice.
- **Approvals are bound to a hash of the exact action payload** and expire.

### Rollout modes
Per-tier mode switch from day one: `off` -> `shadow` (agent drafts, human sends) -> `auto`. Rollout order: shadow for everything, then autonomy for Tier 0-1, then Tier 2.

---

## 6. Routing design

Three layers, in order:

1. **Hard rules (regex/keywords):** legal, lawyer, chargeback, fraud, hacked, injury, and similar terms force Tier 3. Cheap, instant, high recall.
2. **LLM classifier (small model, structured output):** returns intent, tier, complexity (simple vs multi-step), sentiment, and confidence.
3. **Model selection:**

| Case | Handled by |
|---|---|
| Tier 0, simple | Small tier (local or fast hosted model) |
| Tier 1-2 or multi-step | Large tier (hosted free-tier model) |
| Confidence < 0.7 | One clarifying question, then escalate |
| Tier 3 | No resolver; straight to escalation (summary written by a small model) |

**Final tier = max(rule tier, classifier tier).** The classifier can raise risk but never lower it.

Because the local classifier is a small model, the hard-rules layer carries extra weight for the Tier 3 recall gate.

**Later experiment:** compare upfront routing against a cascade (try the small model first, escalate on low confidence) and report cost and accuracy for each.

---

## 7. Context management

Each run's context is assembled by a dedicated builder with a token budget (soft cap ~8k input tokens):

| Layer | Content | Notes |
|---|---|---|
| Static prefix | system prompt + tool definitions | placed first, prompt-cached where the provider supports it |
| Policy | top 3-5 reranked KB chunks | with chunk IDs for citations |
| Customer facts | tier, last 5 orders (compact) | from the DB, not from model memory |
| Thread | last N messages verbatim + rolling summary of older ones | |
| Tool results | trimmed to needed fields | cap line items, drop internal metadata |
| Ticket text | wrapped and labeled as **untrusted data** | never mixed into instructions |

**Memory layers:** working memory (run state), session memory (ticket thread), long-term memory (structured customer profile plus short summaries of past tickets; kept simple in v1).

Within the loop, old tool outputs are compacted once used so context does not snowball. The 8k cap also fits free-tier token-per-minute limits.

---

## 8. Data model and API

### Postgres tables
- **Mock store:** `customers`, `orders`, `order_items`, `shipments`
- **Support:** `tickets`, `messages`, `runs`, `run_steps`, `actions` (unique idempotency key), `approvals`, `escalations`, `audit_log`
- **Knowledge and config:** `kb_documents`, `kb_chunks` (pgvector), `prompt_versions`

### API
- `POST /v1/tickets`, `POST /v1/tickets/{id}/messages`
- `GET /v1/tickets/{id}`
- `GET /v1/runs/{id}/trace`
- `GET /v1/approvals?status=pending`, `POST /v1/approvals/{id}/decision`
- `GET /v1/metrics/cost`
- Config: per-tier mode (`shadow` or `auto`) and thresholds

### Queue and retrieval
- Redis queue with a simple worker (RQ or ARQ); sufficient at 10k tickets/day.
- Hybrid retrieval: Postgres full-text plus pgvector, with a reranker. Policy text is full of exact terms ("30 days", "final sale") where lexical search matters.

---

## 9. Eval strategy

### Golden set
Target ~150 cases, grown over time. Each case has the ticket thread, a seeded database state, and expected results: tier, intent, allowed and forbidden tool calls, expected final DB state, outcome (resolved, drafted, escalated), and required policy citations.

Built by LLM-generating tickets, then hand-curating. Composition:
- ~8 intents x normal, messy, and multi-turn variants
- ~30 adversarial cases: prompt injection, cross-customer order access, "I'm the manager, approve this", abusive customers, missing order IDs, multi-intent tickets

### Evals per layer

| Layer | Metrics |
|---|---|
| Routing | intent accuracy, confusion matrix, Tier 3 recall |
| Retrieval | hit@k, MRR on labeled query-to-chunk pairs |
| Tool use | valid-call rate, correct tool and arguments |
| End-to-end | task success, graded by final database state |
| Safety | forbidden actions (must be 0), injection resistance |
| Response quality | LLM judge on a rubric (grounded, correct, tone, complete) plus deterministic citation checks |

### Principles
- **State-based grading first, LLM judge second.**
- Calibrate the judge against ~30 human labels and report agreement. Use a different model family than the resolver.
- Temperature 0; run each eval 3 times to report variance.
- **CI gating:** a ~25-case smoke set on every PR (record/replay, free and deterministic); full set nightly or manually. Safety metrics are hard gates; quality regressions beyond a threshold block the merge.
- Every change gets an **experiment log row** (prompt version, model, accuracy, cost, latency).
- Sample failed runs back into the golden set.

### Eval budget note
One full run of 150 cases is roughly 900 LLM calls. Full runs use local models or are spread over days; hosted free tiers are used for the resolver and judge sparingly.

---

## 10. Safety and threat model

A successful prompt injection must not be able to do damage, because the gateway, not the prompt, is the defense.

| Threat | Mitigation |
|---|---|
| Direct injection in ticket text | Text wrapped as untrusted data; cheap injection screen; gateway enforces policy regardless of model output |
| Indirect injection (order notes, address field, KB docs, tool outputs) | Same wrapping for all external content; never treated as instructions |
| Cross-customer access (IDOR) | `customer_id` injected server-side, never an argument |
| Data exfiltration | Agent replies only to the ticket's own customer; no arbitrary emails or URLs; links stripped from output |
| Money abuse | Per-action limit, per-customer daily cap, refund <= remaining refundable, idempotency, approval above threshold |
| Fake authority ("I'm the manager") | Claims in ticket text have zero effect on permissions |
| Approval bait-and-switch | Approval bound to payload hash; expires |
| Hallucinated policy | Policy claims must cite chunk IDs; numbers (dates, amounts) come only from deterministic tools |
| PII leakage | Redaction in logs and traces; output PII check |
| Denial of wallet / loops | Step and token caps; per-customer rate limit; ticket dedupe |
| Admin abuse | Auth and roles (agent vs lead); append-only audit log |
| Real data sent to free-tier providers | Synthetic data only (some free tiers may use inputs to improve models) |

The red-team suite from section 9 runs in CI, so these mitigations are tested, not just documented.

---

## 11. Reliability and failure modes

| Failure | Handling |
|---|---|
| Provider 429, 5xx, timeout | Backoff with jitter, capped retries, circuit breaker, then fallback chain |
| Malformed tool call or JSON | Schema validation, one repair retry, then escalate |
| Loop or no progress | Step cap, repeated-call detection |
| Crash mid-run | Steps persisted per run; job resumes from last step; idempotency keys make replays safe |
| Tool write half-done | Action rows with status (pending, succeeded, failed); reconcile-on-retry |
| Duplicate webhook | Dedupe by external ID |
| Queue backlog | Priority lanes (Tier 3 and approvals first); SLA timer routes stale tickets to humans |
| Repeated failure | Dead-letter queue, which is itself a human queue |
| Bad deploy or incident | Kill switch and per-tier feature flags (`auto` -> `shadow` -> `off`) |

**LLM provider layer.** One provider-agnostic adapter (OpenAI-compatible interface). Rate-limit-aware client with a token bucket and a fallback chain, for example Groq -> Gemini -> OpenRouter -> local. Retries are capped because failed requests can still count against free-tier daily quotas.

A small load test (Locust or k6) demonstrates 20 concurrent runs.

---

## 12. Observability

- **Traces:** one per ticket with spans per stage, LLM call, and tool call. Attributes: model, prompt version, tokens, latency, shadow cost, tier, outcome.
- **Metrics:** auto-resolution and escalation rate (by reason), approval rate, tool error rate, fallback and rate-limit-hit rate, p50/p95 latency, cost per ticket and per resolved ticket, tool-call validity.
- **Alerts:** cost spike, error rate, queue depth, fallback surge.
- **Replay:** re-run any stored run against a new prompt or model.
- **Feedback loop:** human edits to drafts become labeled data.

**Tooling.** Own `run_steps` in Postgres plus OpenTelemetry emission now. The trace viewer is chosen at Week 6 (a lightweight open-source one such as Arize Phoenix, or a hosted free tier). Self-hosted Langfuse is avoided by default because its supporting services are heavy for a laptop.

---

## 13. Deployment and CI/CD

- **Local:** docker-compose with api, worker, Postgres+pgvector, Redis. Alembic migrations; health and readiness endpoints.
- **CI (GitHub Actions):** lint (ruff), types (mypy), unit tests, gateway property tests, smoke eval, Docker build, deploy on `main`. Nightly full eval posts a results table.
- **Config as code:** prompts versioned in git and `prompt_versions`; model tiers and thresholds in config.
- **Hosting:** decided in Week 7 after checking current free/cheap options.
- **Public demo protections:** per-IP rate limit, daily demo budget, graceful "demo busy" mode.
- **Admin UI:** minimal, server-rendered (HTMX/Jinja), about a day of work.

---

## 14. Models and cost model

### Model plan

| Role | Source |
|---|---|
| Intake classifier, Tier 0 resolver | Small local model (3-4B class fits 4 GB VRAM at 4-bit; 7-8B may spill to system RAM) or a fast hosted small model |
| Complex resolver (Tier 1-2) | Hosted free tier (for example Gemini Flash), with a larger open model on Groq or OpenRouter as fallback |
| Judge (evals) | Hosted model of a different family than the resolver; small samples only |
| Embeddings and reranker | Local, so retrieval costs no quota |

Exact models are set in config because free-tier model lists rotate. Free-tier limits change; re-check current limits at implementation time.

### Shadow pricing
Real spend is $0, but every call logs tokens and cost is computed from a price table of reference per-token prices in config. Results are reported "at reference prices".

Rough per-resolved-ticket usage: classifier ~1.2k tokens in; 3 resolver steps totalling ~12-15k tokens in (including a cached static prefix); ~1k tokens out.

Illustrative (placeholder prices, not quotes):

| Reference price (per 1M in/out) | Per ticket | At 10k tickets/day |
|---|---|---|
| $0.15 / $0.60 (small-model class) | ~$0.003 | ~$30 |
| $3 / $15 (frontier class) | ~$0.06 | ~$600 |

The ~20x gap is why routing matters. The README should report measured numbers: share of tickets handled by the small tier, accuracy delta, and savings.

**Cost levers:** routing, prefix caching where supported, context trimming, step caps, cached answers for pure FAQ intents. Track cost per resolved ticket and per escalation, not just the average.

---

## 15. Risks, open questions, milestones

### Risks
- Small or local models too weak for tool calling -> repair-retry, tighter tool schemas, measure the gap.
- Free-tier limits or model availability change -> provider-agnostic layer and fallback chain.
- Eval set quality and label noise -> hand-curate; double-check a sample.
- Scope creep for a solo builder -> vertical slices; cut intents before cutting evals.
- Live demo killed by quota exhaustion -> demo protections (section 13).

### Open questions
- Exact local model choice for the 4 GB GPU (benchmark in Week 1-2).
- Trace viewer (Week 6) and hosting (Week 7).
- Final quality targets after baseline.

### 8-week plan

| Week | Focus | Exit criteria |
|---|---|---|
| 1 | Repo, seeded mock store, Tool Gateway with read tools, LLM adapter, minimal loop | One ticket works end to end with a stored trace |
| 2 | Eval harness v1 (state-based), 60 cases, hard rules plus classifier | Baseline routing and task-success numbers |
| 3 | KB, hybrid retrieval plus reranker, context builder, grounding checks | Retrieval eval meets its target |
| 4 | Tier 1-2 tools, approvals, idempotency, audit log, shadow/auto switch, safety suite v1 | Zero forbidden actions in the red-team set |
| 5 | Queue workers, retries, fallback chain, rate-limit client, DLQ, kill switch, replay | Chaos test: kill a provider, ticket still handled |
| 6 | Tracing, metrics, admin UI, shadow pricing, routing experiments | Cost-vs-accuracy table for routing strategies |
| 7 | CI/CD, Docker, deploy, load test, demo protections | Public demo plus green pipeline |
| 8 | Golden set to 150+, hardening, README with diagram and real numbers, demo video | Buffer and polish |

### Ship gates
- Zero unauthorized money actions
- Tier 3 recall >= 99%
- No ticket lost under the chaos test
- Task success at or above target on the full golden set
