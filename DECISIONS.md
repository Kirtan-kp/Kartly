# Architecture Decision Records

Short ADRs for the Kartly Support Agent. Status is **Accepted** unless noted. Numbering here supersedes the informal numbering used during planning. See `DESIGN.md` for the full context.

---

## ADR-001: Chat-style API only in v1
- **Context:** Supporting email, chat, and social adds adapters without teaching anything new about the agent.
- **Decision:** v1 exposes a chat-style HTTP API. Email becomes a later adapter on the same ticket model.
- **Alternatives:** Email-first; multi-channel from the start.
- **Consequences:** Faster to build and test; channel adapters stay out of the core.

## ADR-002: English only
- **Context:** Multilingual support multiplies the eval set and prompt surface.
- **Decision:** English only in v1.
- **Alternatives:** Hindi/English mix.
- **Consequences:** Smaller eval set; multilingual is a clear future extension.

## ADR-003: Risk tiers and INR 1,000 auto-refund threshold
- **Context:** Autonomy must match risk. Money actions need limits.
- **Decision:** Four tiers (0 read-only, 1 low-risk writes, 2 money, 3 never autonomous). Refunds up to INR 1,000 are automatic; above that require human approval. The threshold is configurable per tier.
- **Alternatives:** Everything needs approval; higher threshold.
- **Consequences:** Routing, permissions, and evals all key off the tier. Threshold can be tuned with data.

## ADR-004: Shadow-first rollout with per-tier mode switch
- **Context:** Real teams do not launch agents at full autonomy.
- **Decision:** Per-tier mode (`off`, `shadow`, `auto`) exists from day one. Order: shadow everywhere, then auto for Tier 0-1, then Tier 2.
- **Alternatives:** Build autonomy only, add drafts later.
- **Consequences:** Draft path is a first-class feature; doubles as a kill switch.

## ADR-005: v1 scope capped at about 8 intents
- **Context:** Each intent multiplies eval cases and tool surface.
- **Decision:** order status, tracking, return, refund, cancel, address change, policy FAQ, escalation.
- **Alternatives:** Broader coverage.
- **Consequences:** Eval set stays manageable; cut intents before cutting evals.

## ADR-006: Workflow with a bounded agent loop, plain Python, no framework
- **Context:** Fully autonomous agents are harder to test and secure. Frameworks hide behavior.
- **Decision:** Deterministic stages around a single bounded LLM tool-calling loop, implemented as a plain Python state machine.
- **Alternatives:** LangGraph or similar; fully autonomous agent.
- **Consequences:** More code to write, fewer hidden behaviors, more learning. A framework can still be discussed as an alternative.

## ADR-007: Tool Gateway as the sole enforcement point
- **Context:** Prompts cannot be trusted to enforce policy against injection.
- **Decision:** The LLM never touches the database. All actions go through a deterministic gateway that does authz, eligibility checks, limits, idempotency, and audit logging. `customer_id` is injected server-side and is never a tool argument. Idempotency keys are generated server-side. Approvals are bound to a payload hash and expire.
- **Alternatives:** Policy in the system prompt; per-tool ad hoc checks.
- **Consequences:** A successful injection cannot cross customer boundaries or double-refund. The gateway needs thorough property tests.

## ADR-008: Hybrid retrieval with a reranker
- **Context:** Policy text contains exact terms ("30 days", "final sale") where pure vector search is weak.
- **Decision:** Postgres full-text plus pgvector, followed by a reranker.
- **Alternatives:** Vectors only; external vector DB.
- **Consequences:** One database to run; retrieval gets its own eval (hit@k, MRR).

## ADR-009: Upfront routing with a hard-rules floor
- **Context:** Cost and risk both depend on classifying early. A small classifier can miss dangerous tickets.
- **Decision:** Keyword/regex hard rules, then an LLM classifier. Final tier is the max of the two, so the classifier can raise risk but never lower it. Cascade routing is a later measured experiment.
- **Alternatives:** Cascade first; classifier only.
- **Consequences:** High recall on Tier 3; routing quality is measurable on its own.

## ADR-010: Hard run limits
- **Context:** Agent loops can burn budget and time.
- **Decision:** Max 6 loop steps, 8 tool calls, a per-run token budget, and a wall-clock timeout. The original $0.30 per-run cap is expressed as tokens, with a shadow-cost equivalent.
- **Alternatives:** Dollar caps only.
- **Consequences:** Limits stay meaningful even when real spend is zero.

## ADR-011: Provider-agnostic LLM adapter
- **Context:** No paid API access; free tiers and local models change often.
- **Decision:** One adapter with an OpenAI-compatible interface (own thin layer or LiteLLM); all models selected by config.
- **Alternatives:** Hard-wire one provider.
- **Consequences:** Models and providers can be swapped without code changes; supports fallback chains.

## ADR-012: Two model tiers on free resources
- **Context:** Student budget; RTX 3050 with 4 GB VRAM.
- **Decision:** A small local model (3-4B class, 4-bit) for the classifier, embeddings, reranker, and smoke evals; hosted free tiers for the complex resolver and the judge. Exact models are config.
- **Alternatives:** Paid APIs; 7-8B local models that spill into system RAM.
- **Consequences:** Weaker small-model tool calling is expected and measured; guardrails and repair-retry matter more.

## ADR-013: Shadow pricing
- **Context:** Real spend is $0, but cost is a core engineering metric.
- **Decision:** Log tokens for every call and compute cost from a reference price table in config. Report results "at reference prices".
- **Alternatives:** Skip cost tracking.
- **Consequences:** Routing and caching experiments produce meaningful numbers.

## ADR-014: Rate-limit-aware client with fallback chain
- **Context:** Free tiers have tight per-minute and per-day limits, and failed requests may still count against quotas.
- **Decision:** Token-bucket client, 429 handling with capped retries and jitter, circuit breaker, and a configurable fallback chain ending at the local model and then the human queue.
- **Alternatives:** Unbounded retries.
- **Consequences:** Reliability work becomes a visible feature; demo needs quota protection.

## ADR-015: Synthetic data only
- **Context:** Some free tiers may use inputs to improve their models.
- **Decision:** No real customer data is ever sent to hosted providers; the store and tickets are synthetic.
- **Alternatives:** None considered.
- **Consequences:** Safe to use free tiers; realism comes from careful curation.

## ADR-016: State-based grading first, LLM judge second
- **Context:** Reply text is a poor proxy for whether the right action happened.
- **Decision:** Primary grading is final database state and tool-call correctness. An LLM judge scores quality on a rubric, uses a different model family than the resolver, and is calibrated against ~30 human labels.
- **Alternatives:** Judge-only grading.
- **Consequences:** Cheaper, more reliable evals; judge agreement is reported.

## ADR-017: CI uses a smoke set with record/replay
- **Context:** Free-tier quotas cannot support a full eval on every PR.
- **Decision:** ~25-case smoke set on every PR using recorded LLM responses; full golden set nightly or manually. Safety metrics are hard gates.
- **Alternatives:** Full eval in CI.
- **Consequences:** Deterministic, free PR checks; full-eval drift caught nightly.

## ADR-018: Own run_steps plus OpenTelemetry; viewer decided in Week 6
- **Context:** Self-hosted Langfuse runs several heavy services; a laptop is constrained.
- **Decision:** Persist `run_steps` in Postgres and emit OpenTelemetry now. Choose a lightweight viewer (for example Arize Phoenix) or hosted free tier at the observability milestone.
- **Alternatives:** Self-hosted Langfuse from the start.
- **Consequences:** Tooling choice is deferred without blocking replay and debugging.

## ADR-019: Hosting decision deferred to Week 7
- **Context:** Free/cheap hosting options change frequently.
- **Decision:** Check current options at the deployment milestone. Design stays container-based and portable.
- **Alternatives:** Pick a host now.
- **Consequences:** No lock-in; deployment is verified by CI and a load test.

## ADR-020: 8-week plan with Week 8 as buffer
- **Context:** Solo build with real risk of overruns.
- **Decision:** Seven weeks of milestones, each ending in something runnable, plus one week for the golden set, hardening, and README.
- **Alternatives:** Plan without buffer.
- **Consequences:** Intents are cut before evals if time slips.