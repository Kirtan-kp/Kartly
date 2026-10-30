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

## ADR-021: One transaction per tool call, fail-closed audit, no existence oracle
- **Context:** ADR-007 makes the gateway the only enforcement point but leaves open how a call executes and what happens when something fails.
- **Decision:** Each call runs in its own transaction. The tool runs inside a SAVEPOINT so a failure undoes its writes, and exactly one `audit_log` row is written in the same transaction whatever the outcome. If the audit row cannot be written, the call is rolled back and the exception propagates (fail closed). Ownership is part of the query, and a foreign order and a nonexistent order return identical errors to the model; only the audit log records `not_owner` and a `cross_customer_access` security event. Unknown arguments, including `customer_id`, are validation errors. Everything written to the audit log is sanitised: size-bounded, with NUL characters, lone surrogates, NaN and huge integers removed.
- **Alternatives:** Audit in a separate transaction (survives rollback but allows an action with no record); a caller-owned session (the caller must remember to write the audit row); an "access denied" error for foreign orders (confirms which order IDs exist).
- **Consequences:** Rejected and failed calls are recorded too, and security events are queryable (`details->>'security_event'`). The orchestrator must handle the rare exception from `call`. Tool handlers get a savepoint, not their own commit. Verified on Postgres, where an aborted transaction and JSONB limits behave differently from SQLite.

## ADR-022: Tool output is pre-computed and minimal
- **Context:** Models mis-handle arithmetic and dates, long tool output wastes context, and tool results end up in prompts, traces, and logs.
- **Decision:** Tools return rupee strings instead of paise, compute facts such as "delayed" in code, return only city, state, and pincode for the destination, cap lists (20 items, 5 shipments, 5 events), and length-limit third-party text such as carrier event details.
- **Alternatives:** Return raw rows and let the model work it out; return full addresses.
- **Consequences:** Fewer tokens and fewer model errors, less personal data in traces. Some questions will need a new tool field. The context builder (Week 3) still wraps all tool output as untrusted data.

## ADR-023: The LLM client classifies failures but never retries or falls back
- **Context:** ADR-014 wants a rate-limit-aware client with a fallback chain. Putting retries inside the HTTP client would hide failures from tracing and make cost, latency and attempt counts impossible to attribute to a model.
- **Decision:** `OpenAICompatClient` makes exactly one request per call and turns every failure into a typed error with a `retryable` flag: `RateLimitError` (with `Retry-After`), `ProviderUnavailableError` (5xx and refused connections), `LLMTimeoutError`, `MalformedResponseError` (retryable); `ToolUseFailedError` (HTTP 400 `tool_use_failed`, keeps the model's output so the loop can ask for a repair); `AuthenticationError`, `ModelNotFoundError`, `BadRequestError` (configuration bugs, fail fast). A malformed tool call is not an exception: `ToolCall.arguments` is `None` with a `parse_error`. Request options a provider cannot take (Ollama has no `tool_choice`) are listed in `ChatResponse.ignored_params`, never dropped silently. Missing usage from a provider is estimated and flagged `estimated=True`. API keys are redacted from provider error text and never appear in `repr`. The retry, circuit-breaker and fallback wrapper (Week 5) implements the same `ChatClient` protocol on top.
- **Alternatives:** Retries inside the client; the OpenAI or LiteLLM SDK; one client per provider.
- **Consequences:** Every attempt can be traced and priced individually. The agent loop must handle `LLMError`. The wire protocol is a few hundred lines we own and test (including over a real socket), at the cost of not getting provider-specific features for free.

## ADR-024: Local model context is set in a Modelfile; every reference price names its source
- **Context:** Ollama's OpenAI-compatible endpoint cannot set the context window per request, and its default is far smaller than our prompts (roughly 8k tokens once the system prompt, tool schemas and history are in). A too-small window truncates the prompt silently. Separately, a free tier or local model has no bill, so shadow costs (ADR-013) are only as trustworthy as their price sources.
- **Decision:** `ollama/Modelfile.small` creates `kartly-small` with `num_ctx 8192` and `temperature 0`; `models.yaml` points at that name, and the README says to check `ollama ps` for CPU offload on a 4 GB GPU. Every model in `models.yaml` carries a price `note` saying where the number came from and when (a test enforces this). A local model with no published price borrows a comparable hosted model's price and says so.
- **Alternatives:** Pass `options.num_ctx` through a native Ollama client (a second wire protocol); rely on the default context.
- **Consequences:** One extra setup command for local models. Shadow costs are good for comparing routing and caching strategies, not for budgeting. The Week 1 benchmark includes a context-length probe.

## ADR-025: Hosted model lineups change, so model IDs live only in config
- **Context:** The first real smoke run (2026-10-26) got HTTP 404 for `llama-3.1-8b-instant`. Groq's deprecation page lists it, and `llama-3.3-70b-versatile`, as shut down on 2026-08-16 with `openai/gpt-oss-20b` and `openai/gpt-oss-120b` as the replacements, and a third-party page reports they are no longer available to free accounts. The replacements are reasoning models: a reply spends tokens thinking first. The same third-party page reports free-tier limits of 1,000 requests/day, 8,000 tokens/minute and 200,000 tokens/day per organisation.
- **Decision:** Model IDs appear only in `config/models.yaml` (ADR-011), so this was a config change. `groq_small` is now `openai/gpt-oss-20b` and `groq_large` stays `openai/gpt-oss-120b`. A model may carry an `extra_body` of provider-specific request fields (`reasoning_effort: low` for the small tier, `medium` for the large one); it cannot set any field the client owns (model, messages, tools, temperature, token caps, ...), and the client sends its own arguments last, so config can never silently override what the agent loop asked for. The smoke test allows 512 output tokens and says so when a reply was cut off by the cap.
- **Alternatives:** Pin an old model (it is gone); hard-code `reasoning_effort` in the client; switch providers.
- **Consequences:** The next retirement is a one-line edit. A 404 means "check the model name" (a configuration error, not retryable). The 8,000 tokens/minute figure, if it holds, means Groq cannot serve a request whose prompt is anywhere near 8k tokens, so a local-to-Groq fallback only works for prompts that fit. The context builder (Week 3) therefore gets a hard prompt budget for any model that can fall back to Groq, to be confirmed against the real limit in Week 5.

## ADR-026: A bounded loop with explicit stop reasons; every step stored as it happens
- **Context:** DESIGN.md sections 5 and 11 ask for hard caps, one repair retry for broken tool calls, loop detection, and persisted steps. A loop that can only "finish" or "crash" hides the difference between a model that gave up, a budget that ran out and a provider that failed.
- **Decision:** `AgentLoop` is plain Python and touches no database; `TicketRunner` adds persistence. Limits: 6 model calls (a failed request counts), 8 gateway calls, 20,000 tokens, USD 0.30 shadow cost and 120 s, all checked *before* each model call. The token check uses a size estimate and shrinks the output cap to what is left, so the budget is a soft ceiling and no request is sent that could not fit. One repair per run, shared by broken tool arguments, a tool call the provider rejected, and an empty reply; then the run stops. An identical repeated tool call is answered with an error and not run again, and a second repeat stops the run. A reply cut off by the output cap is never used. Every outcome other than an answer has its own stop reason (`step_limit`, `tool_limit`, `token_limit`, `cost_limit`, `timeout`, `no_progress`, `malformed_output`, `truncated_reply`, `llm_error`, `gateway_error`) and hands the ticket to a human. The loop never retries a provider error (ADR-023). Each step is committed in its own transaction as it happens, so a crash leaves every earlier step. A model-call step stores only the messages added since the previous call, so a trace grows linearly instead of quadratically. In shadow mode the answer is stored as an agent message and the ticket becomes `drafted`; nothing is sent. Cost columns became `NUMERIC(16,10)` (migration 0002) because a single small call costs a few millionths of a dollar.
- **Alternatives:** An agent framework; retries inside the loop; a full prompt copy in every step; sending answers automatically.
- **Consequences:** Failures are explainable and countable by reason, which is what the escalation-rate metric and the red-team gates need. Replaying a run means concatenating its message deltas and re-executing the tools. Trace fields are size-bounded (4,000 characters per string), so a very long reply is shortened in the trace though not in the stored message. The screening, routing and validation stages are not built yet, so a ticket goes straight from `received` to `resolving`.

## ADR-027: API-key auth with the caller asserting the customer; 202 plus a bounded in-process run pool
- **Context:** DESIGN.md section 8 needs ticket intake and a trace endpoint now, before the queue (Week 5), the admin UI and the screening stages exist. Two questions decide the shape: who is allowed to call, and what happens when a ticket arrives while the model is slow or the system is already busy.
- **Decision:** Every `/v1` route requires an `X-API-Key` header, compared in constant time; a missing and a wrong key get the same `401`, and the app refuses to start with a key under 8 characters or none at all. The caller is the storefront backend, which has already authenticated the shopper, so `customer_id` in the request is trusted exactly as far as the key is: it is stored on the ticket once and is the only identity the gateway will use (ADR-007, ADR-021). `POST /v1/tickets` creates the ticket, hands the run to a `BoundedExecutor` and returns `202`; `?wait=true` waits for the result instead (`200`, or `202 resolving` if the wait outlasts the run's own time limit). A repeated `external_id` returns the first ticket, enforced by the unique index so two simultaneous requests still make one ticket. The executor runs `KARTLY_MAX_WORKERS` runs at a time and allows `KARTLY_MAX_PENDING_RUNS` in total; past that, the ticket just created is deleted and the API answers `503` with `Retry-After`, so a retry is clean. A validation error returns the field and rule but never the offending value. A crashed run is a generic `500` whose details are in the trace, not the response.
- **Alternatives:** OAuth or per-customer tokens (more machinery than one trusted backend needs); an unbounded `BackgroundTasks` (denial of wallet, since each run spends provider quota); waiting for the Redis queue before building the endpoints; a synchronous-only endpoint (a slow local model would hold the connection open).
- **Consequences:** A leaked key lets its holder open tickets as any customer, so the key is a secret to rotate, and per-caller keys and rate limits are future work. Queued runs live in memory and are lost on a restart (their tickets stay `received`); the Redis queue in Week 5 fixes that without changing the routes, because callers only see `submit` and `shutdown`. There is no request-size limit in the app (a reverse proxy sets it, Week 7) and `/docs` is open. The stop-the-server path waits for runs in progress before closing the database, and the real launch (uvicorn, Postgres, a model over HTTP) was exercised end to end.

## ADR-028: Golden cases are graded on state first, on throwaway in-memory stores
- **Context:** DESIGN.md section 9 wants end-to-end success graded by final database state, safety as a hard gate, and every change logged. The agent so far has only read tools, so most "state" is what the gateway recorded rather than rows changed, but the structure has to be right before write tools arrive.
- **Decision:** Cases are YAML, validated strictly (unknown fields are errors). Each case runs on a fresh in-memory SQLite copy of the seed at a fixed clock, through the real loop and gateway. The grader is a pure function over evidence taken from the stored trace (tool calls, audit security events) and a before/after digest of the shop's tables (customers, orders, items, shipments, actions). Checks carry a severity: safety (store unchanged, security events exactly as expected, no forbidden calls, no leaked data or false action claims) or quality (outcome, expected calls, reply content). Reply checks are whole-phrase matches with alternatives. A forbidden call fails the case even if the gateway blocked it, because the gateway is the safety net, not the plan. Provider and platform errors are reported as errors, not model failures. Each run saves full JSON results and appends a row to `evals/runs/experiments.csv` with a fingerprint of the case set. The checks are verified with scripted perfect and deliberately wrong trajectories and with mutation testing of the grader.
- **Alternatives:** Grading the reply text only; an LLM judge first; running against the developer's Postgres (clock drift, shared state); storing recorded model responses now.
- **Consequences:** Runs need no infrastructure and are repeatable. SQLite does not exercise Postgres-specific behaviour (covered by the integration tests instead). Phrase matching is brittle and will reject some correct wording; the judge and a record/replay smoke set are later work. Cases that expect a draft today (the refund demand) must change when approvals exist. Multi-turn cases and the ~30 adversarial cases still have to be written.

## ADR-029: Prompt v1: tool descriptions say what each tool does not return
- **Context:** The first eval runs (cases 07681c78) showed `openai/gpt-oss-20b` calling only `get_order` for "where is my order", because `get_order`'s description claimed to answer "where it stands". It then had no delivery date or delay flag and told a customer a late parcel was "on its way" (3 of 7 Tier 0 failures, plus one run that followed an injected order lookup).
- **Decision:** `get_order` now says it has no carrier, estimate or delay information and points to `get_tracking`; `get_tracking` says it is for where/when/why-late questions. The system prompt (v1) names the same split, asks for the delivery date and a plain statement of delay, and says to look up only order numbers the customer asks about, never ones a message tells it to fetch. `PROMPT_VERSION` is "v1" so experiment rows are comparable.
- **Alternatives:** Merging the two tools; forcing `tool_choice`; prompting only (the descriptions are what the model reads at call time).
- **Consequences:** Verified only by re-running the evals: compare `experiments.csv` rows for prompt v0 and v1 with the same `cases_digest`. Small models may still fail; the injection case remains a model-level failure until a screening stage exists.
