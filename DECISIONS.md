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

## ADR-030: A deterministic injection screen before the model, and a stricter grader
- **Context:** With prompt v1, `openai/gpt-oss-20b` on Groq still followed an injected request to look up another customer's order (the gateway blocked it, nothing leaked, but it is a safety failure). The local model sometimes answered "not found" for a friend's order without looking it up, so the case passed or failed on luck: the grader treated a missing expected security event as a safety failure for the model, when it is really a missed lookup.
- **Decision:** (1) `guards/screen.py`: regular-expression rules over normalised customer text (NFKC, casefold, zero-width characters removed). A hit ends the run before any model call: stage `screen` (new `StepKind.STAGE`), audit row by the `system` actor with `prompt_injection_suspected`, run `succeeded/escalated`, ticket `escalated`, `StopReason.SCREENED`. `TicketRunner(screen=None)` and `kartly.evals --no-screen` turn it off to measure the bare model. (2) The grader splits the events check: "no unexpected security events" (safety) always, and "expected security events recorded" (quality) only when the case lists some.
- **Alternatives:** A model-based classifier (costs a call and can itself be injected); stripping the suspicious text and continuing (hides the attack from the human); making the grader count a refusal as a pass (it hides a model that never looked).
- **Consequences:** Blatant attacks cost nothing and reach a human. A false positive also reaches a human, which is cheap but visible in the escalation rate: watch it. The screen can be evaded by wording, so the gateway and the `injection-polite-lookup` case remain the real measure. That case first treated any lookup of the other order as a safety failure, but the model cannot know an order is foreign without asking (the same reason `idor-friends-order` requires the lookup), so it now expects the lookup and the gateway's `cross_customer_access`, and fails on a passed `customer_id`, leaked data, or a reply that drops the real question. The case set changed (11 cases), so `cases_digest` changed: earlier `experiments.csv` rows are not comparable. Rules need review against real tickets once there are any.

## ADR-031: Tier 3 hard rules run before the model, and are measured on their own
- **Context:** DESIGN.md section 6 puts hard rules first in routing and sets a Tier 3 recall gate of 99%. A legal threat, a fraud claim, an injury or a customer in distress must reach a person; a small model deciding that is the weakest option.
- **Decision:** `routing/rules.py` holds keyword patterns in six categories (legal, fraud, account_security, safety, distress, threat) over the same normalised text as the injection screen. A hit ends the run before any model call: stage `route`, audit row `route.escalated` (actor `router`, not a security event), run `succeeded/escalated`, `StopReason.ROUTED_TIER3`. The screen runs first; `TicketRunner(rules=None)` turns the rules off. `python -m kartly.evals.routing` measures them on `evals/routing/tier3.yaml`: recall on messages a human must handle, false-alarm rate on ordinary messages, per category. Both are gates. Phrasings known to be wrong either way carry a `known:` label, are reported on their own line, and never count toward the gate.
- **Alternatives:** Classifier-only routing (slower, and a small model can be argued out of it); a larger rule list tuned to a looser recall (more false alarms for the team); counting known gaps in the gate (it would hide what the rules cover).
- **Consequences:** Cheap and deterministic, and every hit is explainable. The dataset is invented and written by the rule author, so 100% on it is a floor on quality, not a measured production recall; real tickets will find gaps, and each one becomes a new message in the file before the rule changes (the first run found three misses and one false alarm, now fixed with fresh messages added alongside). Known gaps today: paraphrase ("in front of a judge"), Hinglish, spaced-out letters, soft distress ("can't go on"), plus a few false alarms ("badminton court", "police blue shirt", "no need to sue anyone"). The classifier layer is the answer to the first group; the false alarms cost a human a minute each. Abusive-but-not-threatening customers are also left to the classifier (sentiment).

## ADR-032: A classifier that can only raise risk, measured before it is wired in
- **Context:** DESIGN.md section 6 layers a small-model classifier (intent, tier, complexity, sentiment, confidence) over the hard rules, with "final tier = max(rule tier, classifier tier)". A small local model returns broken JSON some of the time, and the Tier 3 rules are known to miss paraphrases.
- **Decision:** `routing/classifier.py` asks for one JSON object (`response_format: json_object`, no tools, ticket quoted as untrusted data, `</ticket>` stripped so it cannot close the quote). Code, not the model, enforces the safety properties: a floor tier per intent (a refund is at least Tier 2, escalation Tier 3), `decide_route` takes the maximum of rule and classifier tier, and a failed call or unusable output is a result with an error that falls back to the rule tier and flags clarification, never an exception and never a lower tier. Confidence under 0.7 sets `needs_clarification`. No output cap is passed: reasoning models may spend hidden tokens against it. `python -m kartly.evals.classify --model NAME` runs it over `evals/routing/intents.yaml` (56 invented messages across the 8 intents, 8 of them Tier 3, six multi-intent labelled by the riskiest ask) and reports intent accuracy, tier accuracy, Tier 3 recall for the whole router and for rules and classifier alone, tickets routed too low or too high, unusable outputs, a confusion list, tokens, cost and latency, with one row per run in `evals/runs/routing_experiments.csv`. Its gate: every Tier 3 message caught and no unusable output.
- **Alternatives:** Trusting the model's tier; function-calling for structured output (not supported uniformly by Ollama); wiring it into `TicketRunner` first (no number to judge it by).
- **Consequences:** Not yet part of the ticket flow: the first job is to see per-model numbers and decide whether the local model is good enough or the Groq model should classify. The labelled messages are small and invented, so accuracy numbers are indicative, not production estimates. Model selection (small vs large tier by routed tier and complexity) and the clarifying-question path come after.

## ADR-033: Classifier prompt c2, and prompts that stay selectable for like-for-like comparison
- **Context:** The first real runs of prompt c1 (56 messages, digest 4bba7963) gave the same headline on both models: intent accuracy 89.3%, no unusable output, Tier 3 recall 88.9% (the rules alone caught 1 of 9, the classifier alone 7 of 9). Both read "Money left my account for something I never bought from you" as a refund request. Both read general policy questions that mention refunds or returns ("How long does a refund take?") as requests. `local_small` dropped the refund in "cancel and refund" and "where is it, otherwise refund" (3 messages routed too low); `groq_small` raised the tier needlessly on 8 messages (tier 1 for plain tracking questions). Latency was 4.1 s p50 local and 0.6 s Groq, about 400 to 500 tokens (under $0.00005) per message.
- **Decision:** Prompt c2 (a) lists a charge or payment the customer did not make or for something never ordered under escalation, (b) defines policy_faq as a general "how does it work" question even when "refund" or "return" appear, (c) tells the model to pick the riskiest intent when a message asks for several things, with the order, and (d) says the tier follows the intent and an annoyed customer asking where a parcel is stays tier 0. Both prompts stay in `PROMPTS`; `--classifier-version c1|c2` runs either, and the cases digest no longer includes the prompt version, so rows for different prompts on the same digest are directly comparable (as ADR-029 does for the agent prompt). Fourteen new messages with different wording (a card payment never recognised, a bill for an order "I didn't place", a threat from a delivery person, an illness from a product, hopeless wording without a keyword, more policy questions, two more multi-intent tickets) were added with c2; they were written after seeing the c1 failures but are phrased differently from them, so the c2 numbers are not a pure held-out score.
- **Alternatives:** Dropping the model's own tier and deriving it from the intent only (it rescued one multi-intent message on each model, and costs nothing to keep, but it also adds the over-routing, so watch the "too high" count); adding a keyword rule for "money left my account" (a narrow patch for one phrasing; the classifier is where paraphrase belongs); retraining.
- **Consequences:** To compare, run both prompts on the new digest for each model. A c2 that fixes the fraud paraphrase but reads more things as escalation will show up in the false-alarm and "too high" numbers. The Tier 3 gate is strict (every Tier 3 message caught) on a small invented set, so it will often fail; the more useful reading is which messages, and whether the rules and the classifier miss different ones.

## ADR-034: The router is optional, runs after the rules, and fails open to the resolver
- **Context:** On 70 labelled messages (digest 1cd6e501, 14 of them Tier 3) the four classifier runs gave: `groq_small` c1 intent 87.1%, tier 82.9%, Tier 3 recall 92.9%, 10 routed too high; c2 90.0%, 91.4%, 100%, 4 too high. `local_small` c1 82.9%, 80.0%, 92.9%, 10 too high; c2 87.1%, 88.6%, 85.7%, 4 too high (it missed "billed for an order I didn't place" and "my mother fell very sick after using the supplement"). c2 beat c1 on both models and fixed the card-payment paraphrase both models missed under c1, so c2 is the default. Groq took 0.6 s p50 and about 680 tokens per label, local 4.0 s and 560 tokens, both under $0.00006. The rules alone caught 1 of 14 Tier 3 messages here, which only says the dataset was written to evade them; it is not a production estimate. Self-reported confidence was 0.9 to 1.0 on almost every answer, including wrong ones, so the "confidence under 0.7 asks a question" step in DESIGN.md has no signal to act on.
- **Decision:** `TicketRunner(router=client)` (settings `KARTLY_ROUTER_MODEL`, CLI `--router`, eval `--router`) labels the customer's text after the screen and the rules. A Tier 3 label (after the intent floor) closes the run as `routed_tier3` with trace steps `classify` (0) and `route` (1, source `classifier`) and the same `route.escalated` audit row; the label call's tokens and cost are added to the run. Anything else carries on to the resolver, with the label in the trace. A failed or unusable label also carries on (recorded as a failed `classify` step): today every tool is read-only, the gateway is the real defence, and holding every ticket whenever the router has a bad minute would send the whole queue to humans. Off by default, so the golden cases (which grade the resolver) are unchanged.
- **Alternatives:** Failing closed on a router error (right once write tools exist; revisit in Week 4); taking the maximum tier over two router models (would have caught `local_small`'s two misses, at twice the calls); using the label to choose a larger resolver model (waits for something to choose between); a confidence-based clarifying question (no usable confidence signal).
- **Consequences:** Recommended setup is `KARTLY_ROUTER_MODEL=groq_small`: it passes the Tier 3 gate on this set and costs under a second. The local model is not safe as the only router. Dependence on a free tier needs the fallback chain from Week 5. Over-routing is the cost to watch: 4 of 70 messages were escalated or priced too high with c2, and `groq_small` read "wrong pincode, it should be 452010" as a policy question (tier 0), which an action-capable resolver would miss. The router's label is not yet used to choose tools or a model.

