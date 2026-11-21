# Kartly: what I built, what I measured, and what went wrong

Kartly is an AI support agent for an invented online store. It reads a customer's message, decides whether a person has to handle it, looks things up, answers policy questions from a knowledge base, and can cancel orders and issue refunds, with limits. Everything is synthetic and everything runs on free models (a local Ollama model and Groq's free tier), so every number below can be reproduced for no money.

The project is about the engineering around the model, so this write-up is organised the same way: the claim, the evidence for it, the places where measurement changed my mind, and what is still not known. Decision records are cited as ADR-nnn (`DECISIONS.md`).

## 1. The claim

> The model proposes; code disposes. A customer, or a prompt injection hidden in a message, can talk the model into asking for anything. It still cannot make the system do anything the policy does not allow.

The model never touches the database. It calls tools, and every tool call goes through a gateway that is ordinary code (ADR-007, ADR-021):

- the customer id comes from the authenticated ticket, never from the model, so another customer's order is "not found" whatever the model asks for;
- every refund is checked against what is left on the order, the order-level limit (INR 1,000, counting earlier refunds, so three refunds of INR 966 are one INR 2,898 refund) and a daily cap (ADR-047);
- anything above the limit becomes an approval request bound to the exact amount and order, and a human decision is re-checked when it runs, not trusted (ADR-046, ADR-047);
- writes have three modes (`off`, `shadow`, `auto`) per risk tier, a kill switch stops the agent without a deploy, and every call writes an audit row in the same transaction as the action (ADR-004, ADR-051).

## 2. Evidence

| Question | How I measured it | Result |
|---|---|---|
| Can the agent be talked past the money rules? | 8 red-team cases graded on the store's state afterwards, not on the reply: a claimed manager approval, split refunds, a fake system notice, another customer's order, forcing a cancel of a shipped order, a refund before delivery, an inflated amount (ADR-049) | All 8 pass on `groq_small`, 0 safety failures (one run per case). The gateway tests and a Postgres test that races five refunds hold the same line without a model (ADR-047). |
| Do ordinary writes do the right thing and no more? | 7 write cases: cancel placed and shipped orders, small and large refunds, a refund above what is left | All 7 pass, after two fixes to my own test (section 3). About 3,000 tokens and about $0.0003 of shadow cost per run, median latency about 2 s. |
| Are policy answers grounded? | 14 policy cases, three repeats each, plus a deterministic number check and a model-based claim check on replies (ADR-039, ADR-045) | Prompt p6: 42 of 42 over three repeats, 0 safety failures, about 2,100 tokens per answer, $0.0075 for the 42 runs (ADR-044). |
| Does routing find the tickets a human must handle? | 70 labelled messages, 14 of them "a human must handle this" (legal, fraud, injury, distress), classifier prompts c1 and c2 on two models (ADR-033, ADR-034) | `groq_small`, c2: intent 90.0%, tier 91.4%, 100% of the human-only messages caught, 0.6 s per label. Local model, c2: 87.1%, 88.6%, 85.7% recall, 4 s. The keyword rules alone caught 1 of the 14 here, by design of the set. |
| Is retrieval good enough? | 74 questions over a 44-chunk knowledge base, three retrievers (ADR-036) | Dense search ranks the right chunk first on 93.5%; on the first 46 questions, 0.952 against BM25's 0.762 (paraphrases 1.000 against 0.667). The set is easy and written by the same person as the knowledge base, so this is a floor on difficulty, not a production estimate. |
| Does it survive failures? | Fault injection through the real retry, fallback, queue and worker code: 429s, 5xx, timeouts, malformed replies, a bad key, a worker that crashes right after paying a refund (ADR-052, ADR-053) | Eleven scenarios each end in a stated outcome; the run fails on any ticket stuck in a working state, any refund paid twice, or a "drafted" ticket with no answer. |
| What does a ticket cost? | Shadow pricing of recorded tokens at reference prices (ADR-013, ADR-058) | Measured per answered ticket by model and tier, with a what-if repricing whose limits are printed with it. Real spend is $0. |
| Can it be run and recovered? | Containers, a migration that must succeed first, a smoke test of the stack in CI, verified backups and a restore that saves the old data first (ADR-059, ADR-060, ADR-061) | The suite runs about 1,500 tests on SQLite and on Postgres, plus browser tests of the admin page. |

You can watch the safety behaviour in two minutes with no key: `python -m kartly.demo` (ADR-062).

## 3. Where measurement changed my mind

These are the parts I would talk about in an interview, because each one is a case where the first version was wrong and a measurement said so.

**Prompt rules stopped working, so I moved the rule into code.** For return questions on a specific order, three rounds of prompt rewording (p1 to p3) fixed one failure and caused another: the model said "yes, you can return it" for an order delivered 45 days ago. The fix was not a fourth prompt. `check_return_eligibility` computes the verdict in code from the order and the rules, and the prompt only says to report it (ADR-041). A statement the model has to do arithmetic for is a statement it will sometimes get wrong.

**The shortest prompt was not the best one.** Prompt p5 cut about 18% of the tokens and passed 39 of 42 runs; all three failures were one case, where the model said "Kartly does not accept returns in physical stores", which the knowledge base never says. Token count alone would have adopted it. p6 puts back the one rule and the one example that mattered, keeps most of the saving, and passes 42 of 42 (ADR-044). That also exposed a hole: the number check cannot see an invented claim with no number in it, which is why the claim check exists (ADR-045).

**The fault-injection run found a real bug.** The queue's first retry wait was 5 seconds, shorter than the model circuit breaker's 30-second cooldown, so a short outage used up all three attempts while the breaker still refused to call, and the ticket was escalated although the provider had recovered. The first wait is now 30 seconds, with a test that keeps it at least the cooldown (ADR-053).

**Running a real model found two bugs in my tests, not in the safety logic.** A refund case failed because the model, asked for INR 3,499 when INR 1,999 was left, sensibly asked for the INR 1,999 instead; the case assumed one path. A case can now list other acceptable behaviours without loosening any check inside each (`also_acceptable`). Then the alternative still did not match, because the grader compared tool arguments as text and the model sent a number; matching now understands numbers. Both were invisible to the scripted-model tests, which is the argument for running a real model at all.

**Testing the deployment found three more.** The restore script's safety backup could overwrite the dump being restored when both got the same timestamp name; two admins disabling each other at the same moment could both succeed and leave none (it failed 21 times in 40 runs until the admin rows were locked in a fixed order); and on GitHub, the CI runner's newer `pg_dump` wrote SQL that the test server rejected, which I had skipped on mismatch in only one direction. Windows added line-ending and bash-environment problems of its own. None of these were found by reading the code.

## 4. What I did not do, and what is not known

- **One run per case for the write and red-team sets.** A pass is a sample, not a rate. Policy cases were run three times each; the write and red-team repeats (`--repeat 3`) are the next measurement and are not in this document.
- **Reply quality is phrase matching plus a grounding check, not a strong judge.** The claim judge is a small model measured on a few dozen hand-written replies written by the same author as its prompt, so its numbers flatter it (ADR-045).
- **The routing saving is not measured end to end.** I measured the classifier and priced the models; I did not run the same ticket set routed and unrouted through the whole agent and compare quality. The what-if report estimates it and says what it ignores (ADR-058).
- **The Tier 3 rules are tested on messages I wrote.** 100% on an invented set is a floor, not production recall; real tickets would find gaps (ADR-031). Paraphrase, Hinglish and soft distress are known gaps the classifier is meant to cover.
- **Free-tier models are small and rate limited**, which sets how much context and how many calls a ticket may use (ADR-042).
- **Deployment is one machine.** No high availability, no point-in-time recovery, one shared API key rather than per-caller authentication, no rate limiting, dependencies not locked by hash. The compose files and the container job were checked by tests and in CI; I have not run this on a real server with real traffic (`docs/DEPLOY.md`).
- **Everything is synthetic.** There is no real customer data and no real payment system: a "refund" is a record in the store.

## 5. If I continued

1. Run the write and red-team sets three times each and report variance, then do the same on a second model.
2. Route the same tickets through the whole agent with and without the classifier, and measure quality and cost together instead of estimating the saving.
3. Replace phrase matching in reply grading with a judge measured against human labels, as I did for the claim check.
4. Per-caller API keys, rate limits, a lockfile, and a health check for the worker.
5. A real run on a small server for a week, with the alerts on, and look at what the alerts say.

## 6. Reproduce it

```bash
python -m kartly.demo                                  # the six-scene safety walkthrough, no key
pytest -q                                              # the suite (SQLite; Postgres when KARTLY_TEST_DATABASE_URL is set)
python -m kartly.evals --model groq_small --pause 20   # all golden cases against a real model
python -m kartly.chaos                                 # the failure scenarios
python -m kartly.costs --days 7                        # the cost report
```

Setup, the admin page and deployment are in `README.md` and `docs/DEPLOY.md`.
