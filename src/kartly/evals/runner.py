"""Run golden cases against a model and summarise the results.

Every case gets its own throwaway in-memory database holding the standard seed at a fixed
clock, so cases cannot affect each other, the fixture orders never drift ("delayed" stays
delayed) and your real database is never touched. The real ``TicketRunner``, agent loop and Tool
Gateway run unchanged; only the model is whatever you pass in.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from kartly.evals.cases import Case
from kartly.evals.grader import Check, Evidence, ToolCallEvidence, grade, passed
from kartly.gateway import Mode, ToolGateway
from kartly.guards import screen_text
from kartly.guards.claims import ClaimJudge
from kartly.llm import ChatClient
from kartly.orchestrator import RunBudget, TicketRunner, load_trace
from kartly.orchestrator.prompts import DEFAULT_POLICY_PROMPT, prompt_version
from kartly.sanitize import clean_str
from kartly.store.models import Action, Approval, Customer, Order, OrderItem, Shipment
from kartly.store.seed import seed_mock_store
from kartly.store.sqlite import make_sqlite_engine
from kartly.tools import RiskTier, Tool, default_tools, write_tools

EVAL_NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)
# What the agent must not change while answering a question. Tickets, messages, runs and the
# audit log are expected to change; these hold the shop's own facts.
STORE_TABLES: tuple[type[Any], ...] = (Customer, Order, OrderItem, Shipment, Action)
# The run broke for reasons that say nothing about the model's answer (provider down, rate
# limited, platform fault). Reported as errors and left out of pass rates.
ERROR_STOPS = frozenset({"llm_error", "gateway_error", "crash"})
# Gateway answers that mean the model's call was malformed, as opposed to well-formed but refused.
INVALID_CALL_CODES = frozenset({"tool_unavailable", "invalid_arguments", "duplicate_call"})

SessionFactory = sessionmaker[Session]


@dataclass
class CaseResult:
    case_id: str
    tier: int
    attempt: int
    status: str  # pass | fail | error
    stop_reason: str
    detail: str = ""
    checks: list[Check] = field(default_factory=list)
    answer: str | None = None
    tool_calls: list[ToolCallEvidence] = field(default_factory=list)
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Decimal = Decimal(0)
    latency_s: float = 0.0
    trace: dict[str, Any] | None = None

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def safety_failed(self) -> bool:
        return any(c.severity == "safety" and not c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "tier": self.tier,
            "attempt": self.attempt,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "detail": self.detail,
            "checks": [
                {"name": c.name, "severity": c.severity, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
            "answer": self.answer,
            "tool_calls": [
                {"name": t.name, "arguments": t.arguments, "ok": t.ok, "error_code": t.error_code}
                for t in self.tool_calls
            ],
            "llm_calls": self.llm_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": f"{self.cost_usd:.10f}",
            "latency_s": round(self.latency_s, 3),
            "trace": self.trace,
        }


# ------------------------------------------------------------------------------ one case


def snapshot_store(session: Session) -> dict[str, str]:
    """A digest per store table: equal before and after means nothing in it changed."""
    digests: dict[str, str] = {}
    for model in STORE_TABLES:
        columns = [c.key for c in inspect(model).columns]
        rows = sorted(
            repr([getattr(row, key) for key in columns]) for row in session.scalars(select(model))
        )
        digests[model.__tablename__] = hashlib.sha256("\n".join(rows).encode()).hexdigest()
    return digests


def read_state(session: Session, case: Case) -> dict[str, object]:
    """The order values a case asked to check, as they are after the run."""
    values: dict[str, object] = {}
    for check in case.expect.state:
        order = session.get(Order, check.id)
        value: object = None
        if order is not None:
            value = getattr(order, check.field)
            value = getattr(value, "value", value)  # enums to their stored text
        values[f"orders.{check.id}.{check.field}"] = value
    return values


def _evidence(
    trace: dict[str, Any],
    outcome: str,
    answer: str | None,
    changed: list[str],
    approvals: int = 0,
    state: dict[str, object] | None = None,
) -> Evidence:
    calls = []
    for step in trace["steps"]:
        if step["kind"] != "tool_call":
            continue
        arguments = (step["input"] or {}).get("arguments")
        output = step["output"] or {}
        calls.append(
            ToolCallEvidence(
                name=step["name"],
                arguments=arguments if isinstance(arguments, dict) else {},
                ok=bool(output.get("ok")),
                error_code=output.get("error_code") or step["error"],
            )
        )
    events = frozenset(
        str(a["details"]["security_event"])
        for a in trace["audit"]
        if a["details"] and a["details"].get("security_event")
    )
    return Evidence(outcome, answer, calls, events, changed, approvals, state or {})


def run_case(
    case: Case,
    client: ChatClient,
    *,
    model_name: str,
    budget: RunBudget | None = None,
    attempt: int = 1,
    screen: bool = True,
    router: ChatClient | None = None,
    claim_judge: ClaimJudge | None = None,
    tools: Sequence[Tool] | None = None,
    policy_prompt: str = DEFAULT_POLICY_PROMPT,
) -> CaseResult:
    engine = make_sqlite_engine()
    try:
        factory: SessionFactory = sessionmaker(engine, expire_on_commit=False)
        with factory() as session:
            seed_mock_store(session, now=EVAL_NOW)
            session.commit()
        with factory() as session:  # a fresh session: rows as the database stores them
            before = snapshot_store(session)
        case_tools: Sequence[Tool] | None = tools
        modes = None
        if case.write_mode is not None:  # write tools are offered only to cases that ask for them
            case_tools = (*(tools if tools is not None else default_tools()), *write_tools())
            mode = Mode(case.write_mode)
            modes = {RiskTier.LOW_RISK_WRITE: mode, RiskTier.MONEY: mode}
        if modes is None:
            gateway = ToolGateway(factory, case_tools, clock=lambda: EVAL_NOW)
        else:
            gateway = ToolGateway(factory, case_tools, clock=lambda: EVAL_NOW, modes=modes)
        runner = TicketRunner(
            factory,
            client,
            gateway,
            model_tier=model_name,
            budget=budget,
            screen=screen_text if screen else None,
            router=router,
            claim_judge=claim_judge,
            policy_prompt=policy_prompt,
        )
        try:
            ticket_id = runner.create_ticket(case.customer_id, case.message)
            started = time.monotonic()
            summary = runner.run(ticket_id)
            latency = time.monotonic() - started
        except Exception as exc:  # a crash is reported, not allowed to stop the whole suite
            detail = clean_str(f"{type(exc).__name__}: {exc}", 300)
            return CaseResult(case.id, case.tier, attempt, "error", "crash", detail)
        with factory() as session:
            changed = [t for t, d in snapshot_store(session).items() if d != before[t]]
            trace = load_trace(session, summary.run_id)
            approvals = session.scalar(select(func.count()).select_from(Approval)) or 0
            state = read_state(session, case)
        assert trace is not None
    finally:
        engine.dispose()

    evidence = _evidence(trace, summary.outcome, summary.answer, changed, approvals, state)
    checks = grade(case, evidence)
    stop = str(summary.stop_reason)
    status = "pass" if passed(checks) else "fail"
    if stop in ERROR_STOPS:
        status = "error"
    return CaseResult(
        case_id=case.id,
        tier=case.tier,
        attempt=attempt,
        status=status,
        stop_reason=stop,
        detail=clean_str(summary.detail, 300),
        checks=checks,
        answer=summary.answer,
        tool_calls=evidence.tool_calls,
        llm_calls=summary.llm_calls,
        tokens_in=summary.tokens_in,
        tokens_out=summary.tokens_out,
        cost_usd=summary.cost_usd,
        latency_s=latency,
        trace=trace,
    )


# ----------------------------------------------------------------------------- the suite


def run_suite(
    cases: Sequence[Case],
    client_for: Callable[[Case], ChatClient],
    *,
    model_name: str,
    repeat: int = 1,
    pause_s: float = 0.0,
    budget: RunBudget | None = None,
    screen: bool = True,
    router: ChatClient | None = None,
    claim_judge: ClaimJudge | None = None,
    tools: Sequence[Tool] | None = None,
    policy_prompt: str = DEFAULT_POLICY_PROMPT,
    on_result: Callable[[CaseResult], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[CaseResult]:
    """Run every case ``repeat`` times, one after another (``pause_s`` apart, for rate limits)."""
    results: list[CaseResult] = []
    total = len(cases) * repeat
    for attempt in range(1, repeat + 1):
        for case in cases:
            result = run_case(
                case,
                client_for(case),
                model_name=model_name,
                budget=budget,
                attempt=attempt,
                screen=screen,
                router=router,
                claim_judge=claim_judge,
                tools=tools,
                policy_prompt=policy_prompt,
            )
            results.append(result)
            if on_result is not None:
                on_result(result)
            if pause_s > 0 and len(results) < total:
                sleep(pause_s)
    return results


# --------------------------------------------------------------------------- the summary


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0.0


@dataclass(frozen=True)
class Summary:
    model: str
    prompt_version: str
    cases_digest: str
    cases: int
    attempts: int
    passed: int
    failed: int
    errored: int
    pass_rate: float | None
    tier0_pass_rate: float | None
    safety_failures: int
    valid_call_rate: float | None
    unstable_cases: list[str]
    avg_tokens: float
    total_cost_usd: Decimal
    p50_s: float
    p95_s: float

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "total_cost_usd": f"{self.total_cost_usd:.10f}"}


def cases_digest(cases: Iterable[Case]) -> str:
    """Short fingerprint of the case set, so two runs are only compared if it matches."""
    canonical = json.dumps([c.model_dump(mode="json") for c in cases], sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def summarize(
    results: Sequence[CaseResult],
    cases: Sequence[Case],
    model_name: str,
    *,
    policy: bool = False,
    policy_prompt: str = DEFAULT_POLICY_PROMPT,
) -> Summary:
    graded = [r for r in results if r.status != "error"]
    ok = [r for r in graded if r.status == "pass"]
    tier0 = [r for r in graded if r.tier == 0]
    calls = [t for r in results for t in r.tool_calls]
    valid = [t for t in calls if t.error_code not in INVALID_CALL_CODES]
    by_case: dict[str, set[str]] = defaultdict(set)
    for r in graded:
        by_case[r.case_id].add(r.status)
    times = [r.latency_s for r in graded]
    return Summary(
        model=model_name,
        prompt_version=prompt_version(policy, policy_prompt),
        cases_digest=cases_digest(cases),
        cases=len(cases),
        attempts=len(results),
        passed=len(ok),
        failed=len(graded) - len(ok),
        errored=len(results) - len(graded),
        pass_rate=_rate(len(ok), len(graded)),
        tier0_pass_rate=_rate(sum(r.status == "pass" for r in tier0), len(tier0)),
        safety_failures=sum(r.safety_failed for r in results),
        valid_call_rate=_rate(len(valid), len(calls)),
        unstable_cases=sorted(c for c, statuses in by_case.items() if len(statuses) > 1),
        avg_tokens=sum(r.tokens_in + r.tokens_out for r in results) / len(results)
        if results
        else 0,
        total_cost_usd=sum((r.cost_usd for r in results), Decimal(0)),
        p50_s=_percentile(times, 0.5),
        p95_s=_percentile(times, 0.95),
    )


# ----------------------------------------------------------------------------- reporting


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def format_result(result: CaseResult) -> str:
    tag = {"pass": "PASS", "fail": "FAIL", "error": "ERROR"}[result.status]
    attempt = f" #{result.attempt}" if result.attempt > 1 else ""
    head = (
        f"[{tag}] {result.case_id}{attempt}  {result.latency_s:.1f}s  "
        f"{result.tokens_in + result.tokens_out:,} tok  ${result.cost_usd:.6f}"
    )
    lines = [head]
    if result.status == "error":
        lines.append(f"        {result.stop_reason}: {result.detail}")
    for check in result.failed_checks:
        label = "SAFETY" if check.severity == "safety" else "quality"
        lines.append(f"        - {label}: {check.name}: {check.detail}")
    if result.stop_reason == "ungrounded":
        lines.append(f"        held: {result.detail}")
    if result.status == "fail" and result.answer:
        lines.append(f"        reply: {clean_str(result.answer, 240)!r}")
    return "\n".join(lines)


def format_summary(summary: Summary) -> str:
    lines = [
        f"Model {summary.model}, prompt {summary.prompt_version}, cases {summary.cases_digest}",
        f"  passed {summary.passed}/{summary.passed + summary.failed} graded runs"
        f" ({_pct(summary.pass_rate)}); tier 0: {_pct(summary.tier0_pass_rate)}",
        f"  safety failures: {summary.safety_failures}  (must be 0)",
        f"  valid tool-call rate: {_pct(summary.valid_call_rate)}",
        f"  avg {summary.avg_tokens:,.0f} tokens/run, shadow cost ${summary.total_cost_usd:.6f},"
        f" latency p50 {summary.p50_s:.1f}s p95 {summary.p95_s:.1f}s",
    ]
    if summary.errored:
        lines.append(
            f"  {summary.errored} run(s) ERRORED (provider or platform problem, not graded); "
            "re-run them with --only, or add --pause for rate limits"
        )
    if summary.unstable_cases:
        lines.append(f"  unstable across repeats: {', '.join(summary.unstable_cases)}")
    return "\n".join(lines)


_LOG_COLUMNS = (
    "timestamp_utc", "model", "prompt_version", "cases_digest", "attempts", "passed", "failed",
    "errored", "pass_rate", "tier0_pass_rate", "safety_failures", "valid_call_rate", "avg_tokens",
    "total_cost_usd", "p50_s", "p95_s", "note", "results_file",
)  # fmt: skip


def save_run(
    results: Sequence[CaseResult],
    summary: Summary,
    out_dir: Path,
    *,
    note: str = "",
    now: datetime | None = None,
) -> Path:
    """Write the full results as JSON and append one row to ``experiments.csv``."""
    now = now or datetime.now(UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", summary.model)
    path = out_dir / f"{now:%Y%m%d-%H%M%S}_{safe_model}.json"
    document = {
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "note": note,
        "summary": summary.to_dict(),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")

    log = out_dir / "experiments.csv"
    is_new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(_LOG_COLUMNS)
        s = summary
        writer.writerow(
            [
                now.isoformat(timespec="seconds"), s.model, s.prompt_version, s.cases_digest,
                s.attempts, s.passed, s.failed, s.errored,
                "" if s.pass_rate is None else f"{s.pass_rate:.3f}",
                "" if s.tier0_pass_rate is None else f"{s.tier0_pass_rate:.3f}",
                s.safety_failures,
                "" if s.valid_call_rate is None else f"{s.valid_call_rate:.3f}",
                f"{s.avg_tokens:.0f}", f"{s.total_cost_usd:.6f}",
                f"{s.p50_s:.2f}", f"{s.p95_s:.2f}", note, path.name,
            ]
        )  # fmt: skip
    return path
