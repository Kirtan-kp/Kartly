"""Fault injection: what does a customer see when the model provider misbehaves? (ADR-053)

    python -m kartly.chaos            # a table of scenarios and what each one ended in
    python -m kartly.chaos --json

Every scenario runs the real stack (queue, worker, runner, tools, gateway, retry/fallback/breaker,
an in-memory database) against a fake model that fails on a schedule. Time is virtual: waits are
added to a counter, so a three-minute outage takes milliseconds. Nothing here calls a real model.

The rules every scenario must satisfy (``violations``): the ticket ends drafted or escalated
(never stuck, never failed), a customer never receives an answer the run did not produce, and a
refund is paid at most once however many times the run is retried.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import Mode, ToolGateway
from kartly.llm.errors import (
    AuthenticationError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ProviderUnavailableError,
    RateLimitError,
)
from kartly.llm.resilient import ResilientClient, RetryPolicy
from kartly.llm.types import ChatResponse, ToolCall, Usage
from kartly.orchestrator import TicketRunner
from kartly.queue import JobQueue
from kartly.store.models import Action, ActionStatus, Run, Ticket, TicketStatus
from kartly.store.seed import seed_mock_store
from kartly.store.sqlite import make_sqlite_engine
from kartly.tools import RiskTier, default_tools
from kartly.tools.writes import make_refund_tool
from kartly.worker import Worker

CUSTOMER = "C-0004"
ORDER = "KRT-100007"  # delivered, INR 599, belongs to C-0004
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

Faults = Callable[[int], str | None]
"""Given the 0-based number of this call to a model, the fault to inject (or None to behave)."""


class VirtualTime:
    def __init__(self) -> None:
        self.t = 0.0

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    def monotonic(self) -> float:
        return self.t

    def now(self) -> datetime:
        return T0 + timedelta(seconds=self.t)


# ------------------------------------------------------------------------------ the fake model


def _response(name: str, content: str | None, calls: Sequence[ToolCall] = ()) -> ChatResponse:
    return ChatResponse(
        model=name,
        provider_model=name,
        content=content,
        tool_calls=tuple(calls),
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(100, 20),
        cost_usd=Decimal("0"),
        latency_ms=5,
    )


def _tool(name: str, args: dict[str, Any]) -> ToolCall:
    return ToolCall("call_1", name, args, json.dumps(args))


class HealthyModel:
    """A model that does the right thing: look the order up (or refund it), then say so."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def reply(self, name: str, messages: Sequence[Mapping[str, Any]]) -> ChatResponse:
        if any(m.get("role") == "tool" for m in messages):
            text = (
                f"Your order {ORDER} was delivered."
                if self.kind == "read"
                else "Your refund of INR 599 has been issued."
            )
            return _response(name, text)
        if self.kind == "read":
            return _response(name, None, [_tool("get_order", {"order_id": ORDER})])
        args = {"order_id": ORDER, "amount_inr": 599, "reason_code": "damaged"}
        return _response(name, None, [_tool("issue_refund", args)])


class FaultyClient:
    """Wraps a healthy model and breaks it as ``faults`` says. ``calls`` counts every request."""

    def __init__(self, name: str, healthy: HealthyModel, faults: Faults, time: VirtualTime) -> None:
        self.name = name
        self.calls = 0
        self._healthy = healthy
        self._faults = faults
        self._time = time

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
        n = self.calls
        self.calls += 1
        fault = self._faults(n)
        if fault is not None:
            kind, _, arg = fault.partition(":")
            if kind == "429":
                raise RateLimitError(
                    "rate limited", model=self.name, status=429, retry_after=float(arg or 0) or None
                )
            if kind == "5xx":
                raise ProviderUnavailableError("upstream error", model=self.name, status=503)
            if kind == "timeout":
                self._time.sleep(float(arg or 30))  # the caller waited this long for nothing
                raise LLMTimeoutError("request timed out", model=self.name)
            if kind == "malformed":
                raise MalformedResponseError("not a chat completion", model=self.name)
            if kind == "auth":
                raise AuthenticationError("invalid api key", model=self.name, status=401)
            if kind == "slow":  # answers, but late
                self._time.sleep(float(arg or 10))
            else:
                raise LLMError(f"unknown fault {fault!r}")
        return self._healthy.reply(self.name, messages)


# ------------------------------------------------------------------------------ scenarios


def _always(fault: str) -> Faults:
    return lambda n: fault


def _first(count: int, fault: str) -> Faults:
    return lambda n: fault if n < count else None


def _never(n: int) -> str | None:
    return None


@dataclass(frozen=True)
class Scenario:
    name: str
    story: str
    primary: Faults = _never
    fallback: Faults = _never
    kind: str = "read"  # "read" looks up the order; "refund" issues INR 599 and then replies
    expect_ticket: tuple[str, ...] = ("drafted",)  # the allowed ends
    expect_job: str = "succeeded"
    max_refunds: int = 1


def _fail_after_first(fault: str) -> Faults:
    return lambda n: None if n == 0 else fault


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("healthy", "No faults: the baseline."),
    Scenario(
        "blip",
        "The provider returns one 503, then recovers.",
        primary=_first(1, "5xx"),
    ),
    Scenario(
        "rate_limit_short",
        "A 429 asking us to wait 2 seconds; we wait and retry the same model.",
        primary=_first(1, "429:2"),
    ),
    Scenario(
        "rate_limit_long",
        "A 429 asking for 120 seconds; waiting is worse than switching, so we use the fallback.",
        primary=_always("429:120"),
    ),
    Scenario(
        "primary_down",
        "The primary model is down all day; the fallback answers every call.",
        primary=_always("5xx"),
    ),
    Scenario(
        "timeouts_then_ok",
        "Two 30-second timeouts use up the call's 60-second deadline; the queue retries it.",
        primary=_first(2, "timeout:30"),
    ),
    Scenario(
        "slow_but_alive",
        "Every answer takes 20 seconds; the customer waits but gets a normal answer.",
        primary=_always("slow:20"),
    ),
    Scenario(
        "outage_then_recovery",
        "Both providers fail the first attempt; the queue retries after 30 s, the "
        "circuit breaker lets a probe through, and the second attempt succeeds.",
        primary=_first(3, "5xx"),
        fallback=_first(3, "5xx"),
    ),
    Scenario(
        "total_outage",
        "Both providers are down for good: after three attempts the ticket goes to a human.",
        primary=_always("5xx"),
        fallback=_always("5xx"),
        expect_ticket=("escalated",),
        expect_job="dead",
    ),
    Scenario(
        "bad_api_key",
        "The key was revoked: not retried against the same model, no fallback hides it; "
        "the ticket goes to a human and the job is parked.",
        primary=_always("auth"),
        fallback=_always("auth"),
        expect_ticket=("escalated",),
        expect_job="dead",
    ),
    Scenario(
        "crash_after_refund",
        "The refund went through, then the provider died before the reply. A retry would "
        "refund twice, so the job is parked for a human instead.",
        kind="refund",
        primary=_fail_after_first("5xx"),
        fallback=_always("5xx"),
        expect_ticket=("escalated",),
        expect_job="dead",
    ),
)


# ------------------------------------------------------------------------------ running them


@dataclass
class Outcome:
    scenario: str
    story: str
    ticket: str = ""
    job: str = ""
    attempts: int = 0
    runs: int = 0
    model_calls: int = 0
    refunds: int = 0
    seconds: float = 0.0
    customer_sees: str = ""
    error: str | None = None
    violations: list[str] = field(default_factory=list)


def _build(
    scenario: Scenario, time: VirtualTime
) -> tuple[sessionmaker[Session], JobQueue, Worker, TicketRunner, list[FaultyClient]]:
    engine = make_sqlite_engine()
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        seed_mock_store(session, now=T0)
        session.commit()
    healthy = HealthyModel(scenario.kind)
    primary = FaultyClient("primary", healthy, scenario.primary, time)
    backup = FaultyClient("backup", healthy, scenario.fallback, time)
    client = ResilientClient(
        [primary, backup],
        RetryPolicy(jitter=0.0),
        sleep=time.sleep,
        clock=time.monotonic,
        rng=random.Random(0),
    )
    modes = {RiskTier.LOW_RISK_WRITE: Mode.AUTO, RiskTier.MONEY: Mode.AUTO}
    gateway = ToolGateway(
        factory, (*default_tools(), make_refund_tool()), modes=modes, clock=time.now
    )
    runner = TicketRunner(factory, client, gateway, model_tier="chaos")
    queue = JobQueue(factory, max_attempts=3, lease_s=300, clock=time.now)
    return (
        factory,
        queue,
        Worker(queue, runner.run, factory, worker_id="chaos"),
        runner,
        [
            primary,
            backup,
        ],
    )


def run_scenario(scenario: Scenario, *, horizon_s: float = 3600.0) -> Outcome:
    time = VirtualTime()
    factory, queue, worker, runner, clients = _build(scenario, time)
    out = Outcome(scenario.name, scenario.story)
    text = (
        f"Where is my order {ORDER}?"
        if scenario.kind == "read"
        else f"Order {ORDER} arrived damaged, please refund INR 599."
    )
    ticket_id = runner.create_ticket(CUSTOMER, text)
    job_id = queue.enqueue(ticket_id)

    while queue.depth() > 0 and time.t < horizon_s:
        if not worker.run_once():
            time.sleep(1.0)  # nothing due yet: the backoff is still running
    out.seconds = round(time.t, 1)

    view = queue.get(job_id)
    with factory() as session:
        ticket = session.get(Ticket, ticket_id)
        assert ticket is not None
        out.ticket = ticket.status.value
        out.job = view.status.value if view else "missing"
        out.attempts = view.attempts if view else 0
        out.error = view.last_error if view else None
        out.runs = session.scalar(select(func.count()).select_from(Run)) or 0
        out.refunds = (
            session.scalar(
                select(func.count())
                .select_from(Action)
                .where(Action.tool_name == "issue_refund", Action.status == ActionStatus.SUCCEEDED)
            )
            or 0
        )
        out.customer_sees = _customer_view(session, ticket, view.result if view else None)
    out.model_calls = sum(c.calls for c in clients)
    out.violations = violations(scenario, out)
    return out


def _customer_view(session: Session, ticket: Ticket, result: dict[str, Any] | None) -> str:
    if ticket.status == TicketStatus.DRAFTED and result and result.get("answer"):
        return f"reply drafted for review: {result['answer']!r}"
    if ticket.status == TicketStatus.ESCALATED:
        return "no automated reply; a human has the ticket"
    return f"nothing sent (ticket {ticket.status.value})"


def violations(scenario: Scenario, out: Outcome) -> list[str]:
    found = []
    if out.ticket not in scenario.expect_ticket:
        found.append(f"ticket ended {out.ticket}, expected one of {scenario.expect_ticket}")
    if out.job != scenario.expect_job:
        found.append(f"job ended {out.job}, expected {scenario.expect_job}")
    if out.ticket in ("received", "screened", "routed", "resolving", "validating", "failed"):
        found.append("the ticket was left in a state no one is working on")
    if out.refunds > scenario.max_refunds:
        found.append(f"{out.refunds} refunds paid; at most {scenario.max_refunds} allowed")
    if out.ticket == "drafted" and not out.customer_sees.startswith("reply drafted"):
        found.append("a ticket was marked drafted without an answer")
    return found


def run_all(only: Sequence[str] | None = None) -> list[Outcome]:
    chosen = [s for s in SCENARIOS if not only or s.name in only]
    return [run_scenario(s) for s in chosen]


def render(outcomes: Sequence[Outcome]) -> str:
    lines = []
    for o in outcomes:
        mark = "FAIL" if o.violations else "ok  "
        lines.append(
            f"{mark} {o.scenario:<22} ticket={o.ticket:<10} job={o.job:<9} attempts={o.attempts} "
            f"model_calls={o.model_calls:<2} refunds={o.refunds} waited={o.seconds:>6.0f}s"
        )
        lines.append(f"       {o.story}")
        lines.append(f"       customer: {o.customer_sees}")
        if o.error:
            lines.append(f"       parked because: {o.error}")
        for v in o.violations:
            lines.append(f"       VIOLATION: {v}")
    bad = sum(1 for o in outcomes if o.violations)
    lines.append(f"\n{len(outcomes) - bad}/{len(outcomes)} scenarios behaved as designed")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Break the model provider and report the damage.")
    parser.add_argument("--only", nargs="*", help="scenario names")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    unknown = set(args.only or []) - {s.name for s in SCENARIOS}
    if unknown:
        print(f"unknown scenario(s): {', '.join(sorted(unknown))}")
        return 2
    outcomes = run_all(args.only)
    print(json.dumps([asdict(o) for o in outcomes], indent=2) if args.json else render(outcomes))
    return 1 if any(o.violations for o in outcomes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
