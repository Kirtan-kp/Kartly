"""Replay a stored run against today's code (ADR-053).

    python -m kartly.replay <run_id>

The run's model replies were recorded in its trace. Replay builds a throwaway database with the
standard synthetic seed, puts the same customer messages in it, and runs the current code with a
fake model that hands back the recorded replies, in order (a recorded provider error is replayed
as an error). The report says whether today's code still does what the stored run did: the same
tool calls with the same arguments, the same outcome, the same final answer.

It answers "did my change alter behaviour on real traffic?" without calling a model or touching
the real database (read-only). What it does not do:

- It does not test the model. The replies are frozen; a different model or prompt needs a new run.
- The router step is skipped, so a run that was routed by the classifier replays from the agent.
- The throwaway database is the seed, not the production data at the time of the run. A run that
  read an order that was later changed, or whose writes already happened, can differ for that
  reason alone; the report lists which tool results differ so that is visible.
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.llm.errors import ProviderUnavailableError
from kartly.llm.types import ChatResponse, ToolCall, Usage
from kartly.orchestrator import TicketRunner
from kartly.orchestrator.trace import load_trace
from kartly.sanitize import clean_str
from kartly.store.models import Message, MessageRole, Run, StepKind, Ticket
from kartly.store.seed import seed_mock_store
from kartly.store.sqlite import make_sqlite_engine


class ReplayExhaustedError(RuntimeError):
    """Today's code asked the model for more replies than the stored run had."""


class RecordedClient:
    """Plays back the model replies recorded in a trace, in order."""

    name = "replay"

    def __init__(self, steps: Sequence[Mapping[str, Any]]) -> None:
        self._steps = [s for s in steps if s["kind"] == StepKind.LLM_CALL.value]
        self._next = 0
        self.asked = 0

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
        self.asked += 1
        if self._next >= len(self._steps):
            raise ReplayExhaustedError("the code asked for more model replies than were recorded")
        step = self._steps[self._next]
        self._next += 1
        model = str(step.get("model") or "replay")
        if step.get("error") or step.get("output") is None:
            raise ProviderUnavailableError(
                str(step.get("error") or "recorded failure"), model=model
            )
        out = step["output"]
        calls = tuple(_tool_call(c) for c in out.get("tool_calls") or [])
        return ChatResponse(
            model=model,
            provider_model=str(out.get("provider_model") or model),
            content=out.get("content"),
            tool_calls=calls,
            finish_reason=out.get("finish_reason"),
            usage=Usage(int(step.get("tokens_in") or 0), int(step.get("tokens_out") or 0)),
            cost_usd=Decimal(str(step.get("shadow_cost_usd") or 0)),
            latency_ms=int(step.get("latency_ms") or 0),
        )


def _tool_call(recorded: Mapping[str, Any]) -> ToolCall:
    raw = str(recorded.get("arguments") or "")
    try:
        parsed = json.loads(raw)
        arguments = parsed if isinstance(parsed, dict) else None
        error = None if arguments is not None else "arguments are not a JSON object"
    except ValueError as exc:
        arguments, error = None, f"invalid JSON: {exc}"
    return ToolCall(
        str(recorded.get("id") or "call"), str(recorded.get("name")), arguments, raw, error
    )


@dataclass
class Signature:
    """What a run did, in the form two runs are compared by."""

    tool_calls: list[tuple[str, str]]  # (name, arguments as sorted JSON)
    tool_errors: list[str | None]
    outcome: str | None
    answer: str | None
    model_calls: int


@dataclass
class ReplayReport:
    run_id: str
    original: Signature
    replayed: Signature
    differences: list[str] = field(default_factory=list)

    @property
    def same(self) -> bool:
        return not self.differences


def signature(trace: Mapping[str, Any], answer: str | None) -> Signature:
    steps = trace["steps"]
    tools = [s for s in steps if s["kind"] == StepKind.TOOL_CALL.value]
    return Signature(
        tool_calls=[
            (str(s["name"]), json.dumps((s["input"] or {}).get("arguments"), sort_keys=True))
            for s in tools
        ],
        tool_errors=[s.get("error") for s in tools],
        outcome=trace["run"]["outcome"],
        answer=answer,
        model_calls=sum(1 for s in steps if s["kind"] == StepKind.LLM_CALL.value),
    )


def compare(original: Signature, replayed: Signature) -> list[str]:
    found = []
    if original.tool_calls != replayed.tool_calls:
        found.append(f"tool calls differ: was {original.tool_calls}, now {replayed.tool_calls}")
    elif original.tool_errors != replayed.tool_errors:
        found.append(
            f"same tool calls but different results: was errors {original.tool_errors}, "
            f"now {replayed.tool_errors}"
        )
    if original.outcome != replayed.outcome:
        found.append(f"outcome was {original.outcome!r}, now {replayed.outcome!r}")
    if original.answer != replayed.answer:
        found.append(f"final answer was {original.answer!r}, now {replayed.answer!r}")
    if original.model_calls != replayed.model_calls:
        found.append(f"model calls were {original.model_calls}, now {replayed.model_calls}")
    return found


def original_answer(session: Session, trace: Mapping[str, Any]) -> str | None:
    """The reply the stored run produced: the first agent message written from its start on."""
    run = trace["run"]
    started = _naive(datetime.fromisoformat(run["started_at"]))
    ended = _naive(datetime.fromisoformat(run["finished_at"])) if run["finished_at"] else None
    query = (
        select(Message)
        .where(
            Message.ticket_id == uuid.UUID(run["ticket_id"]),
            Message.role == MessageRole.AGENT,
        )
        .order_by(Message.created_at, Message.id)
    )
    for message in session.scalars(query):
        at = _naive(message.created_at)
        if at >= started and (ended is None or at <= ended + timedelta(seconds=5)):
            return message.content
    return None


def _naive(moment: datetime) -> datetime:
    return moment.replace(tzinfo=None) if moment.tzinfo else moment


def thread_before(session: Session, trace: Mapping[str, Any]) -> list[tuple[MessageRole, str]]:
    """The messages the run worked from.

    The run's start time is stored to the second, so a customer message written a moment before
    the run can look later than it. Customer and human messages therefore count until the run
    ended; agent messages only if they were written before it began (earlier answers, never this
    run's own).
    """
    run = trace["run"]
    started = _naive(datetime.fromisoformat(run["started_at"]))
    ended = _naive(datetime.fromisoformat(run["finished_at"])) if run["finished_at"] else None
    out = []
    for m in session.scalars(
        select(Message)
        .where(Message.ticket_id == uuid.UUID(run["ticket_id"]))
        .order_by(Message.created_at, Message.id)
    ):
        at = _naive(m.created_at)
        if m.role is MessageRole.AGENT:
            keep = at < started
        else:
            keep = ended is None or at <= ended + timedelta(seconds=1)
        if keep:
            out.append((m.role, m.content))
    return out


def replay_run(
    source: sessionmaker[Session],
    run_id: uuid.UUID,
    build_gateway: Any,
    *,
    model_tier: str = "replay",
    context_budget: int | None = None,
) -> ReplayReport:
    """Replay ``run_id`` (read from ``source``) against the current code.

    ``build_gateway(factory)`` must return the ToolGateway the live app would use.
    """
    with source() as session:
        trace = load_trace(session, run_id)
        if trace is None:
            raise LookupError(f"no run {run_id}")
        thread = thread_before(session, trace)
        before = signature(trace, original_answer(session, trace))
        customer_id = trace["ticket"]["customer_id"] if trace["ticket"] else None
    if customer_id is None or not any(role is MessageRole.CUSTOMER for role, _ in thread):
        raise ValueError("the run has no customer message to replay")

    engine = make_sqlite_engine()
    factory = sessionmaker(engine, expire_on_commit=False)
    try:
        with factory() as session:
            seed_mock_store(session)
            ticket = Ticket(customer_id=customer_id)
            session.add(ticket)
            session.flush()
            base = None
            for i, (role, text) in enumerate(thread):
                base = base or datetime.now(UTC) - timedelta(minutes=5)
                session.add(
                    Message(
                        ticket_id=ticket.id,
                        role=role,
                        content=clean_str(text, 20000),
                        created_at=base + timedelta(seconds=i),
                    )
                )
            session.commit()
            ticket_id = ticket.id
        client = RecordedClient(trace["steps"])
        gateway: ToolGateway = build_gateway(factory)
        kwargs: dict[str, Any] = {"model_tier": model_tier}
        if context_budget is not None:
            kwargs["context_budget"] = context_budget
        runner = TicketRunner(factory, client, gateway, **kwargs)
        try:
            summary = runner.run(ticket_id)
            answer = summary.answer
        except ReplayExhaustedError:
            answer = None
        with factory() as session:
            new_run = session.scalar(select(Run).where(Run.ticket_id == ticket_id))
            new_trace = load_trace(session, new_run.id) if new_run else None
        if new_trace is None:
            raise RuntimeError("the replay produced no run")
        after = signature(new_trace, answer)
    finally:
        engine.dispose()
    return ReplayReport(str(run_id), before, after, compare(before, after))


def render(report: ReplayReport) -> str:
    if report.same:
        o = report.original
        return (
            f"run {report.run_id}: replay matches the stored run "
            f"({len(o.tool_calls)} tool call(s), outcome {o.outcome}, "
            f"{o.model_calls} model call(s))"
        )
    lines = [f"run {report.run_id}: replay DIFFERS from the stored run"]
    lines += [f"  - {d}" for d in report.differences]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.config import get_settings, load_env
    from kartly.gateway import modes_from_settings
    from kartly.retrieval.setup import gateway_tools
    from kartly.store.db import make_engine, make_session_factory

    parser = argparse.ArgumentParser(description="Replay a stored run against today's code.")
    parser.add_argument("run_id", type=uuid.UUID)
    args = parser.parse_args(argv)
    load_env()
    settings = get_settings()
    engine = make_engine()
    try:
        report = replay_run(
            make_session_factory(engine),
            args.run_id,
            lambda factory: ToolGateway(
                factory,
                gateway_tools(settings),
                modes=modes_from_settings(settings.tier1_mode, settings.tier2_mode),
            ),
            context_budget=settings.context_budget_tokens,
        )
    except (LookupError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    finally:
        engine.dispose()
    print(render(report))
    return 0 if report.same else 1


if __name__ == "__main__":
    raise SystemExit(main())
