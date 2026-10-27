"""The loop and run recording on a real Postgres: JSONB, varchar limits, per-step commits."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.llm.types import ChatResponse
from kartly.orchestrator import StopReason, TicketRunner, load_trace
from kartly.store.models import Message, MessageRole, Run, RunStatus, RunStep
from tests.llm_support import ScriptedClient, make_response, tool_call

pytestmark = pytest.mark.postgres


def test_a_run_is_stored_on_postgres(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    client = ScriptedClient(
        [
            make_response(
                None,
                [tool_call("get_tracking", {"order_id": "KRT-100006"})],
                prompt_tokens=400,
                completion_tokens=25,
                cost="0.0000012345",
            ),
            make_response("It is delayed.", cost="0.0000007"),
        ]
    )
    runner = TicketRunner(pg_factory, client, pg_gateway, model_tier="scripted")
    ticket_id = runner.create_ticket("C-0003", "Where is KRT-100006?")
    summary = runner.run(ticket_id)
    assert summary.outcome == "drafted"

    with pg_factory() as session:
        run = session.get(Run, summary.run_id)
        assert run is not None and run.status is RunStatus.SUCCEEDED
        # ten decimal places: a few-millionths-of-a-dollar call is not rounded away
        assert Decimal(run.shadow_cost_usd) == Decimal("0.0000019345")
        steps = list(session.scalars(select(RunStep).order_by(RunStep.idx)))
        assert Decimal(steps[0].shadow_cost_usd) == Decimal("0.0000012345")
        assert steps[1].output is not None and steps[1].output["data"]["shipments"][0]["delayed"]
        trace = load_trace(session, summary.run_id)
        flagged = session.execute(
            text("SELECT count(*) FROM audit_log WHERE run_id = :r"), {"r": summary.run_id}
        ).scalar_one()
    assert trace is not None and trace["run"]["shadow_cost_usd"] == "0.0000019345"
    assert flagged == 1


def test_hostile_model_output_does_not_break_the_trace_on_postgres(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    """NUL bytes, lone surrogates, NaN, huge numbers and overlong names: all rejected by JSONB or
    varchar, all of which a model can produce."""
    hostile = make_response(
        "reply with \x00 nul and \ud800 surrogate",
        [
            tool_call(
                "tool_" + "n" * 500,
                {
                    "a": "nul\x00",
                    "b": "\ud800",
                    "c": float("nan"),
                    "d": 10**5000,
                    "e": "x" * 100_000,
                },
                call_id="id\x00" + "i" * 500,
                raw="{}",  # (a real provider's JSON could not carry a 5,000-digit integer)
            ),
            tool_call(
                "get_order",
                {
                    "order_id": "KRT-100003",
                    "note": "\x00\ud800",
                    "deep": {"a": {"b": {"c": {"d": {}}}}},
                },
                call_id="ok",
            ),
        ],
    )
    client = ScriptedClient([hostile, make_response("fine \x00 \ud800 done")])
    runner = TicketRunner(pg_factory, client, pg_gateway)
    ticket_id = runner.create_ticket("C-0001", "nul\x00 \ud800 ticket text")
    summary = runner.run(ticket_id)  # must not raise

    assert summary.outcome == "drafted"
    with pg_factory() as session:
        steps = list(session.scalars(select(RunStep).order_by(RunStep.idx)))
        assert [s.idx for s in steps] == list(range(len(steps))) and len(steps) == 4
        assert all(len(s.name) <= 80 for s in steps)
        agent = session.scalars(select(Message).where(Message.role == MessageRole.AGENT)).one()
        assert "\x00" not in agent.content


def test_each_step_is_committed_as_it_happens(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    """A crash after the first model call still leaves that step in the database."""

    class CrashesSecondCall(ScriptedClient):
        def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
            if self.calls:
                raise RuntimeError("worker killed")
            return super().chat(messages, **kwargs)

    client = CrashesSecondCall(
        [make_response(None, [tool_call("get_order", {"order_id": "KRT-100001"})])]
    )
    runner = TicketRunner(pg_factory, client, pg_gateway)
    with pytest.raises(RuntimeError):
        runner.run(runner.create_ticket("C-0001", "hi"))
    with pg_factory() as session:
        run = session.scalars(select(Run)).one()
        assert run.status is RunStatus.FAILED and run.error is not None
        assert [s.kind.value for s in session.scalars(select(RunStep).order_by(RunStep.idx))] == [
            "llm_call",
            "tool_call",
        ]


def test_a_stopped_run_is_escalated_on_postgres(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    client = ScriptedClient([make_response("")] * 3)
    runner = TicketRunner(pg_factory, client, pg_gateway)
    summary = runner.run(runner.create_ticket("C-0001", "hi"))
    assert summary.stop_reason is StopReason.MALFORMED_OUTPUT and summary.outcome == "escalated"
