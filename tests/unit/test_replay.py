"""Replaying a stored run against today's code."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.config import Settings
from kartly.gateway import ToolGateway
from kartly.llm.errors import ProviderUnavailableError
from kartly.orchestrator import TicketRunner
from kartly.replay import (
    RecordedClient,
    ReplayExhaustedError,
    Signature,
    compare,
    main,
    render,
    replay_run,
)
from kartly.store.models import Message, MessageRole, Run, Ticket
from kartly.tools import default_tools
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW


def gateway(factory: sessionmaker[Session]) -> ToolGateway:
    return ToolGateway(factory, default_tools())


def stored_run(factory: sessionmaker[Session], script: list[Any]) -> uuid.UUID:
    runner = TicketRunner(factory, ScriptedClient(script), gateway(factory), model_tier="scripted")
    ticket = runner.create_ticket("C-0002", "Where is my order KRT-100002?")
    runner.run(ticket)
    with factory() as session:
        run = session.scalar(select(Run).where(Run.ticket_id == ticket))
        assert run is not None
        return run.id


def happy() -> list[Any]:
    return [
        make_response(None, [tool_call("get_tracking", {"order_id": "KRT-100002"})]),
        make_response("It is in transit."),
    ]


def test_a_run_replays_to_the_same_behaviour(seeded_factory: sessionmaker[Session]) -> None:
    run_id = stored_run(seeded_factory, happy())
    report = replay_run(seeded_factory, run_id, gateway)
    assert report.same, report.differences
    assert report.original.tool_calls == [("get_tracking", '{"order_id": "KRT-100002"}')]
    assert report.replayed.answer == report.original.answer == "It is in transit."
    assert "matches the stored run" in render(report)


def test_replay_never_touches_the_source_database(seeded_factory: sessionmaker[Session]) -> None:
    run_id = stored_run(seeded_factory, happy())
    with seeded_factory() as session:
        before = (
            len(session.scalars(select(Run)).all()),
            len(session.scalars(select(Ticket)).all()),
        )
    replay_run(seeded_factory, run_id, gateway)
    with seeded_factory() as session:
        after = (
            len(session.scalars(select(Run)).all()),
            len(session.scalars(select(Ticket)).all()),
        )
    assert before == after == (1, 1)


def test_a_change_in_the_tools_shows_up_as_a_difference(
    seeded_factory: sessionmaker[Session],
) -> None:
    run_id = stored_run(seeded_factory, happy())
    report = replay_run(
        seeded_factory,
        run_id,
        lambda f: ToolGateway(f, tuple(t for t in default_tools() if t.name != "get_tracking")),
    )
    assert not report.same
    assert any("different results" in d or "tool calls differ" in d for d in report.differences)
    assert "DIFFERS" in render(report)


def test_a_recorded_provider_failure_is_replayed_as_a_failure(
    seeded_factory: sessionmaker[Session],
) -> None:
    run_id = stored_run(seeded_factory, [ProviderUnavailableError("down", model="m")])
    report = replay_run(seeded_factory, run_id, gateway)
    assert report.same, report.differences
    assert report.original.answer is None and report.original.model_calls == 1


def test_only_the_messages_before_the_run_are_replayed(
    seeded_factory: sessionmaker[Session],
) -> None:
    run_id = stored_run(seeded_factory, happy())
    with seeded_factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        session.add(
            Message(
                ticket_id=run.ticket_id,
                role=MessageRole.CUSTOMER,
                content="Also, what about my other order?",
                created_at=FIXED_NOW.replace(year=2030),
            )
        )
        session.commit()
    assert replay_run(seeded_factory, run_id, gateway).same


def test_an_unknown_run_is_an_error(seeded_factory: sessionmaker[Session]) -> None:
    with pytest.raises(LookupError):
        replay_run(seeded_factory, uuid.uuid4(), gateway)


def test_the_recorded_client_hands_replies_back_in_order_and_then_runs_dry() -> None:
    steps: list[dict[str, Any]] = [
        {"kind": "stage", "name": "x"},
        {
            "kind": "llm_call",
            "model": "m1",
            "tokens_in": 7,
            "tokens_out": 3,
            "output": {
                "content": None,
                "tool_calls": [{"id": "c1", "name": "get_order", "arguments": '{"order_id": "A"}'}],
                "finish_reason": "tool_calls",
            },
        },
        {"kind": "llm_call", "model": "m1", "output": None, "error": "Timeout: slow"},
        {
            "kind": "llm_call",
            "model": "m1",
            "output": {
                "content": "x",
                "tool_calls": [{"id": "c2", "name": "t", "arguments": "{not json"}],
            },
        },
    ]
    client = RecordedClient(steps)
    first = client.chat([])
    assert first.tool_calls[0].arguments == {"order_id": "A"} and first.usage.prompt_tokens == 7
    with pytest.raises(ProviderUnavailableError, match="Timeout: slow"):
        client.chat([])
    third = client.chat([])
    assert third.tool_calls[0].arguments is None and third.tool_calls[0].parse_error
    with pytest.raises(ReplayExhaustedError):
        client.chat([])
    assert client.asked == 4


def sig(**kw: Any) -> Signature:
    base: dict[str, Any] = {
        "tool_calls": [("get_order", "{}")],
        "tool_errors": [None],
        "outcome": "drafted",
        "answer": "hi",
        "model_calls": 2,
    }
    return Signature(**{**base, **kw})


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"tool_calls": [("get_order", '{"a": 1}')]}, "tool calls differ"),
        ({"tool_errors": ["not_found"]}, "different results"),
        ({"outcome": "escalated"}, "outcome was"),
        ({"answer": "bye"}, "final answer"),
        ({"model_calls": 3}, "model calls"),
    ],
)
def test_every_kind_of_difference_is_reported(change: dict[str, Any], fragment: str) -> None:
    [only] = compare(sig(), sig(**change))
    assert fragment in only
    assert compare(sig(), sig()) == []


def test_the_command_line_reports_a_missing_run(
    seeded_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("kartly.store.db.make_engine", lambda *a, **k: seeded_factory.kw["bind"])
    monkeypatch.setattr("kartly.config.load_env", lambda: None)
    monkeypatch.setattr("kartly.config.get_settings", lambda: Settings(_env_file=None))  # type: ignore[call-arg]
    monkeypatch.setattr("sqlalchemy.engine.Engine.dispose", lambda self, *a, **k: None)
    assert main([str(uuid.uuid4())]) == 2
    assert "no run" in capsys.readouterr().out


def test_the_command_line_replays_a_stored_run(
    seeded_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = stored_run(seeded_factory, happy())
    monkeypatch.setattr("kartly.store.db.make_engine", lambda *a, **k: seeded_factory.kw["bind"])
    monkeypatch.setattr("kartly.config.load_env", lambda: None)
    monkeypatch.setattr("kartly.config.get_settings", lambda: Settings(_env_file=None))  # type: ignore[call-arg]
    monkeypatch.setattr("sqlalchemy.engine.Engine.dispose", lambda self, *a, **k: None)
    assert main([str(run_id)]) == 0
    assert "matches the stored run" in capsys.readouterr().out


def test_both_calls_and_results_differing_is_one_difference_not_two() -> None:
    changed = sig(tool_calls=[("other", "{}")], tool_errors=["boom"])
    [only] = compare(sig(), changed)
    assert "tool calls differ" in only


def test_the_thread_is_what_the_run_saw(seeded_factory: sessionmaker[Session]) -> None:
    from datetime import timedelta

    from kartly.replay import thread_before

    base = FIXED_NOW
    with seeded_factory() as session:
        ticket = Ticket(customer_id="C-0002")
        session.add(ticket)
        session.flush()
        for offset, role, text in [
            (0, MessageRole.CUSTOMER, "first question"),
            (5, MessageRole.AGENT, "earlier answer"),
            (60, MessageRole.CUSTOMER, "follow-up"),
            (62, MessageRole.AGENT, "this run's own answer"),
            (65, MessageRole.CUSTOMER, "written within a second of the end"),
            (66, MessageRole.CUSTOMER, "two seconds after the end"),
            (400, MessageRole.CUSTOMER, "much later"),
        ]:
            session.add(
                Message(
                    ticket_id=ticket.id,
                    role=role,
                    content=text,
                    created_at=base + timedelta(seconds=offset),
                )
            )
        session.commit()
        run = {
            "ticket_id": str(ticket.id),
            "started_at": (base + timedelta(seconds=61)).isoformat(),
            "finished_at": (base + timedelta(seconds=64)).isoformat(),
        }
        thread = thread_before(session, {"run": run})
    # excluded: the run's own answer (62) and anything later than a second after its end (400)
    assert [t for _r, t in thread] == [
        "first question",
        "earlier answer",
        "follow-up",
        "written within a second of the end",
    ]


def test_the_original_answer_is_this_runs_not_an_earlier_one(
    seeded_factory: sessionmaker[Session],
) -> None:
    from datetime import timedelta

    from kartly.replay import original_answer

    with seeded_factory() as session:
        ticket = Ticket(customer_id="C-0002")
        session.add(ticket)
        session.flush()
        for offset, text in [(5, "answer from an earlier run"), (62, "this run's answer")]:
            session.add(
                Message(
                    ticket_id=ticket.id,
                    role=MessageRole.AGENT,
                    content=text,
                    created_at=FIXED_NOW + timedelta(seconds=offset),
                )
            )
        session.commit()
        run = {
            "ticket_id": str(ticket.id),
            "started_at": (FIXED_NOW + timedelta(seconds=60)).isoformat(),
            "finished_at": (FIXED_NOW + timedelta(seconds=64)).isoformat(),
        }
        assert original_answer(session, {"run": run}) == "this run's answer"
        run["started_at"] = (FIXED_NOW + timedelta(seconds=100)).isoformat()
        run["finished_at"] = (FIXED_NOW + timedelta(seconds=110)).isoformat()
        assert original_answer(session, {"run": run}) is None  # it wrote no answer
