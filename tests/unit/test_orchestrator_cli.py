"""The CLI over a real HTTP server and a real (SQLite) database: client, loop, gateway, trace."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import sessionmaker

from kartly.orchestrator.cli import main
from kartly.store.models import AuditLog, Run, Ticket, TicketStatus
from tests.llm_support import FakeOpenAIServer, completion_payload, write_models_config
from tests.support import make_sqlite_engine, seeded_session_factory


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    engine = make_sqlite_engine()
    seeded_session_factory(engine)
    monkeypatch.setattr("kartly.orchestrator.cli.make_engine", lambda: engine)
    monkeypatch.setattr("kartly.orchestrator.cli.load_env", lambda: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("FAKE_KEY", raising=False)
    yield engine
    engine.dispose()


def config(tmp_path: Path, server: FakeOpenAIServer) -> list[str]:
    return [
        "--config",
        str(write_models_config(tmp_path, server.base_url)),
        "--model",
        "fake_small",
    ]


def test_a_ticket_runs_through_the_real_stack(
    db: Engine,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([*config(tmp_path, fake_server), "Where is KRT-100002?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "asks for get_tracking" in out and "-> ok" in out
    assert "Draft reply:\nYour order KRT-100002 is in transit." in out
    assert "is now drafted" in out

    factory = sessionmaker(db)
    with factory() as session:
        assert session.scalars(select(Ticket)).one().status is TicketStatus.DRAFTED
        run = session.scalars(select(Run)).one()
        assert (run.tokens_in, run.tokens_out) == (100, 20)  # two calls of 50 + 10 each
        assert run.model_tier == "fake_small"
        audit = session.scalars(select(AuditLog)).one()
        assert audit.run_id == run.id and audit.actor_id == f"run:{run.id}"

    # the request the model got carries the real tool schema and the system prompt
    first = fake_server.requests[0]["body"]
    assert first["messages"][0]["role"] == "system"
    assert [t["function"]["name"] for t in first["tools"]] == [
        "get_order",
        "get_tracking",
        "cancel_order",
        "issue_refund",
    ]
    assert "customer_id" not in str(first["tools"])


def test_asking_for_someone_elses_order_is_denied_and_recorded(
    db: Engine,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # KRT-100002 belongs to C-0001; the fake model asks for it on behalf of C-0002.
    code = main([*config(tmp_path, fake_server), "--customer", "C-0002", "Where is KRT-100002?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "error order_not_found" in out
    with sessionmaker(db)() as session:
        details = session.scalars(select(AuditLog)).one().details
    assert details is not None and details["security_event"] == "cross_customer_access"


def test_a_provider_failure_escalates_with_exit_code_1(
    db: Engine,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_server.responder = lambda body: (503, {"error": {"message": "overloaded"}}, {})
    code = main(config(tmp_path, fake_server))
    out = capsys.readouterr().out
    assert code == 1
    assert "Escalated to a human: llm_error" in out and "stopped because: llm_error" in out
    with sessionmaker(db)() as session:
        assert session.scalars(select(Ticket)).one().status is TicketStatus.ESCALATED


def test_a_model_that_never_stops_calling_tools_is_cut_off(
    db: Engine,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    counter = iter(range(1, 1000))

    def endless(body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        n = next(counter)
        call = {
            "id": f"call_{n}",
            "type": "function",
            "function": {"name": "get_order", "arguments": f'{{"order_id": "KRT-9{n:05d}"}}'},
        }
        return 200, completion_payload(None, [call]), {}

    fake_server.responder = endless
    code = main(config(tmp_path, fake_server))
    assert code == 1
    assert "step_limit" in capsys.readouterr().out
    assert len(fake_server.requests) == 6  # exactly the step cap, not one more


def test_unknown_customer_exits_2(
    db: Engine,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([*config(tmp_path, fake_server), "--customer", "C-9999", "hi"]) == 2
    assert "C-9999" in capsys.readouterr().err
    assert fake_server.requests == []


def test_missing_api_key_exits_2(
    db: Engine, fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_models_config(tmp_path, fake_server.base_url, key_env="FAKE_KEY")
    assert main(["--config", str(path), "--model", "fake_small"]) == 2
    assert "FAKE_KEY" in capsys.readouterr().err


def test_an_unmigrated_database_gives_a_hint_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty = create_engine("sqlite://")  # no tables
    monkeypatch.setattr("kartly.orchestrator.cli.make_engine", lambda: empty)
    monkeypatch.setattr("kartly.orchestrator.cli.load_env", lambda: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    assert main(config(tmp_path, fake_server)) == 2
    assert "alembic upgrade head" in capsys.readouterr().err


def test_a_router_can_be_named_and_its_label_call_is_in_the_trace(
    db: Engine,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([*config(tmp_path, fake_server), "--router", "fake_small"])
    out = capsys.readouterr().out
    assert "classify: FAILED" in out  # the fake model does not answer in labels; it carried on
    assert fake_server.requests[0]["body"]["response_format"] == {"type": "json_object"}


def test_a_claim_check_model_can_be_named(
    monkeypatch: pytest.MonkeyPatch, db: Engine, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    from kartly.orchestrator import TicketRunner

    seen: dict[str, Any] = {}
    real = TicketRunner

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("kartly.orchestrator.cli.TicketRunner", spy)
    main([*config(tmp_path, fake_server), "--claim-check", "fake_small"])
    assert seen["claim_judge"] is not None and seen["claim_judge"].name == "fake_small"
