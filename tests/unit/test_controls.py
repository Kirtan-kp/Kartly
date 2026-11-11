"""The operator switches: the store, the agent switch, frozen writes, the API and the CLI."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.api import ApiState, create_app
from kartly.controls import AGENT, WRITES, ControlError, Controls, main
from kartly.gateway import Mode, ToolGateway
from kartly.orchestrator import StopReason, TicketRunner
from kartly.store.models import (
    Action,
    ActorType,
    Approval,
    ApprovalStatus,
    AuditLog,
    Control,
    Order,
    RunStep,
    Ticket,
    TicketStatus,
)
from kartly.tools import RiskTier, default_tools
from kartly.tools.writes import make_refund_tool
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW, make_sqlite_engine, real_ctx, seeded_session_factory
from tests.unit.test_api_approvals import AUTH, Clock, Inline

KEY = "test-api-key-123456"
AUTO = {RiskTier.LOW_RISK_WRITE: Mode.AUTO, RiskTier.MONEY: Mode.AUTO}
REFUND = {"order_id": "KRT-100007", "amount_inr": 599, "reason_code": "damaged"}


def audit_rows(factory: sessionmaker[Session], action: str) -> list[AuditLog]:
    with factory() as session:
        return list(session.scalars(select(AuditLog).where(AuditLog.action == action)))


# ----------------------------------------------------------------------------- the store


def test_every_control_starts_on_its_default(seeded_factory: sessionmaker[Session]) -> None:
    controls = Controls(seeded_factory)
    states = {s.name: s for s in controls.all()}
    assert states[AGENT].value == "on" and states[WRITES].value == "on"
    assert states[AGENT].allowed == ("on", "off") and states[WRITES].allowed == ("on", "frozen")
    assert states[AGENT].updated_at is None and states[AGENT].updated_by is None
    assert controls.agent_on() and controls.writes_on()


def test_setting_a_control_records_who_when_why_and_an_audit_row(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory, clock=lambda: FIXED_NOW)
    state, changed = controls.set(AGENT, "off", "maya", "wrong refunds")
    assert changed and state.value == "off" and not controls.agent_on()
    assert (state.updated_by, state.reason, state.updated_at) == (
        "maya",
        "wrong refunds",
        FIXED_NOW,
    )
    [row] = audit_rows(seeded_factory, "control.set")
    assert row.actor_type is ActorType.HUMAN and row.actor_id == "maya"
    assert row.target_type == "control" and row.target_id == AGENT
    assert row.details == {"from": "on", "to": "off", "reason": "wrong refunds"}


def test_setting_the_value_it_already_has_changes_and_records_nothing(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    assert controls.set(WRITES, "frozen", "maya", "incident")[1] is True
    again, changed = controls.set(WRITES, "frozen", "joe", "me too")
    assert changed is False and again.updated_by == "maya" and again.reason == "incident"
    assert len(audit_rows(seeded_factory, "control.set")) == 1


def test_a_control_can_be_turned_back_and_the_history_is_kept(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    controls.set(AGENT, "off", "maya", "x")
    state, changed = controls.set(AGENT, "on", "joe")
    assert changed and state.value == "on" and state.reason is None
    details = [r.details for r in audit_rows(seeded_factory, "control.set")]
    assert [(d or {})["to"] for d in details] == ["off", "on"]


@pytest.mark.parametrize(
    ("name", "value", "actor", "message"),
    [
        ("nope", "on", "maya", "unknown control"),
        (AGENT, "frozen", "maya", "not allowed"),
        (WRITES, "off", "maya", "not allowed"),
        (AGENT, "", "maya", "not allowed"),
        (AGENT, "off", "", "who"),
        (AGENT, "off", "  ", "who"),
    ],
)
def test_unknown_controls_bad_values_and_anonymous_changes_are_refused(
    seeded_factory: sessionmaker[Session], name: str, value: str, actor: str, message: str
) -> None:
    controls = Controls(seeded_factory)
    with pytest.raises(ControlError, match=message):
        controls.set(name, value, actor)
    assert audit_rows(seeded_factory, "control.set") == [] and controls.agent_on()
    with pytest.raises(ControlError, match="unknown control"):
        controls.get("nope")


def test_a_stored_value_that_is_not_allowed_means_stop() -> None:
    factory = seeded_session_factory(make_sqlite_engine())
    with factory() as session:
        session.add_all(
            [
                Control(name=AGENT, value="maybe", updated_by="x", updated_at=FIXED_NOW),
                Control(name=WRITES, value="ON", updated_by="x", updated_at=FIXED_NOW),
            ]
        )
        session.commit()
    controls = Controls(factory)
    assert not controls.agent_on() and not controls.writes_on()
    assert [s.value for s in controls.all()] == ["off", "frozen"]
    state, changed = controls.set(AGENT, "off", "maya")  # repairing the row is a change
    assert changed and state.value == "off"


def test_a_long_reason_is_cut_and_cleaned(seeded_factory: sessionmaker[Session]) -> None:
    state, _ = Controls(seeded_factory).set(AGENT, "off", "maya", "x\x00" + "y" * 1000)
    assert state.reason is not None and len(state.reason) <= 301 and "\x00" not in state.reason


def test_reading_inside_an_open_session_sees_its_uncommitted_change() -> None:
    factory = seeded_session_factory(make_sqlite_engine())
    controls = Controls(factory)
    with factory() as session:
        session.add(Control(name=AGENT, value="off", updated_by="x", updated_at=FIXED_NOW))
        session.flush()
        assert controls.agent_on(session) is False


# ------------------------------------------------------------------------ the agent switch


def runner(
    factory: sessionmaker[Session], gateway: ToolGateway, controls: Controls, script: list[Any]
) -> tuple[TicketRunner, ScriptedClient]:
    client = ScriptedClient(script)
    return TicketRunner(factory, client, gateway, model_tier="scripted", controls=controls), client


def test_with_the_agent_off_a_ticket_goes_to_a_human_before_any_model_call(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    controls.set(AGENT, "off", "maya", "model is misbehaving")
    gateway = ToolGateway(seeded_factory, controls=controls)
    tickets, client = runner(seeded_factory, gateway, controls, [])
    ticket_id = tickets.create_ticket("C-0001", "Where is KRT-100002?")
    summary = tickets.run(ticket_id)

    assert summary.outcome == "escalated" and summary.stop_reason is StopReason.KILL_SWITCH
    assert summary.answer is None and (summary.llm_calls, summary.tool_calls) == (0, 0)
    assert "maya" in summary.detail and "model is misbehaving" in summary.detail
    assert client.calls == []
    with seeded_factory() as session:
        ticket = session.get(Ticket, ticket_id)
        assert ticket is not None and ticket.status is TicketStatus.ESCALATED
        step = session.scalars(select(RunStep)).one()
        assert step.name == "kill_switch" and step.input == {"control": "agent", "value": "off"}
        assert session.scalars(select(Action)).all() == []
    [row] = audit_rows(seeded_factory, "kill_switch.held")
    assert row.actor_id == "kill_switch" and row.ticket_id == ticket_id
    assert row.details == {"control": "agent", "set_by": "maya", "reason": "model is misbehaving"}


def test_turning_the_agent_back_on_lets_the_next_ticket_through(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = ToolGateway(seeded_factory, controls=controls)
    script = [make_response("Your order is on its way.")]
    tickets, _ = runner(seeded_factory, gateway, controls, script)
    controls.set(AGENT, "off", "maya")
    first = tickets.run(tickets.create_ticket("C-0001", "Where is KRT-100002?"))
    controls.set(AGENT, "on", "maya")
    second = tickets.run(tickets.create_ticket("C-0001", "Where is KRT-100002?"))
    assert first.outcome == "escalated" and second.outcome == "drafted"


def test_a_runner_without_controls_behaves_as_before(
    seeded_factory: sessionmaker[Session],
) -> None:
    client = ScriptedClient([make_response("Hello.")])
    tickets = TicketRunner(seeded_factory, client, ToolGateway(seeded_factory), model_tier="s")
    assert tickets.run(tickets.create_ticket("C-0001", "hi")).outcome == "drafted"


def test_a_corrupt_agent_value_stops_the_agent(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as session:
        session.add(Control(name=AGENT, value="yes", updated_by="x", updated_at=FIXED_NOW))
        session.commit()
    controls = Controls(seeded_factory)
    tickets, client = runner(seeded_factory, ToolGateway(seeded_factory), controls, [])
    assert tickets.run(tickets.create_ticket("C-0001", "hi")).stop_reason is StopReason.KILL_SWITCH
    assert client.calls == []


# ------------------------------------------------------------------------- frozen writes


def refund_gateway(
    factory: sessionmaker[Session], controls: Controls | None, modes: Any = AUTO
) -> ToolGateway:
    return ToolGateway(
        factory,
        (*default_tools(), make_refund_tool()),
        clock=Clock(),
        modes=modes,
        controls=controls,
    )


def refunded(factory: sessionmaker[Session]) -> int:
    with factory() as session:
        order = session.get(Order, "KRT-100007")
        assert order is not None
        return order.refunded_paise


def refund(gateway: ToolGateway, factory: sessionmaker[Session]) -> Any:
    return gateway.call(real_ctx(factory, "C-0004"), "issue_refund", REFUND)


def test_frozen_writes_turn_an_automatic_refund_into_a_proposal(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = refund_gateway(seeded_factory, controls)
    controls.set(WRITES, "frozen", "maya", "incident 42")
    result = refund(gateway, seeded_factory)
    assert result.data is not None and result.data["status"] == "awaiting_approval"
    assert refunded(seeded_factory) == 0
    with seeded_factory() as session:
        assert session.scalars(select(Approval)).one().reason == "writes_frozen"


def test_writes_run_normally_when_not_frozen(seeded_factory: sessionmaker[Session]) -> None:
    controls = Controls(seeded_factory)
    result = refund(refund_gateway(seeded_factory, controls), seeded_factory)
    assert result.ok and refunded(seeded_factory) == 59900


def test_freezing_does_not_touch_reads_or_change_a_shadow_reason(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    controls.set(WRITES, "frozen", "maya")
    gateway = refund_gateway(seeded_factory, controls, modes={})  # default: shadow
    read = gateway.call(real_ctx(seeded_factory, "C-0004"), "get_order", {"order_id": "KRT-100007"})
    assert read.ok
    refund(gateway, seeded_factory)
    with seeded_factory() as session:
        assert session.scalars(select(Approval)).one().reason == "shadow_mode"


def test_a_tier_that_is_off_stays_off_whatever_the_freeze_says(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = refund_gateway(seeded_factory, controls, modes={RiskTier.MONEY: Mode.OFF})
    result = refund(gateway, seeded_factory)
    assert not result.ok and result.error_code == "tool_unavailable"


def test_nothing_can_be_approved_while_writes_are_frozen_but_a_request_can_be_rejected(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = refund_gateway(seeded_factory, controls, modes={})  # proposals
    proposal = refund(gateway, seeded_factory)
    approval_id = uuid.UUID(proposal.data["approval_id"])
    controls.set(WRITES, "frozen", "maya", "incident")

    blocked = gateway.decide(approval_id, approve=True, actor="human:joe")
    assert not blocked.ok and blocked.status == "refused" and "frozen" in blocked.message
    assert refunded(seeded_factory) == 0
    with seeded_factory() as session:
        assert session.scalars(select(Approval)).one().status is ApprovalStatus.PENDING
    assert len(audit_rows(seeded_factory, "approval.blocked_writes_frozen")) == 1

    controls.set(WRITES, "on", "maya")  # unfrozen: the same request can now be approved
    done = gateway.decide(approval_id, approve=True, actor="human:joe")
    assert done.ok and refunded(seeded_factory) == 59900


def test_a_request_can_still_be_rejected_while_frozen(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = refund_gateway(seeded_factory, controls, modes={})
    approval_id = uuid.UUID(refund(gateway, seeded_factory).data["approval_id"])
    controls.set(WRITES, "frozen", "maya")
    rejected = gateway.decide(approval_id, approve=False, actor="human:joe")
    assert rejected.ok and rejected.status == "rejected"


def test_a_repeat_decision_is_still_reported_as_a_repeat_while_frozen(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = refund_gateway(seeded_factory, controls, modes={})
    approval_id = uuid.UUID(refund(gateway, seeded_factory).data["approval_id"])
    assert gateway.decide(approval_id, approve=True, actor="human:joe").ok
    controls.set(WRITES, "frozen", "maya")
    assert gateway.decide(approval_id, approve=True, actor="human:joe").status == "already_decided"


# ------------------------------------------------------------------------------- the API


def client_for(
    factory: sessionmaker[Session], script: list[Any] | None = None, *, with_controls: bool = True
) -> tuple[TestClient, Controls]:
    controls = Controls(factory)
    gateway = ToolGateway(factory, controls=controls)
    tickets = TicketRunner(
        factory,
        ScriptedClient(script or []),
        gateway,
        model_tier="scripted",
        controls=controls,
    )
    state = ApiState(
        session_factory=factory,
        runner=tickets,
        executor=Inline(),
        api_key=KEY,
        gateway=gateway,
        controls=controls if with_controls else None,
    )
    return TestClient(create_app(state)), controls


def test_the_control_routes_need_the_key(seeded_factory: sessionmaker[Session]) -> None:
    http, _ = client_for(seeded_factory)
    body = {"value": "off", "reviewer": "maya"}
    for headers in ({}, {"X-API-Key": "wrong-key-123456"}):
        assert http.get("/v1/controls", headers=headers).status_code == 401
        assert http.put("/v1/controls/agent", json=body, headers=headers).status_code == 401
    assert http.get("/v1/controls", headers=AUTH).status_code == 200


def test_the_routes_list_and_flip_the_switches(seeded_factory: sessionmaker[Session]) -> None:
    http, controls = client_for(seeded_factory)
    listed = http.get("/v1/controls", headers=AUTH).json()["controls"]
    assert [(c["name"], c["value"], c["allowed"]) for c in listed] == [
        ("agent", "on", ["on", "off"]),
        ("writes", "on", ["on", "frozen"]),
    ]
    flipped = http.put(
        "/v1/controls/writes",
        json={"value": "frozen", "reviewer": "maya", "reason": "bad refunds"},
        headers=AUTH,
    )
    assert flipped.status_code == 200
    assert flipped.json()["changed"] is True and flipped.json()["value"] == "frozen"
    assert flipped.json()["updated_by"] == "maya" and flipped.json()["reason"] == "bad refunds"
    assert not controls.writes_on() and controls.agent_on()
    same = http.put(
        "/v1/controls/writes", json={"value": "frozen", "reviewer": "joe"}, headers=AUTH
    )
    assert same.json()["changed"] is False and same.json()["updated_by"] == "maya"


@pytest.mark.parametrize(
    ("path", "body", "status"),
    [
        ("/v1/controls/nope", {"value": "on", "reviewer": "maya"}, 404),
        ("/v1/controls/agent", {"value": "frozen", "reviewer": "maya"}, 422),
        ("/v1/controls/agent", {"value": "off", "reviewer": "a b"}, 422),
        ("/v1/controls/agent", {"value": "off"}, 422),
        ("/v1/controls/agent", {"value": "off", "reviewer": "maya", "extra": 1}, 422),
    ],
)
def test_bad_control_requests_are_refused(
    seeded_factory: sessionmaker[Session], path: str, body: dict[str, Any], status: int
) -> None:
    http, controls = client_for(seeded_factory)
    assert http.put(path, json=body, headers=AUTH).status_code == status
    assert controls.agent_on() and controls.writes_on()


def test_without_controls_the_routes_say_so(seeded_factory: sessionmaker[Session]) -> None:
    http, _ = client_for(seeded_factory, with_controls=False)
    assert http.get("/v1/controls", headers=AUTH).status_code == 503
    body = {"value": "off", "reviewer": "maya"}
    assert http.put("/v1/controls/agent", json=body, headers=AUTH).status_code == 503


def test_flipping_the_agent_over_http_changes_what_the_next_ticket_gets(
    seeded_factory: sessionmaker[Session],
) -> None:
    http, _ = client_for(seeded_factory, [make_response("It is on its way.")])
    off = {"value": "off", "reviewer": "maya", "reason": "test"}
    assert http.put("/v1/controls/agent", json=off, headers=AUTH).status_code == 200
    body = {"customer_id": "C-0001", "message": "Where is KRT-100002?"}
    held = http.post("/v1/tickets?wait=true", json=body, headers=AUTH).json()
    assert held["outcome"] == "escalated" and held["answer"] is None
    on = {"value": "on", "reviewer": "maya"}
    assert http.put("/v1/controls/agent", json=on, headers=AUTH).status_code == 200
    ok = http.post("/v1/tickets?wait=true", json=body, headers=AUTH).json()
    assert ok["outcome"] == "drafted" and ok["answer"] == "It is on its way."


def test_approving_over_http_is_blocked_while_frozen(
    seeded_factory: sessionmaker[Session],
) -> None:
    controls = Controls(seeded_factory)
    gateway = refund_gateway(seeded_factory, controls, modes={})
    tickets = TicketRunner(
        seeded_factory,
        ScriptedClient(
            [
                make_response(None, [tool_call("issue_refund", REFUND)]),
                make_response("A colleague will review your refund."),
            ]
        ),
        gateway,
        model_tier="scripted",
        controls=controls,
    )
    state = ApiState(
        session_factory=seeded_factory,
        runner=tickets,
        executor=Inline(),
        api_key=KEY,
        gateway=gateway,
        controls=controls,
    )
    http = TestClient(create_app(state))
    created = http.post(
        "/v1/tickets?wait=true",
        json={"customer_id": "C-0004", "message": "Refund my bottle please"},
        headers=AUTH,
    )
    assert created.json()["outcome"] == "awaiting_approval"
    approval_id = http.get("/v1/approvals", headers=AUTH).json()["approvals"][0]["id"]
    controls.set(WRITES, "frozen", "maya", "incident")
    blocked = http.post(
        f"/v1/approvals/{approval_id}/approve", json={"reviewer": "joe"}, headers=AUTH
    )
    assert blocked.status_code == 409 and blocked.json()["result"] == "refused"
    assert refunded(seeded_factory) == 0
    assert http.get(f"/v1/approvals/{approval_id}", headers=AUTH).json()["status"] == "pending"


# -------------------------------------------------------------------------------- the CLI


@pytest.fixture
def cli_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    engine = make_sqlite_engine()
    factory = seeded_session_factory(engine)
    monkeypatch.setattr("kartly.store.db.make_engine", lambda *a, **k: engine)
    monkeypatch.setattr(engine, "dispose", lambda *a, **k: None)  # the test db lives in memory
    monkeypatch.setattr("kartly.config.load_env", lambda: None)
    return factory


def test_the_cli_shows_and_flips_the_switches(
    cli_db: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "agent: on (default)" in out and "writes: on (default)" in out

    assert main(["set", "writes", "frozen", "--by", "maya", "--reason", "incident 7"]) == 0
    assert capsys.readouterr().out.startswith("changed: writes: frozen")
    assert main(["set", "writes", "frozen", "--by", "joe"]) == 0
    assert capsys.readouterr().out.startswith("already: writes: frozen")
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "writes: frozen  (set by maya" in out and '"incident 7"' in out
    assert not Controls(cli_db).writes_on()


def test_the_cli_refuses_a_bad_value_with_a_message(
    cli_db: sessionmaker[Session], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["set", "agent", "frozen", "--by", "maya"]) == 2
    assert "not allowed for agent" in capsys.readouterr().err
    assert Controls(cli_db).agent_on()
    with pytest.raises(SystemExit):
        main(["set", "agent", "off"])  # --by is required
    with pytest.raises(SystemExit):
        main(["set", "nope", "off", "--by", "maya"])


def test_the_clock_used_for_the_timestamp_can_be_injected() -> None:
    factory = seeded_session_factory(make_sqlite_engine())
    moment = datetime(2030, 1, 2, 3, 4, tzinfo=UTC)
    state, _ = Controls(factory, clock=lambda: moment).set(AGENT, "off", "maya")
    assert state.updated_at == moment
