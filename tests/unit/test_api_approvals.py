"""The approvals routes, the ticket state after a decision, and the expiry sweep."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from concurrent.futures import Future
from datetime import timedelta
from typing import Any, TypeVar

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.api import ApiState, create_app
from kartly.gateway import Mode, ToolGateway
from kartly.orchestrator import TicketRunner
from kartly.store.models import (
    Action,
    ActionStatus,
    Approval,
    ApprovalStatus,
    AuditLog,
    Order,
    Ticket,
    TicketStatus,
)
from kartly.tools import RiskTier, default_tools
from kartly.tools.writes import make_refund_tool
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW, real_ctx

T = TypeVar("T")
KEY = "test-api-key-123456"
AUTH = {"X-API-Key": KEY}
SMALL = "KRT-100007"  # C-0004, delivered, INR 599
BIG = "KRT-100001"  # C-0001, delivered, INR 2,898
SHADOW = {RiskTier.LOW_RISK_WRITE: Mode.SHADOW, RiskTier.MONEY: Mode.SHADOW}


class Inline:
    def submit(self, fn: Callable[[], T]) -> Future[T]:
        future: Future[T] = Future()
        try:
            future.set_result(fn())
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.now = FIXED_NOW

    def __call__(self) -> Any:
        return self.now


def make_gateway(factory: sessionmaker[Session], clock: Clock | None = None) -> ToolGateway:
    tools = (*default_tools(), make_refund_tool())
    return ToolGateway(factory, tools, clock=clock or Clock(), modes=SHADOW)


def refund_script(order: str = SMALL, amount: float = 599) -> list[Any]:
    args = {"order_id": order, "amount_inr": amount, "reason_code": "damaged"}
    return [
        make_response(None, [tool_call("issue_refund", args)]),
        make_response("Your refund request is with a colleague for review."),
    ]


def app_for(
    factory: sessionmaker[Session], gateway: ToolGateway | None, script: list[Any]
) -> TestClient:
    runner = TicketRunner(
        factory, ScriptedClient(script), gateway or make_gateway(factory), model_tier="scripted"
    )
    state = ApiState(
        session_factory=factory,
        runner=runner,
        executor=Inline(),
        api_key=KEY,
        gateway=gateway,
    )
    return TestClient(create_app(state))


def open_request(http: TestClient, customer: str = "C-0004") -> uuid.UUID:
    created = http.post(
        "/v1/tickets?wait=true",
        json={"customer_id": customer, "message": "My item arrived damaged, refund please"},
        headers=AUTH,
    )
    assert created.status_code in (200, 202), created.text
    assert created.json()["outcome"] == "awaiting_approval"
    return uuid.UUID(created.json()["ticket_id"])


def pending_id(http: TestClient) -> str:
    listed = http.get("/v1/approvals", headers=AUTH).json()
    [only] = listed["approvals"]
    return str(only["id"])


def decide(
    http: TestClient, approval_id: str, verb: str, reviewer: str = "maya", **extra: Any
) -> Any:
    return http.post(
        f"/v1/approvals/{approval_id}/{verb}",
        json={"reviewer": reviewer, **extra},
        headers=AUTH,
    )


def ticket_status(factory: sessionmaker[Session], ticket_id: uuid.UUID) -> TicketStatus:
    with factory() as session:
        ticket = session.get(Ticket, ticket_id)
        assert ticket is not None
        return ticket.status


def refunded(factory: sessionmaker[Session], order: str) -> int:
    with factory() as session:
        row = session.get(Order, order)
        assert row is not None
        return row.refunded_paise


# ------------------------------------------------------------------------------ access


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/v1/approvals"),
        ("get", f"/v1/approvals/{uuid.uuid4()}"),
        ("post", f"/v1/approvals/{uuid.uuid4()}/approve"),
        ("post", f"/v1/approvals/{uuid.uuid4()}/reject"),
    ],
)
def test_every_approval_route_needs_the_key(
    seeded_factory: sessionmaker[Session], method: str, path: str
) -> None:
    http = app_for(seeded_factory, make_gateway(seeded_factory), [])
    for headers in ({}, {"X-API-Key": "wrong-key-123456"}):
        response = http.request(method, path, headers=headers, json={"reviewer": "maya"})
        assert response.status_code == 401


def test_without_a_gateway_the_routes_say_so(seeded_factory: sessionmaker[Session]) -> None:
    runner = TicketRunner(
        seeded_factory, ScriptedClient([]), make_gateway(seeded_factory), model_tier="x"
    )
    state = ApiState(session_factory=seeded_factory, runner=runner, executor=Inline(), api_key=KEY)
    http = TestClient(create_app(state))
    assert http.get("/v1/approvals", headers=AUTH).status_code == 503


def test_unknown_requests_are_404_and_bad_bodies_422(
    seeded_factory: sessionmaker[Session],
) -> None:
    http = app_for(seeded_factory, make_gateway(seeded_factory), [])
    missing = str(uuid.uuid4())
    assert http.get(f"/v1/approvals/{missing}", headers=AUTH).status_code == 404
    assert decide(http, missing, "approve").status_code == 404
    assert decide(http, missing, "reject").status_code == 404
    real = str(uuid.uuid4())
    for body in ({}, {"reviewer": ""}, {"reviewer": "a b"}, {"reviewer": "m", "extra": 1}):
        response = http.post(f"/v1/approvals/{real}/approve", json=body, headers=AUTH)
        assert response.status_code == 422
    assert http.get("/v1/approvals?status=bogus", headers=AUTH).status_code == 422
    assert http.get("/v1/approvals?limit=0", headers=AUTH).status_code == 422


# ------------------------------------------------------------------- approve and reject


def test_a_refund_waits_is_listed_and_is_paid_once_when_approved(
    seeded_factory: sessionmaker[Session],
) -> None:
    http = app_for(seeded_factory, make_gateway(seeded_factory), refund_script())
    ticket_id = open_request(http)
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.AWAITING_APPROVAL
    assert refunded(seeded_factory, SMALL) == 0

    [item] = http.get("/v1/approvals", headers=AUTH).json()["approvals"]
    assert item["status"] == "pending" and item["reason"] == "shadow_mode"
    assert item["tool"] == "issue_refund" and item["customer_id"] == "C-0004"
    assert item["args"]["amount_inr"] == 599.0
    assert item["preview"]["amount_inr"] == "599.00" and item["outcome"] is None

    response = decide(http, item["id"], "approve", note="photo checked")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] and body["result"] == "executed" and body["ticket_status"] == "drafted"
    assert body["data"]["refunded_inr"] == "599.00"
    assert refunded(seeded_factory, SMALL) == 59900
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.DRAFTED

    again = decide(http, item["id"], "approve")
    assert again.status_code == 409 and again.json()["result"] == "already_decided"
    assert decide(http, item["id"], "reject").status_code == 409
    assert refunded(seeded_factory, SMALL) == 59900

    done = http.get(f"/v1/approvals/{item['id']}", headers=AUTH).json()
    assert done["status"] == "approved" and done["decided_by"] == "human:maya"
    assert done["decision_note"] == "photo checked" and done["preview"] is None
    assert done["outcome"]["refunded_inr"] == "599.00" and done["action_status"] == "succeeded"
    assert http.get("/v1/approvals", headers=AUTH).json()["approvals"] == []
    both = http.get("/v1/approvals?status=all", headers=AUTH).json()["approvals"]
    assert [a["id"] for a in both] == [item["id"]]

    with seeded_factory() as session:
        audits = session.scalars(select(AuditLog).where(AuditLog.action == "approval.approved"))
        [row] = list(audits)
        assert row.actor_id == "human:maya" and row.ticket_id == ticket_id


def test_a_rejected_refund_changes_nothing_and_escalates_the_ticket(
    seeded_factory: sessionmaker[Session],
) -> None:
    http = app_for(seeded_factory, make_gateway(seeded_factory), refund_script())
    ticket_id = open_request(http)
    approval_id = pending_id(http)
    response = decide(http, approval_id, "reject", note="not in the photos")
    assert response.status_code == 200
    assert response.json()["result"] == "rejected"
    assert response.json()["ticket_status"] == "escalated"
    assert refunded(seeded_factory, SMALL) == 0
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.ESCALATED
    assert decide(http, approval_id, "approve").status_code == 409  # a rejection is final
    assert refunded(seeded_factory, SMALL) == 0
    shown = http.get(f"/v1/approvals/{approval_id}", headers=AUTH).json()
    assert shown["status"] == "rejected" and shown["action_status"] == "rejected"


def test_an_approval_fails_cleanly_when_the_order_changed_while_it_waited(
    seeded_factory: sessionmaker[Session],
) -> None:
    http = app_for(seeded_factory, make_gateway(seeded_factory), refund_script())
    ticket_id = open_request(http)
    approval_id = pending_id(http)
    with seeded_factory() as session:
        order = session.get(Order, SMALL)
        assert order is not None
        order.refunded_paise = 50_000  # INR 500 was refunded some other way: only INR 99 is left
        session.commit()
    response = decide(http, approval_id, "approve")
    assert response.status_code == 409 and response.json()["result"] == "not_executed"
    assert "Only INR 99.00" in response.json()["message"]
    assert refunded(seeded_factory, SMALL) == 50_000
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.ESCALATED


def test_a_tampered_action_is_refused_and_escalates(
    seeded_factory: sessionmaker[Session],
) -> None:
    http = app_for(seeded_factory, make_gateway(seeded_factory), refund_script())
    ticket_id = open_request(http)
    approval_id = pending_id(http)
    with seeded_factory() as session:
        action = session.scalars(select(Action)).one()
        action.args = {**action.args, "amount_inr": 599.0, "reason_code": "other"}
        session.commit()
    response = decide(http, approval_id, "approve")
    assert response.status_code == 409 and response.json()["result"] == "refused"
    assert refunded(seeded_factory, SMALL) == 0
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.ESCALATED


# ------------------------------------------------------------------------------ expiry


def test_listing_closes_expired_requests_and_escalates_their_tickets(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    http = app_for(seeded_factory, make_gateway(seeded_factory, clock), refund_script())
    ticket_id = open_request(http)
    approval_id = pending_id(http)

    clock.now = FIXED_NOW + timedelta(hours=23, minutes=59)
    still = http.get("/v1/approvals", headers=AUTH).json()
    assert still["expired_now"] == 0 and len(still["approvals"]) == 1

    clock.now = FIXED_NOW + timedelta(hours=24)
    swept = http.get("/v1/approvals", headers=AUTH).json()
    assert swept["expired_now"] == 1 and swept["approvals"] == []
    assert http.get("/v1/approvals", headers=AUTH).json()["expired_now"] == 0
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.ESCALATED
    shown = http.get(f"/v1/approvals/{approval_id}", headers=AUTH).json()
    assert shown["status"] == "expired" and shown["action_status"] == "rejected"
    assert decide(http, approval_id, "approve").status_code == 409
    assert refunded(seeded_factory, SMALL) == 0
    with seeded_factory() as session:
        row = session.scalars(select(AuditLog).where(AuditLog.action == "approval.expired")).one()
        assert row.actor_id == "gateway" and row.actor_type.value == "system"


def test_approving_an_expired_request_is_refused_even_before_any_sweep(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    http = app_for(seeded_factory, make_gateway(seeded_factory, clock), refund_script())
    ticket_id = open_request(http)
    approval_id = pending_id(http)
    clock.now = FIXED_NOW + timedelta(days=2)
    response = decide(http, approval_id, "approve")
    assert response.status_code == 409 and response.json()["result"] == "expired"
    assert refunded(seeded_factory, SMALL) == 0
    assert ticket_status(seeded_factory, ticket_id) is TicketStatus.ESCALATED


# ------------------------------------------------------- ticket state across requests


def propose(gateway: ToolGateway, ctx: Any, order: str, amount: float) -> uuid.UUID:
    result = gateway.call(
        ctx,
        "issue_refund",
        {"order_id": order, "amount_inr": amount, "reason_code": "damaged"},
    )
    assert result.data is not None
    return uuid.UUID(result.data["approval_id"])


def test_a_ticket_waits_until_every_request_on_it_is_decided(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = make_gateway(seeded_factory)
    ctx = real_ctx(seeded_factory, "C-0001")
    first = propose(gateway, ctx, BIG, 100)
    second = propose(gateway, ctx, BIG, 200)
    with seeded_factory() as session:
        ticket = session.get(Ticket, ctx.ticket_id)
        assert ticket is not None
        ticket.status = TicketStatus.AWAITING_APPROVAL
        session.commit()

    assert gateway.decide(first, approve=True, actor="human:maya").ok
    assert ticket_status(seeded_factory, ctx.ticket_id) is TicketStatus.AWAITING_APPROVAL
    assert gateway.decide(second, approve=False, actor="human:maya").ok
    assert ticket_status(seeded_factory, ctx.ticket_id) is TicketStatus.ESCALATED
    assert refunded(seeded_factory, BIG) == 10_000  # only the approved one ran


def test_a_ticket_with_every_request_carried_out_is_drafted(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = make_gateway(seeded_factory)
    ctx = real_ctx(seeded_factory, "C-0001")
    ids = [propose(gateway, ctx, BIG, 100), propose(gateway, ctx, BIG, 200)]
    with seeded_factory() as session:
        ticket = session.get(Ticket, ctx.ticket_id)
        assert ticket is not None
        ticket.status = TicketStatus.AWAITING_APPROVAL
        session.commit()
    for approval_id in ids:
        assert gateway.decide(approval_id, approve=True, actor="human:maya").ok
    assert ticket_status(seeded_factory, ctx.ticket_id) is TicketStatus.DRAFTED
    assert refunded(seeded_factory, BIG) == 30_000


@pytest.mark.parametrize("status", [TicketStatus.DRAFTED, TicketStatus.CLOSED, TicketStatus.SENT])
def test_a_decision_never_moves_a_ticket_that_is_not_waiting(
    seeded_factory: sessionmaker[Session], status: TicketStatus
) -> None:
    gateway = make_gateway(seeded_factory)
    ctx = real_ctx(seeded_factory, "C-0001")
    approval_id = propose(gateway, ctx, BIG, 100)
    with seeded_factory() as session:
        ticket = session.get(Ticket, ctx.ticket_id)
        assert ticket is not None
        ticket.status = status
        session.commit()
    assert gateway.decide(approval_id, approve=False, actor="human:maya").ok
    assert ticket_status(seeded_factory, ctx.ticket_id) is status


def test_expire_stale_leaves_decided_and_fresh_requests_alone(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    gateway = make_gateway(seeded_factory, clock)
    ctx = real_ctx(seeded_factory, "C-0001")
    decided = propose(gateway, ctx, BIG, 100)
    assert gateway.decide(decided, approve=True, actor="human:maya").ok
    clock.now = FIXED_NOW + timedelta(hours=1)
    fresh = propose(gateway, ctx, BIG, 200)
    clock.now = FIXED_NOW + timedelta(hours=24, minutes=30)  # the first expired, the second not
    assert gateway.expire_stale() == 0
    with seeded_factory() as session:
        statuses = {a.id: a.status for a in session.scalars(select(Approval))}
    assert statuses[decided] is ApprovalStatus.APPROVED
    assert statuses[fresh] is ApprovalStatus.PENDING
    with seeded_factory() as session:
        action = session.scalars(select(Action).order_by(Action.created_at)).first()
        assert action is not None and action.status is ActionStatus.SUCCEEDED
