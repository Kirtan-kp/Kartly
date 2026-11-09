"""The gateway's write path: modes, idempotency, proposals, approvals; and ``cancel_order``."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from kartly.config import Settings
from kartly.gateway import DEFAULT_MODES, Mode, ToolGateway, modes_from_settings
from kartly.gateway.gateway import APPROVAL_TTL, canonical_json, idempotency_key, payload_hash
from kartly.orchestrator import TicketRunner
from kartly.store.models import (
    Action,
    ActionStatus,
    ActorType,
    Approval,
    ApprovalStatus,
    AuditLog,
    Order,
    OrderStatus,
    Run,
    Ticket,
    TicketStatus,
)
from kartly.tools import WRITE_TOOLS, RiskTier, Tool, ToolEnv, default_tools
from kartly.tools.base import CallContext
from kartly.tools.writes import CANCELLABLE
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW, real_ctx

ROOT = Path(__file__).resolve().parents[2]
AUTO = {RiskTier.LOW_RISK_WRITE: Mode.AUTO, RiskTier.MONEY: Mode.AUTO}
OFF = {RiskTier.LOW_RISK_WRITE: Mode.OFF, RiskTier.MONEY: Mode.OFF}
UNSHIPPED = "KRT-100003"  # C-0002, placed
SHIPPED = "KRT-100002"  # C-0001, shipped


class Clock:
    def __init__(self) -> None:
        self.now = FIXED_NOW

    def __call__(self) -> Any:
        return self.now


def gateway(
    factory: sessionmaker[Session],
    modes: dict[RiskTier, Mode] | None = None,
    clock: Clock | None = None,
    tools: tuple[Tool, ...] | None = None,
    ttl: timedelta = APPROVAL_TTL,
) -> ToolGateway:
    return ToolGateway(
        factory,
        tools or (*default_tools(), *WRITE_TOOLS),
        clock=clock or Clock(),
        modes=modes,
        approval_ttl=ttl,
    )


make_ctx = real_ctx


def order_status(factory: sessionmaker[Session], order_id: str = UNSHIPPED) -> OrderStatus:
    with factory() as session:
        order = session.get(Order, order_id)
        assert order is not None
        return order.status


def set_status(
    factory: sessionmaker[Session], status: OrderStatus, order_id: str = UNSHIPPED
) -> None:
    with factory() as session:
        order = session.get(Order, order_id)
        assert order is not None
        order.status = status
        session.commit()


def actions(factory: sessionmaker[Session]) -> list[Action]:
    with factory() as session:
        return list(session.scalars(select(Action).order_by(Action.created_at)))


def approvals(factory: sessionmaker[Session]) -> list[Approval]:
    with factory() as session:
        return list(session.scalars(select(Approval)))


def audit(factory: sessionmaker[Session], prefix: str = "") -> list[AuditLog]:
    with factory() as session:
        rows = session.scalars(select(AuditLog).order_by(AuditLog.id))
        return [r for r in rows if r.action.startswith(prefix)]


# ----------------------------------------------------------------------------- the tool itself


def test_the_cancellable_statuses_match_the_policy_text() -> None:
    text = (ROOT / "kb" / "cancellation.md").read_text(encoding="utf-8")
    assert 'Orders show "Packed" or "Shipped" once they are no longer cancellable' in text
    assert set(CANCELLABLE) == {OrderStatus.PLACED, OrderStatus.CONFIRMED}


def test_a_write_tool_must_have_a_preview_and_a_read_tool_must_not() -> None:
    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    def handler(_e: ToolEnv, _a: Any) -> dict[str, Any]:
        return {}

    with pytest.raises(ValueError, match="needs a preview"):
        Tool("w", RiskTier.MONEY, "d", NoArgs, handler)
    with pytest.raises(ValueError, match="only write tools"):
        Tool("r", RiskTier.READ_ONLY, "d", NoArgs, handler, preview=handler)
    with pytest.raises(ValueError, match="only write tools"):
        Tool("r", RiskTier.READ_ONLY, "d", NoArgs, handler, needs_approval=lambda _e, _a: "x")


# --------------------------------------------------------------------------------- the modes


def test_by_default_a_write_is_a_proposal_and_changes_nothing(
    seeded_factory: sessionmaker[Session],
) -> None:
    assert DEFAULT_MODES == {RiskTier.LOW_RISK_WRITE: Mode.SHADOW, RiskTier.MONEY: Mode.SHADOW}
    ctx = make_ctx(seeded_factory)
    result = gateway(seeded_factory).call(ctx, "cancel_order", {"order_id": UNSHIPPED})

    assert result.ok and result.data is not None
    assert result.data["status"] == "awaiting_approval"
    assert "NOT been done" in result.data["note"]
    assert result.data["would_do"]["action"] == "cancel_order"
    assert result.data["would_do"]["current_status"] == "placed"
    assert order_status(seeded_factory) == OrderStatus.PLACED  # nothing changed

    [action] = actions(seeded_factory)
    assert action.status == ActionStatus.AWAITING_APPROVAL and action.tool_name == "cancel_order"
    assert action.args == {"order_id": UNSHIPPED}
    [approval] = approvals(seeded_factory)
    assert approval.status == ApprovalStatus.PENDING and approval.reason == "shadow_mode"
    assert approval.action_id == action.id and approval.ticket_id == ctx.ticket_id
    assert approval.payload_hash == payload_hash("cancel_order", action.args, "C-0002")
    assert approval.expires_at.replace(tzinfo=None) == (FIXED_NOW + APPROVAL_TTL).replace(
        tzinfo=None
    )
    [row] = audit(seeded_factory, "tool.")
    assert row.action == "tool.proposed" and row.details is not None
    assert row.details["action_id"] == str(action.id) and row.details["reason"] == "shadow_mode"


def test_in_auto_mode_the_cancellation_runs_and_is_recorded(
    seeded_factory: sessionmaker[Session],
) -> None:
    result = gateway(seeded_factory, AUTO).call(
        make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED}
    )
    assert result.ok and result.data is not None
    assert result.data["status"] == "cancelled" and result.data["order_id"] == UNSHIPPED
    assert order_status(seeded_factory) == OrderStatus.CANCELLED
    [action] = actions(seeded_factory)
    assert action.status == ActionStatus.SUCCEEDED and action.result is not None
    assert action.result["status"] == "cancelled" and approvals(seeded_factory) == []
    [row] = audit(seeded_factory, "tool.")
    assert row.action == "tool.succeeded" and row.details is not None
    assert row.details["action_id"] == str(action.id) and row.target_id == UNSHIPPED


def test_the_refund_due_is_the_prepaid_amount_and_nothing_for_cash_on_delivery(
    seeded_factory: sessionmaker[Session],
) -> None:
    from kartly.store.models import PaymentMethod
    from kartly.tools.orders import inr

    with seeded_factory() as session:
        order = session.get(Order, UNSHIPPED)
        assert order is not None
        order.payment_method = PaymentMethod.PREPAID
        order.refunded_paise = 0
        total = order.total_paise
        session.commit()
    preview = gateway(seeded_factory).call(
        make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED}
    )
    assert preview.data is not None and preview.data["would_do"]["refund_due_inr"] == inr(total)
    set_status(seeded_factory, OrderStatus.PLACED)
    with seeded_factory() as session:
        order = session.get(Order, UNSHIPPED)
        assert order is not None
        order.payment_method = PaymentMethod.COD
        session.commit()
    cod = gateway(seeded_factory, AUTO).call(
        make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED}
    )
    assert cod.data is not None and cod.data["refund_due_inr"] is None


def test_when_a_tier_is_off_the_model_never_sees_the_tool_and_cannot_call_it(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, OFF)
    assert "cancel_order" not in gw.tool_names
    assert [s["function"]["name"] for s in gw.tool_specs()] == ["get_order", "get_tracking"]
    assert (
        gw.tool_specs(allowed=["get_order", "cancel_order"])[0]["function"]["name"] == "get_order"
    )
    assert len(gw.tool_specs(allowed=["get_order", "cancel_order"])) == 1

    result = gw.call(make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED})
    assert not result.ok and result.error_code == "tool_unavailable"
    assert order_status(seeded_factory) == OrderStatus.PLACED and actions(seeded_factory) == []
    [row] = audit(seeded_factory, "tool.")
    assert row.details is not None and row.details["security_event"] == "disallowed_tool_call"


def test_read_tools_ignore_the_write_modes_and_modes_can_be_set_per_tier(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, {RiskTier.LOW_RISK_WRITE: Mode.OFF})
    assert gw.mode_for(default_tools()[0]) == Mode.AUTO
    assert gw.mode_for(WRITE_TOOLS[0]) == Mode.OFF
    settings = Settings()
    assert (settings.tier1_mode, settings.tier2_mode) == ("shadow", "shadow")
    assert modes_from_settings("auto", "off") == {
        RiskTier.LOW_RISK_WRITE: Mode.AUTO,
        RiskTier.MONEY: Mode.OFF,
    }
    with pytest.raises(ValueError):
        modes_from_settings("sometimes", "off")


# ----------------------------------------------------------------------------- idempotency


def test_a_repeated_call_gets_the_first_result_and_runs_once(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, AUTO)
    ctx = make_ctx(seeded_factory)
    first = gw.call(ctx, "cancel_order", {"order_id": UNSHIPPED})
    again = gw.call(ctx, "cancel_order", {"order_id": "krt-100003"})  # same after validation
    assert first.ok and again.ok and again.data is not None and again.data["replayed"] is True
    assert again.data["status"] == "cancelled"
    assert len(actions(seeded_factory)) == 1
    assert [r.action for r in audit(seeded_factory, "tool.")] == ["tool.succeeded", "tool.replayed"]


def test_a_repeated_proposal_makes_one_approval(seeded_factory: sessionmaker[Session]) -> None:
    gw = gateway(seeded_factory)
    ctx = make_ctx(seeded_factory)
    gw.call(ctx, "cancel_order", {"order_id": UNSHIPPED})
    again = gw.call(ctx, "cancel_order", {"order_id": UNSHIPPED})
    assert again.data is not None and again.data["replayed"] is True
    assert again.data["status"] == "awaiting_approval" and "NOT been done" in again.data["note"]
    assert len(actions(seeded_factory)) == 1 and len(approvals(seeded_factory)) == 1


def test_a_new_run_is_a_new_key_but_the_tool_still_refuses_to_cancel_twice(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, AUTO)
    assert gw.call(make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED}).ok
    second = gw.call(make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED})
    assert not second.ok and second.error_code == "already_cancelled"
    assert len(actions(seeded_factory)) == 1  # a refused preview creates no action


def test_the_key_comes_from_the_run_the_tool_and_the_arguments_only() -> None:
    run = uuid.uuid4()
    args = {"order_id": "KRT-100003"}
    key = idempotency_key(run, "cancel_order", args)
    assert key == idempotency_key(run, "cancel_order", {"order_id": "KRT-100003"})
    assert key != idempotency_key(uuid.uuid4(), "cancel_order", args)
    assert key != idempotency_key(run, "other_tool", args)
    assert key != idempotency_key(run, "cancel_order", {"order_id": "KRT-100004"})
    assert key.startswith(f"{run}:cancel_order:") and len(key) <= 128
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert payload_hash("t", args, "C-1") != payload_hash("t", args, "C-2")


def test_the_model_cannot_supply_a_key_or_an_identity(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, AUTO)
    ctx = make_ctx(seeded_factory)
    key = gw.call(ctx, "cancel_order", {"order_id": UNSHIPPED, "idempotency_key": "x"})
    ident = gw.call(ctx, "cancel_order", {"order_id": UNSHIPPED, "customer_id": "C-0001"})
    assert key.error_code == ident.error_code == "invalid_arguments"
    assert order_status(seeded_factory) == OrderStatus.PLACED and actions(seeded_factory) == []
    events = [r.details.get("security_event") for r in audit(seeded_factory, "tool.") if r.details]
    assert events == [None, "identity_argument_supplied"]


def test_a_lost_race_for_the_key_is_a_replay_but_other_database_errors_are_not_hidden(
    seeded_factory: sessionmaker[Session],
) -> None:
    ctx = make_ctx(seeded_factory)
    with seeded_factory() as session:
        args = {"order_id": UNSHIPPED}
        first = ToolGateway._claim(
            session, ctx, "cancel_order", args, "k1", ActionStatus.PENDING, None
        )
        second = ToolGateway._claim(
            session, ctx, "cancel_order", args, "k1", ActionStatus.PENDING, None
        )
        assert first is not None and second is None
        orphan = CallContext(uuid.uuid4(), uuid.uuid4(), "C-0002")  # no such ticket or run
        with pytest.raises(Exception, match="(?i)integrity|constraint|foreign"):
            ToolGateway._claim(
                session, orphan, "cancel_order", args, "k2", ActionStatus.PENDING, None
            )


# ---------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize("mode", [Mode.SHADOW, Mode.AUTO])
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (OrderStatus.PACKED, "not_cancellable"),
        (OrderStatus.SHIPPED, "not_cancellable"),
        (OrderStatus.OUT_FOR_DELIVERY, "not_cancellable"),
        (OrderStatus.DELIVERED, "not_cancellable"),
        (OrderStatus.CANCELLED, "already_cancelled"),
    ],
)
def test_an_order_that_cannot_be_cancelled_is_refused_in_every_mode_and_nothing_is_queued(
    seeded_factory: sessionmaker[Session], mode: Mode, status: OrderStatus, code: str
) -> None:
    set_status(seeded_factory, status)
    result = gateway(seeded_factory, {RiskTier.LOW_RISK_WRITE: mode}).call(
        make_ctx(seeded_factory), "cancel_order", {"order_id": UNSHIPPED}
    )
    assert not result.ok and result.error_code == code
    assert order_status(seeded_factory) == status
    assert actions(seeded_factory) == [] and approvals(seeded_factory) == []
    [row] = audit(seeded_factory, "tool.")
    assert row.action == "tool.rejected"


def test_cancelling_someone_elses_order_looks_like_a_missing_order_and_is_logged(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, AUTO)
    foreign = gw.call(make_ctx(seeded_factory, "C-0001"), "cancel_order", {"order_id": UNSHIPPED})
    missing = gw.call(
        make_ctx(seeded_factory, "C-0001"), "cancel_order", {"order_id": "KRT-999999"}
    )
    assert not foreign.ok and foreign.message is not None and missing.message is not None
    assert foreign.error_code == missing.error_code == "order_not_found"
    assert foreign.message.replace(UNSHIPPED, "X") == missing.message.replace("KRT-999999", "X")
    assert order_status(seeded_factory) == OrderStatus.PLACED and actions(seeded_factory) == []
    events = [r.details.get("security_event") for r in audit(seeded_factory, "tool.") if r.details]
    assert events == ["cross_customer_access", None]


# --------------------------------------------------------------------------------- approvals


def propose(factory: sessionmaker[Session], gw: ToolGateway, order: str = UNSHIPPED) -> uuid.UUID:
    result = gw.call(make_ctx(factory), "cancel_order", {"order_id": order})
    assert result.data is not None
    return uuid.UUID(result.data["approval_id"])


def test_approving_runs_the_action_once_and_records_who_did_it(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory)
    approval_id = propose(seeded_factory, gw)
    decision = gw.decide(approval_id, approve=True, actor="agent:maya", note="ok")

    assert decision.ok and decision.status == "executed" and decision.data is not None
    assert decision.data["status"] == "cancelled"
    assert order_status(seeded_factory) == OrderStatus.CANCELLED
    [action] = actions(seeded_factory)
    assert action.status == ActionStatus.SUCCEEDED
    [approval] = approvals(seeded_factory)
    assert approval.status == ApprovalStatus.APPROVED
    assert approval.decided_by == "agent:maya" and approval.decision_note == "ok"
    assert approval.decided_at is not None
    [row] = audit(seeded_factory, "approval.")
    assert (row.action, row.actor_type, row.actor_id) == (
        "approval.approved",
        ActorType.HUMAN,
        "agent:maya",
    )
    assert row.details is not None and row.details["outcome"] == "succeeded"

    again = gw.decide(approval_id, approve=True, actor="agent:maya")  # a second click
    assert not again.ok and again.status == "already_decided"
    assert [r.action for r in audit(seeded_factory, "approval.")] == [
        "approval.approved",
        "approval.repeat_refused",
    ]
    assert len(actions(seeded_factory)) == 1


def test_rejecting_changes_nothing_and_cannot_be_undone_by_approving(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory)
    approval_id = propose(seeded_factory, gw)
    decision = gw.decide(approval_id, approve=False, actor="agent:maya", note="not needed")
    assert decision.ok and decision.status == "rejected"
    assert order_status(seeded_factory) == OrderStatus.PLACED
    assert actions(seeded_factory)[0].status == ActionStatus.REJECTED
    assert approvals(seeded_factory)[0].status == ApprovalStatus.REJECTED
    assert gw.decide(approval_id, approve=True, actor="agent:maya").status == "already_decided"
    assert order_status(seeded_factory) == OrderStatus.PLACED


def test_an_approval_that_waited_too_long_expires_and_runs_nothing(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    gw = gateway(seeded_factory, clock=clock, ttl=timedelta(hours=1))
    approval_id = propose(seeded_factory, gw)
    clock.now = FIXED_NOW + timedelta(hours=1)  # the expiry instant itself is too late
    decision = gw.decide(approval_id, approve=True, actor="agent:maya")
    assert not decision.ok and decision.status == "expired"
    assert order_status(seeded_factory) == OrderStatus.PLACED
    assert approvals(seeded_factory)[0].status == ApprovalStatus.EXPIRED
    assert actions(seeded_factory)[0].status == ActionStatus.REJECTED
    assert [r.action for r in audit(seeded_factory, "approval.")] == ["approval.expired"]


def test_just_before_expiry_an_approval_still_works(seeded_factory: sessionmaker[Session]) -> None:
    clock = Clock()
    gw = gateway(seeded_factory, clock=clock, ttl=timedelta(hours=1))
    approval_id = propose(seeded_factory, gw)
    clock.now = FIXED_NOW + timedelta(minutes=59)
    assert gw.decide(approval_id, approve=True, actor="agent:maya").status == "executed"


def test_an_action_changed_after_it_was_proposed_is_not_run(
    seeded_factory: sessionmaker[Session],
) -> None:
    """Approval bait-and-switch: the human saw one order; the stored action now names another."""
    gw = gateway(seeded_factory)
    approval_id = propose(seeded_factory, gw)
    with seeded_factory() as session:
        action = session.scalars(select(Action)).one()
        action.args = {"order_id": "KRT-100004"}
        session.commit()
    decision = gw.decide(approval_id, approve=True, actor="agent:maya")
    assert not decision.ok and decision.status == "refused"
    assert order_status(seeded_factory) == OrderStatus.PLACED
    assert order_status(seeded_factory, "KRT-100004") == OrderStatus.DELIVERED
    assert approvals(seeded_factory)[0].status == ApprovalStatus.REJECTED
    [row] = audit(seeded_factory, "approval.")
    assert row.action == "approval.payload_mismatch" and row.details is not None
    assert row.details["security_event"] == "approval_payload_mismatch"


def test_the_tools_checks_run_again_at_approval_time(seeded_factory: sessionmaker[Session]) -> None:
    gw = gateway(seeded_factory)
    approval_id = propose(seeded_factory, gw)
    set_status(seeded_factory, OrderStatus.SHIPPED)  # it shipped while the request waited
    decision = gw.decide(approval_id, approve=True, actor="agent:maya")
    assert not decision.ok and decision.status == "not_executed"
    assert "can no longer be cancelled" in decision.message
    assert order_status(seeded_factory) == OrderStatus.SHIPPED
    assert actions(seeded_factory)[0].status == ActionStatus.REJECTED
    assert approvals(seeded_factory)[0].status == ApprovalStatus.APPROVED  # a person did decide
    [row] = audit(seeded_factory, "approval.")
    assert row.details is not None and row.details["error_code"] == "not_cancellable"


def test_an_unknown_approval_and_a_vanished_tool_are_refused(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory)
    assert gw.decide(uuid.uuid4(), approve=True, actor="a").status == "refused"
    approval_id = propose(seeded_factory, gw)
    smaller = gateway(seeded_factory, tools=default_tools())  # cancel_order no longer registered
    decision = smaller.decide(approval_id, approve=True, actor="agent:maya")
    assert decision.status == "refused" and "no longer available" in decision.message
    assert order_status(seeded_factory) == OrderStatus.PLACED


def test_a_failing_audit_write_during_a_decision_leaves_everything_undecided(
    seeded_factory: sessionmaker[Session], engine: Any
) -> None:
    from sqlalchemy import event

    gw = gateway(seeded_factory)
    approval_id = propose(seeded_factory, gw)

    def block(_c: Any, _cur: Any, statement: str, *_r: Any) -> None:
        if statement.lstrip().lower().startswith("insert into audit_log"):
            raise RuntimeError("audit store unavailable")

    event.listen(engine, "before_cursor_execute", block)
    try:
        with pytest.raises(RuntimeError, match="audit store unavailable"):
            gw.decide(approval_id, approve=True, actor="agent:maya")
    finally:
        event.remove(engine, "before_cursor_execute", block)
    assert order_status(seeded_factory) == OrderStatus.PLACED
    assert approvals(seeded_factory)[0].status == ApprovalStatus.PENDING


# ---------------------------------------------------------------- a tool that needs approval


class Amount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: int


def money_tool(limit: int) -> Tool:
    def handler(env: ToolEnv, args: Amount) -> dict[str, Any]:
        from kartly.store.models import Customer

        env.session.add(Customer(id=f"C-PAID{args.amount}", name="p", email=f"p{args.amount}@x.io"))
        env.session.flush()
        return {"paid": args.amount}

    return Tool(
        name="pay",
        tier=RiskTier.MONEY,
        description="pay",
        args_model=Amount,
        handler=handler,
        preview=lambda _e, a: {"pay": a.amount},
        needs_approval=lambda _e, a: "over_limit" if a.amount > limit else None,
    )


def test_a_tool_can_require_a_human_even_in_auto_mode(
    seeded_factory: sessionmaker[Session],
) -> None:
    gw = gateway(seeded_factory, AUTO, tools=(money_tool(100),))
    small = gw.call(make_ctx(seeded_factory), "pay", {"amount": 100})
    large = gw.call(make_ctx(seeded_factory), "pay", {"amount": 101})
    assert small.ok and small.data == {"paid": 100}
    assert large.data is not None and large.data["status"] == "awaiting_approval"
    [approval] = approvals(seeded_factory)
    assert approval.reason == "over_limit"
    with seeded_factory() as session:
        ids = set(session.scalars(select(func.lower("id")).select_from(Order).limit(0)))
        from kartly.store.models import Customer

        names = set(session.scalars(select(Customer.id).where(Customer.id.like("C-PAID%"))))
    assert ids == set() and names == {"C-PAID100"}  # the large one has not been paid


def test_shadow_mode_wins_over_the_tools_own_rule(seeded_factory: sessionmaker[Session]) -> None:
    gw = gateway(seeded_factory, {RiskTier.MONEY: Mode.SHADOW}, tools=(money_tool(100),))
    gw.call(make_ctx(seeded_factory), "pay", {"amount": 1})
    assert approvals(seeded_factory)[0].reason == "shadow_mode"


# ------------------------------------------------------------------------ the ticket runner

SEARCHLESS_CANCEL = make_response(None, [tool_call("cancel_order", {"order_id": UNSHIPPED})])


def run_ticket(factory: sessionmaker[Session], modes: dict[RiskTier, Mode] | None) -> Any:
    client = ScriptedClient([SEARCHLESS_CANCEL, make_response("Done, I have sent that on.")])
    runner = TicketRunner(factory, client, gateway(factory, modes), model_tier="scripted")
    return runner.run(runner.create_ticket("C-0002", "Please cancel order KRT-100003"))


def test_a_run_that_proposed_an_action_waits_for_approval(
    seeded_factory: sessionmaker[Session],
) -> None:
    summary = run_ticket(seeded_factory, None)
    assert summary.outcome == "awaiting_approval"
    with seeded_factory() as session:
        ticket = session.get(Ticket, summary.ticket_id)
        run = session.get(Run, summary.run_id)
        assert ticket is not None and run is not None
        assert (
            ticket.status == TicketStatus.AWAITING_APPROVAL and run.outcome == "awaiting_approval"
        )
    assert order_status(seeded_factory) == OrderStatus.PLACED


def test_a_run_that_acted_in_auto_mode_is_an_ordinary_draft(
    seeded_factory: sessionmaker[Session],
) -> None:
    summary = run_ticket(seeded_factory, AUTO)
    assert summary.outcome == "drafted"
    assert order_status(seeded_factory) == OrderStatus.CANCELLED
    with seeded_factory() as session:
        ticket = session.get(Ticket, summary.ticket_id)
        assert ticket is not None and ticket.status == TicketStatus.DRAFTED


def test_a_never_autonomous_tool_is_always_a_proposal_whatever_the_setting(
    seeded_factory: sessionmaker[Session],
) -> None:
    base = money_tool(10**9)
    tool = Tool("wire", RiskTier.NEVER_AUTONOMOUS, "d", Amount, base.handler, preview=base.preview)
    everything_auto = {t: Mode.AUTO for t in RiskTier}
    gw = gateway(seeded_factory, everything_auto, tools=(tool,))
    assert gw.mode_for(tool) == Mode.SHADOW
    result = gw.call(make_ctx(seeded_factory), "wire", {"amount": 5})
    assert result.data is not None and result.data["status"] == "awaiting_approval"
    unset = ToolGateway(seeded_factory, (tool,), clock=Clock(), modes={})
    assert unset.mode_for(tool) == Mode.SHADOW


def test_a_preview_that_writes_has_its_writes_undone(seeded_factory: sessionmaker[Session]) -> None:
    from kartly.store.models import Customer
    from kartly.tools import ToolError

    def sneaky_preview(env: ToolEnv, _a: Any) -> dict[str, Any]:
        env.session.add(Customer(id="C-SNEAK", name="s", email="s@x.io"))
        env.session.flush()
        raise ToolError("not_allowed", "No.")

    base = money_tool(10)
    tool = Tool("sneak", RiskTier.MONEY, "d", Amount, base.handler, preview=sneaky_preview)
    result = gateway(seeded_factory, tools=(tool,)).call(
        make_ctx(seeded_factory), "sneak", {"amount": 1}
    )
    assert not result.ok
    with seeded_factory() as session:
        assert session.get(Customer, "C-SNEAK") is None
