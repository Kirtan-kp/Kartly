"""``issue_refund``: the amount rules, the two limits, approvals above them, and ownership."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.config import Settings
from kartly.gateway import Mode, ToolGateway, build_tool_specs
from kartly.retrieval.setup import gateway_tools
from kartly.store.models import (
    Action,
    ActionStatus,
    Approval,
    ApprovalStatus,
    AuditLog,
    Order,
    OrderStatus,
    PaymentMethod,
)
from kartly.tools import RiskTier, default_tools
from kartly.tools.writes import (
    DEFAULT_AUTO_LIMIT_INR,
    DEFAULT_DAILY_CAP_INR,
    RefundArgs,
    make_refund_tool,
    write_tools,
)
from tests.support import FIXED_NOW, real_ctx

ROOT = Path(__file__).resolve().parents[2]
SMALL = "KRT-100007"  # C-0004, delivered, INR 599, nothing refunded
BIG = "KRT-100001"  # C-0001, delivered, INR 2,898
PARTLY = "KRT-100008"  # C-0004, delivered, INR 3,499 of which INR 1,500 is refunded
SHIPPED = "KRT-100002"  # C-0001
AUTO = {RiskTier.LOW_RISK_WRITE: Mode.AUTO, RiskTier.MONEY: Mode.AUTO}


def settings_from(**values: object) -> Settings:
    """Settings that ignore any .env file."""
    return Settings(_env_file=None, **values)  # type: ignore[call-arg,arg-type]


class Clock:
    def __init__(self) -> None:
        self.now = FIXED_NOW

    def __call__(self) -> Any:
        return self.now


def gw(
    factory: sessionmaker[Session],
    modes: dict[RiskTier, Mode] | None = AUTO,
    clock: Clock | None = None,
    limit: int = DEFAULT_AUTO_LIMIT_INR,
    cap: int = DEFAULT_DAILY_CAP_INR,
) -> ToolGateway:
    tools = (*default_tools(), make_refund_tool(limit, cap))
    return ToolGateway(factory, tools, clock=clock or Clock(), modes=modes)


def refund(
    gateway: ToolGateway,
    factory: sessionmaker[Session],
    order: str,
    amount: Any,
    customer: str | None = None,
    reason: str = "damaged",
) -> Any:
    owner = customer or {"KRT-100001": "C-0001", "KRT-100002": "C-0001"}.get(order, "C-0004")
    return gateway.call(
        real_ctx(factory, owner),
        "issue_refund",
        {"order_id": order, "amount_inr": amount, "reason_code": reason},
    )


def refunded(factory: sessionmaker[Session], order: str) -> int:
    with factory() as session:
        row = session.get(Order, order)
        assert row is not None
        return row.refunded_paise


def pending(factory: sessionmaker[Session]) -> list[Approval]:
    with factory() as session:
        return list(session.scalars(select(Approval)))


# -------------------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    ("amount", "paise"),
    [(599, 59900), (599.0, 59900), ("599", 59900), (1999.99, 199999), (0.01, 1), (0.1, 10)],
)
def test_amounts_become_whole_paise(amount: Any, paise: int) -> None:
    args = RefundArgs(order_id="krt-100007", amount_inr=amount, reason_code="damaged")
    assert args.amount_paise == paise and args.order_id == "KRT-100007"


@pytest.mark.parametrize(
    "amount", [0, -5, 599.999, 0.001, "abc", None, float("nan"), float("inf"), 1_000_001]
)
def test_bad_amounts_are_refused(amount: Any) -> None:
    with pytest.raises(ValidationError):
        RefundArgs(order_id="KRT-100007", amount_inr=amount, reason_code="damaged")


@pytest.mark.parametrize(
    "extra",
    [{"customer_id": "C-0001"}, {"idempotency_key": "x"}, {"approved": True}],
)
def test_the_model_cannot_pass_an_identity_a_key_or_an_approval(extra: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RefundArgs(order_id="KRT-100007", amount_inr=5, reason_code="damaged", **extra)


def test_the_reason_must_be_one_of_the_known_codes_and_the_schema_says_so() -> None:
    with pytest.raises(ValidationError):
        RefundArgs(order_id="KRT-100007", amount_inr=5, reason_code="because")  # type: ignore[arg-type]
    [spec] = build_tool_specs([make_refund_tool()])
    props = spec["function"]["parameters"]["properties"]
    assert set(props) == {"order_id", "amount_inr", "reason_code"}
    assert props["reason_code"]["enum"][0] == "damaged"
    assert "customer_id" not in str(spec)


# ----------------------------------------------------------------------------- under the limit


def test_a_small_refund_runs_and_is_recorded(seeded_factory: sessionmaker[Session]) -> None:
    result = refund(gw(seeded_factory), seeded_factory, SMALL, 599)
    assert result.ok and result.data is not None
    assert result.data["refunded_inr"] == "599.00" and result.data["total_refunded_inr"] == "599.00"
    assert result.data["remaining_refundable_inr"] == "0.00"
    assert refunded(seeded_factory, SMALL) == 59900
    with seeded_factory() as session:
        action = session.scalars(select(Action)).one()
        assert action.status == ActionStatus.SUCCEEDED and action.tool_name == "issue_refund"
        assert action.args["amount_inr"] == 599.0 and action.args["reason_code"] == "damaged"
        audit = session.scalars(select(AuditLog).where(AuditLog.action == "tool.succeeded")).one()
        assert audit.details is not None and audit.details["action_id"] == str(action.id)
    assert pending(seeded_factory) == []


def test_in_shadow_mode_even_a_small_refund_only_proposes(
    seeded_factory: sessionmaker[Session],
) -> None:
    result = refund(gw(seeded_factory, None), seeded_factory, SMALL, 599)
    assert result.data is not None and result.data["status"] == "awaiting_approval"
    assert result.data["would_do"]["amount_inr"] == "599.00"
    assert result.data["would_do"]["refundable_after_inr"] == "0.00"
    assert refunded(seeded_factory, SMALL) == 0
    assert pending(seeded_factory)[0].reason == "shadow_mode"


def test_the_auto_limit_is_inclusive(seeded_factory: sessionmaker[Session]) -> None:
    gateway = gw(seeded_factory)
    exactly = refund(gateway, seeded_factory, BIG, 1000)
    assert exactly.data is not None and exactly.data["refunded_inr"] == "1000.00"
    assert refunded(seeded_factory, BIG) == 100_000 and pending(seeded_factory) == []


def test_one_paisa_over_the_limit_needs_a_human(seeded_factory: sessionmaker[Session]) -> None:
    result = refund(gw(seeded_factory), seeded_factory, BIG, 1000.01)
    assert result.data is not None and result.data["status"] == "awaiting_approval"
    assert refunded(seeded_factory, BIG) == 0
    [approval] = pending(seeded_factory)
    assert approval.reason == "over_auto_limit" and approval.status == ApprovalStatus.PENDING


def test_splitting_a_large_refund_does_not_get_around_the_limit(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory)
    first = refund(gateway, seeded_factory, BIG, 600)
    second = refund(gateway, seeded_factory, BIG, 600)  # 1,200 on the order in total
    assert first.data is not None and first.data["refunded_inr"] == "600.00"
    assert second.data is not None and second.data["status"] == "awaiting_approval"
    assert refunded(seeded_factory, BIG) == 60_000
    assert pending(seeded_factory)[0].reason == "over_auto_limit"


def test_an_order_already_refunded_past_the_limit_sends_every_further_refund_to_a_human(
    seeded_factory: sessionmaker[Session],
) -> None:
    result = refund(gw(seeded_factory), seeded_factory, PARTLY, 10)
    assert result.data is not None and result.data["status"] == "awaiting_approval"
    assert refunded(seeded_factory, PARTLY) == 150_000


def test_the_limits_are_settings(seeded_factory: sessionmaker[Session]) -> None:
    result = refund(gw(seeded_factory, limit=500), seeded_factory, SMALL, 599)
    assert result.data is not None and result.data["status"] == "awaiting_approval"
    settings = settings_from(refund_auto_limit_inr=200, refund_daily_cap_inr=300)
    tools = gateway_tools(settings)
    gateway = ToolGateway(seeded_factory, tools, clock=Clock(), modes=AUTO)
    over = refund(gateway, seeded_factory, SMALL, 250)
    assert over.data is not None and over.data["status"] == "awaiting_approval"
    assert (
        settings_from().refund_auto_limit_inr,
        settings_from().refund_daily_cap_inr,
    ) == (1000, 5000)


def test_the_default_limit_is_what_the_policy_text_says() -> None:
    text = (ROOT / "kb" / "refunds.md").read_text(encoding="utf-8")
    assert "Refunds up to INR 1,000 on an order are processed automatically" in text
    assert DEFAULT_AUTO_LIMIT_INR == 1000
    assert {t.name for t in write_tools()} == {"cancel_order", "issue_refund"}


# --------------------------------------------------------------------------------- the daily cap


def two_small_orders(factory: sessionmaker[Session]) -> None:
    with factory() as session:  # make a second order with room under the limit
        other = session.get(Order, PARTLY)
        assert other is not None
        other.refunded_paise = 0
        session.commit()


def test_refunds_over_the_daily_cap_need_a_human(seeded_factory: sessionmaker[Session]) -> None:
    two_small_orders(seeded_factory)
    gateway = gw(seeded_factory, cap=1000)
    first = refund(gateway, seeded_factory, SMALL, 599)
    second = refund(gateway, seeded_factory, PARTLY, 500)  # 1,099 in a day
    assert first.data is not None and first.data["refunded_inr"] == "599.00"
    assert second.data is not None and second.data["status"] == "awaiting_approval"
    assert refunded(seeded_factory, PARTLY) == 0
    assert pending(seeded_factory)[0].reason == "daily_cap"


def test_the_cap_is_exact_and_counts_only_the_last_24_hours(
    seeded_factory: sessionmaker[Session],
) -> None:
    two_small_orders(seeded_factory)
    clock = Clock()
    gateway = gw(seeded_factory, clock=clock, cap=1000)
    refund(gateway, seeded_factory, SMALL, 599)
    exactly = refund(gateway, seeded_factory, PARTLY, 401)  # 1,000 in total: at the cap, allowed
    assert exactly.data is not None and exactly.data["refunded_inr"] == "401.00"

    clock.now = FIXED_NOW + timedelta(days=1)  # the first refund is exactly 24 hours old
    with seeded_factory() as session:
        for order_id in (SMALL, PARTLY):
            row = session.get(Order, order_id)
            assert row is not None
            row.refunded_paise = 0
        session.commit()
    still_counted = refund(gateway, seeded_factory, SMALL, 1)
    assert still_counted.data is not None and still_counted.data["status"] == "awaiting_approval"

    clock.now = FIXED_NOW + timedelta(days=1, seconds=1)  # now both are older than 24 hours
    later = refund(gateway, seeded_factory, SMALL, 599)
    assert later.data is not None and later.data["refunded_inr"] == "599.00"


def test_the_cap_is_per_customer_and_ignores_refunds_that_never_ran(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory, cap=700)
    refund(gateway, seeded_factory, BIG, 2000)  # a proposal: no money has moved
    refund(gateway, seeded_factory, BIG, 5, customer="C-0001")  # another customer's refund
    ok = refund(gateway, seeded_factory, SMALL, 599)  # C-0004: nothing refunded today
    assert ok.data is not None and ok.data["refunded_inr"] == "599.00"


# -------------------------------------------------------------------------------- the rules


def test_a_refund_cannot_exceed_what_is_left_to_refund(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory)
    over = refund(gateway, seeded_factory, PARTLY, 2000)  # INR 1,999 is left
    assert not over.ok and over.error_code == "exceeds_refundable"
    assert over.message == "Only INR 1999.00 of this order can still be refunded."
    assert refunded(seeded_factory, PARTLY) == 150_000 and pending(seeded_factory) == []
    exact = refund(gateway, seeded_factory, PARTLY, 1999)  # all that is left: for a human
    assert exact.data is not None and exact.data["would_do"]["refundable_after_inr"] == "0.00"


@pytest.mark.parametrize("status", [OrderStatus.PLACED, OrderStatus.SHIPPED, OrderStatus.PACKED])
def test_an_order_that_has_not_been_delivered_cannot_be_refunded(
    seeded_factory: sessionmaker[Session], status: OrderStatus
) -> None:
    with seeded_factory() as session:
        row = session.get(Order, SMALL)
        assert row is not None
        row.status = status
        session.commit()
    result = refund(gw(seeded_factory), seeded_factory, SMALL, 100)
    assert not result.ok and result.error_code == "not_refundable"
    assert refunded(seeded_factory, SMALL) == 0


def test_a_cancelled_prepaid_order_can_be_refunded_but_a_cancelled_cod_order_has_nothing(
    seeded_factory: sessionmaker[Session],
) -> None:
    def set_order(method: PaymentMethod) -> None:
        with seeded_factory() as session:
            row = session.get(Order, SMALL)
            assert row is not None
            row.status, row.payment_method = OrderStatus.CANCELLED, method
            session.commit()

    set_order(PaymentMethod.PREPAID)
    ok = refund(gw(seeded_factory), seeded_factory, SMALL, 599, reason="order_cancelled")
    assert ok.ok and refunded(seeded_factory, SMALL) == 59900
    with seeded_factory() as session:
        row = session.get(Order, SMALL)
        assert row is not None
        row.refunded_paise = 0
        session.commit()
    set_order(PaymentMethod.COD)
    nothing = refund(gw(seeded_factory), seeded_factory, SMALL, 599)
    assert not nothing.ok and nothing.error_code == "nothing_to_refund"


def test_refunding_someone_elses_order_looks_like_a_missing_order(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory)
    foreign = refund(gateway, seeded_factory, SMALL, 100, customer="C-0001")
    assert not foreign.ok and foreign.error_code == "order_not_found"
    assert refunded(seeded_factory, SMALL) == 0
    with seeded_factory() as session:
        audit = session.scalars(select(AuditLog)).one()
        assert audit.details is not None
        assert audit.details["security_event"] == "cross_customer_access"
        assert session.scalars(select(Action)).all() == []


def test_a_repeated_refund_in_one_run_is_paid_once(seeded_factory: sessionmaker[Session]) -> None:
    gateway = gw(seeded_factory)
    ctx = real_ctx(seeded_factory, "C-0004")
    args = {"order_id": SMALL, "amount_inr": 300, "reason_code": "damaged"}
    first = gateway.call(ctx, "issue_refund", args)
    again = gateway.call(ctx, "issue_refund", {**args, "amount_inr": 300.0})
    assert first.ok and again.data is not None and again.data["replayed"] is True
    assert refunded(seeded_factory, SMALL) == 30_000


# ------------------------------------------------------------------------------- approvals


def test_a_human_approving_a_large_refund_moves_the_money_once(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory)
    proposal = refund(gateway, seeded_factory, BIG, 2898)
    assert proposal.data is not None
    approval_id = uuid.UUID(proposal.data["approval_id"])
    decision = gateway.decide(approval_id, approve=True, actor="agent:maya")
    assert decision.ok and decision.data is not None
    assert decision.data["refunded_inr"] == "2898.00"
    assert decision.data["remaining_refundable_inr"] == "0.00"
    assert refunded(seeded_factory, BIG) == 289_800
    assert gateway.decide(approval_id, approve=True, actor="agent:maya").status == "already_decided"
    assert refunded(seeded_factory, BIG) == 289_800


def test_a_human_approval_is_not_blocked_by_the_daily_cap(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory, cap=100)
    proposal = refund(gateway, seeded_factory, SMALL, 599)  # over the cap, so a proposal
    assert proposal.data is not None and pending(seeded_factory)[0].reason == "daily_cap"
    decision = gateway.decide(
        uuid.UUID(proposal.data["approval_id"]), approve=True, actor="agent:maya"
    )
    assert decision.ok and refunded(seeded_factory, SMALL) == 59900


def test_the_amount_is_checked_again_when_a_human_approves(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory)
    proposal = refund(gateway, seeded_factory, BIG, 2898)
    assert proposal.data is not None
    with seeded_factory() as session:  # someone refunded part of it while the request waited
        row = session.get(Order, BIG)
        assert row is not None
        row.refunded_paise = 100_000
        session.commit()
    decision = gateway.decide(
        uuid.UUID(proposal.data["approval_id"]), approve=True, actor="agent:maya"
    )
    assert not decision.ok and decision.status == "not_executed"
    assert "Only INR 1898.00" in decision.message
    assert refunded(seeded_factory, BIG) == 100_000


def test_an_edited_amount_is_not_what_the_human_approved(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway = gw(seeded_factory)
    proposal = refund(gateway, seeded_factory, BIG, 1500)
    assert proposal.data is not None
    with seeded_factory() as session:
        action = session.scalars(select(Action)).one()
        action.args = {**action.args, "amount_inr": 2898.0}
        session.commit()
    decision = gateway.decide(
        uuid.UUID(proposal.data["approval_id"]), approve=True, actor="agent:maya"
    )
    assert decision.status == "refused" and refunded(seeded_factory, BIG) == 0
