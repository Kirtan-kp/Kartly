"""Gateway behaviour that only a real Postgres can prove: JSONB limits, aborted transactions."""

from __future__ import annotations

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.store.models import AuditLog, Customer
from tests.support import CONSTRAINT_TOOL, EXPLODING_TOOL, FIXED_NOW, WRITING_TOOL, make_ctx

pytestmark = pytest.mark.postgres


def test_owner_reads_and_stranger_is_denied(
    pg_gateway: ToolGateway, pg_factory: sessionmaker[Session]
) -> None:
    mine = pg_gateway.call(make_ctx("C-0001"), "get_order", {"order_id": "KRT-100001"})
    theirs = pg_gateway.call(make_ctx("C-0002"), "get_order", {"order_id": "KRT-100001"})

    assert mine.ok and mine.data is not None
    assert mine.data["total_inr"] == "2898.00"
    assert mine.data["delivered_at"] == "2026-10-20T12:00:00+00:00"  # timezone-aware columns
    assert not theirs.ok and theirs.error_code == "order_not_found"

    with pg_factory() as session:
        flagged = session.execute(
            text("SELECT count(*) FROM audit_log WHERE details->>'security_event' = :event"),
            {"event": "cross_customer_access"},
        ).scalar_one()
    assert flagged == 1  # the denial is queryable in JSONB


def test_tracking_marks_a_late_shipment_delayed(pg_gateway: ToolGateway) -> None:
    result = pg_gateway.call(make_ctx("C-0003"), "get_tracking", {"order_id": "KRT-100006"})
    assert result.ok and result.data is not None
    assert result.data["shipments"][0]["delayed"] is True


@pytest.mark.parametrize(
    "hostile",
    [
        {"order_id": "KRT-100001", "note": "nul\x00byte"},
        {"order_id": "KRT-100001", "note": "lone surrogate \ud800"},
        {"order_id": "KRT-100001", "n": float("nan"), "i": float("inf")},
        {"order_id": "KRT-100001", "huge": 10**5000},
        {"order_id": "KRT-100001", "long": "x" * 100_000},
        {"order_id": "KRT-100001", "deep": {"a": {"b": {"c": {"d": {"e": 1}}}}}},
        {"order_id": "KRT-100001", **{f"key{n}": n for n in range(500)}},
        {"order_id": "x" * 100_000},
    ],
    ids=["nul", "surrogate", "nan_inf", "huge_int", "long_string", "deep", "many_keys", "long_id"],
)
def test_hostile_arguments_never_break_the_audit_log(
    pg_gateway: ToolGateway, pg_factory: sessionmaker[Session], hostile: dict[str, object]
) -> None:
    result = pg_gateway.call(make_ctx("C-0001"), "get_order", hostile)
    assert not result.ok and result.error_code == "invalid_arguments"
    with pg_factory() as session:
        rows = list(session.scalars(select(AuditLog)))
    assert len(rows) == 1 and rows[0].action == "tool.rejected"


def test_failed_tools_roll_back_and_the_transaction_survives(
    pg_factory: sessionmaker[Session],
) -> None:
    """A DB error aborts a Postgres transaction; the savepoint must make it recoverable."""
    gateway = ToolGateway(
        pg_factory, [EXPLODING_TOOL, CONSTRAINT_TOOL, WRITING_TOOL], clock=lambda: FIXED_NOW
    )
    ctx = make_ctx("C-0001")
    assert gateway.call(ctx, "explode", {}).error_code == "internal_error"
    assert gateway.call(ctx, "violate", {}).error_code == "internal_error"
    assert gateway.call(ctx, "write", {}).ok

    with pg_factory() as session:
        ids = set(session.scalars(select(Customer.id).where(Customer.id.like("C-%"))))
        actions = [row.action for row in session.scalars(select(AuditLog).order_by(AuditLog.id))]
    assert {"C-BOOM", "C-DUP1", "C-DUP2"}.isdisjoint(ids)
    assert "C-WROTE" in ids
    assert actions == ["tool.failed", "tool.failed", "tool.succeeded"]


# ------------------------------------------------------------------------------ write path


def _write_gateway(pg_factory: sessionmaker[Session], mode: str) -> ToolGateway:
    from kartly.gateway import Mode
    from kartly.tools import WRITE_TOOLS, RiskTier, default_tools

    modes = {RiskTier.LOW_RISK_WRITE: Mode(mode), RiskTier.MONEY: Mode(mode)}
    return ToolGateway(
        pg_factory, (*default_tools(), *WRITE_TOOLS), clock=lambda: FIXED_NOW, modes=modes
    )


def test_a_proposal_is_approved_and_runs_once_on_postgres(
    pg_factory: sessionmaker[Session],
) -> None:
    import uuid

    from kartly.store.models import Action, ActionStatus, Approval, ApprovalStatus, Order
    from tests.support import real_ctx

    gw = _write_gateway(pg_factory, "shadow")
    proposal = gw.call(real_ctx(pg_factory), "cancel_order", {"order_id": "KRT-100003"})
    assert proposal.data is not None and proposal.data["status"] == "awaiting_approval"
    decision = gw.decide(uuid.UUID(proposal.data["approval_id"]), approve=True, actor="agent:maya")
    assert decision.ok and decision.status == "executed"
    with pg_factory() as session:
        order = session.get(Order, "KRT-100003")
        assert order is not None and order.status.value == "cancelled"
        assert session.scalars(select(Action)).one().status == ActionStatus.SUCCEEDED
        approval = session.scalars(select(Approval)).one()
        assert approval.status == ApprovalStatus.APPROVED and approval.decided_at is not None


def test_identical_calls_at_the_same_moment_cancel_the_order_once(
    pg_factory: sessionmaker[Session],
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from kartly.store.models import Action
    from tests.support import real_ctx

    gw = _write_gateway(pg_factory, "auto")
    ctx = real_ctx(pg_factory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _i: gw.call(ctx, "cancel_order", {"order_id": "KRT-100003"}), range(8))
        )
    assert all(r.ok or r.error_code == "action_in_progress" for r in results)
    assert sum(1 for r in results if r.ok and r.data and not r.data.get("replayed")) == 1
    with pg_factory() as session:
        assert len(session.scalars(select(Action)).all()) == 1


def test_concurrent_refunds_cannot_pay_out_more_than_the_order_total(
    pg_factory: sessionmaker[Session],
) -> None:
    """Five different refunds of about INR 600 race on a 2,898 order: the row lock lets exactly
    four through. Without it each reads the same old total and all five succeed."""
    from concurrent.futures import ThreadPoolExecutor

    from kartly.gateway import Mode
    from kartly.store.models import Order
    from kartly.tools import RiskTier, default_tools
    from kartly.tools.writes import make_refund_tool
    from tests.support import real_ctx

    modes = {RiskTier.LOW_RISK_WRITE: Mode.AUTO, RiskTier.MONEY: Mode.AUTO}
    gw = ToolGateway(
        pg_factory,
        (*default_tools(), make_refund_tool(100_000, 100_000)),
        clock=lambda: FIXED_NOW,
        modes=modes,
    )
    ctx = real_ctx(pg_factory, "C-0001")
    amounts = [600.00, 600.01, 600.02, 600.03, 600.04]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(
            pool.map(
                lambda a: gw.call(
                    ctx,
                    "issue_refund",
                    {"order_id": "KRT-100001", "amount_inr": a, "reason_code": "damaged"},
                ),
                amounts,
            )
        )
    done = [r for r in results if r.ok]
    assert len(done) == 4, [(r.error_code, r.message) for r in results]
    assert all(r.error_code == "exceeds_refundable" for r in results if not r.ok)
    with pg_factory() as session:
        order = session.get(Order, "KRT-100001")
        assert order is not None and order.refunded_paise <= order.total_paise
        assert order.refunded_paise == sum(
            round(a * 100) for a, r in zip(amounts, results, strict=True) if r.ok
        )


def test_two_reviewers_approving_at_once_pay_the_refund_once(
    pg_factory: sessionmaker[Session],
) -> None:
    """Both read the request as pending; the lock on the approval row makes the second one wait
    and then see "already decided". Without it both run the refund."""
    import uuid
    from concurrent.futures import ThreadPoolExecutor

    from kartly.store.models import Action, ActionStatus, Order, Ticket, TicketStatus
    from kartly.tools import default_tools
    from kartly.tools.writes import make_refund_tool
    from tests.support import real_ctx

    gw = ToolGateway(
        pg_factory,
        (*default_tools(), make_refund_tool()),
        clock=lambda: FIXED_NOW,
    )  # default modes: shadow, so the refund is a proposal
    ctx = real_ctx(pg_factory, "C-0004")
    proposal = gw.call(
        ctx,
        "issue_refund",
        {"order_id": "KRT-100007", "amount_inr": 599, "reason_code": "damaged"},
    )
    assert proposal.data is not None
    approval_id = uuid.UUID(proposal.data["approval_id"])
    with pg_factory() as session:
        ticket = session.get(Ticket, ctx.ticket_id)
        assert ticket is not None
        ticket.status = TicketStatus.AWAITING_APPROVAL
        session.commit()

    with ThreadPoolExecutor(max_workers=6) as pool:
        decisions = list(
            pool.map(lambda i: gw.decide(approval_id, approve=True, actor=f"human:r{i}"), range(6))
        )
    assert sorted(d.status for d in decisions) == ["already_decided"] * 5 + ["executed"]
    with pg_factory() as session:
        order = session.get(Order, "KRT-100007")
        assert order is not None and order.refunded_paise == 59900
        assert session.scalars(select(Action)).one().status == ActionStatus.SUCCEEDED
        ticket = session.get(Ticket, ctx.ticket_id)
        assert ticket is not None and ticket.status == TicketStatus.DRAFTED
