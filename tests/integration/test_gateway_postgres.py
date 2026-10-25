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
