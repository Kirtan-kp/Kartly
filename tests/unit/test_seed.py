from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from kartly.store.models import (
    Action,
    ActorType,
    AuditLog,
    Customer,
    Message,
    MessageRole,
    Order,
    OrderItem,
    OrderStatus,
    PaymentMethod,
    Run,
    RunMode,
    RunStep,
    Shipment,
    ShipmentStatus,
    StepKind,
    Ticket,
)
from kartly.store.seed import (
    FIXTURE_ORDER_IDS,
    FIXTURE_ORDERS,
    SeedError,
    SeedResult,
    reset_mock_store,
    seed_mock_store,
)

NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)
SHIPPED_STATES = {OrderStatus.SHIPPED, OrderStatus.OUT_FOR_DELIVERY, OrderStatus.DELIVERED}
ALLOWED_SHIPMENT_STATUS = {
    OrderStatus.SHIPPED: {ShipmentStatus.IN_TRANSIT, ShipmentStatus.EXCEPTION},
    OrderStatus.OUT_FOR_DELIVERY: {ShipmentStatus.OUT_FOR_DELIVERY},
    OrderStatus.DELIVERED: {ShipmentStatus.DELIVERED},
}


def _aware(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; treat them as UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _count(session: Session, model: type[Any]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def _order(session: Session, order_id: str) -> Order:
    order = session.get(Order, order_id)
    assert order is not None, order_id
    return order


def _snapshot(session: Session) -> list[tuple[Any, ...]]:
    return [
        (
            o.id,
            o.customer_id,
            o.status.value,
            o.total_paise,
            o.refunded_paise,
            _aware(o.placed_at).isoformat(),
            tuple((i.sku, i.quantity) for i in o.items),
            tuple(s.tracking_number for s in o.shipments),
        )
        for o in session.scalars(select(Order).order_by(Order.id)).all()
    ]


@pytest.fixture
def seeded(session: Session) -> SeedResult:
    result = seed_mock_store(session, now=NOW)
    session.commit()
    return result


def test_default_counts(session: Session, seeded: SeedResult) -> None:
    assert seeded.customers == 50
    assert seeded.orders == 200
    assert _count(session, Customer) == 50
    assert _count(session, Order) == 200
    assert _count(session, OrderItem) == seeded.order_items
    assert _count(session, Shipment) == seeded.shipments


def test_every_order_status_appears(session: Session, seeded: SeedResult) -> None:
    statuses = set(session.scalars(select(Order.status)).all())
    assert statuses == set(OrderStatus)


def test_order_invariants(session: Session, seeded: SeedResult) -> None:
    for order in session.scalars(select(Order)).all():
        assert order.total_paise == sum(i.quantity * i.unit_price_paise for i in order.items)
        assert 0 <= order.refunded_paise <= order.total_paise
        assert bool(order.shipments) == (order.status in SHIPPED_STATES), order.id
        assert (order.delivered_at is not None) == (order.status is OrderStatus.DELIVERED)
        assert _aware(order.placed_at) <= NOW
        if order.status is OrderStatus.CANCELLED:
            expected = order.total_paise if order.payment_method is PaymentMethod.PREPAID else 0
            assert order.refunded_paise == expected, order.id


def test_shipment_invariants(session: Session, seeded: SeedResult) -> None:
    for order in session.scalars(select(Order)).all():
        for shipment in order.shipments:
            assert shipment.status in ALLOWED_SHIPMENT_STATUS[order.status], order.id
            shipped = _aware(shipment.shipped_at)
            assert _aware(order.placed_at) < shipped <= NOW
            if order.status is OrderStatus.DELIVERED:
                assert order.delivered_at is not None
                assert shipment.delivered_at is not None
                assert shipped < _aware(order.delivered_at) <= NOW
            stamps = [datetime.fromisoformat(e["ts"]) for e in shipment.events]
            assert stamps == sorted(stamps), order.id
            assert all(ts <= NOW for ts in stamps), order.id


def test_seeding_is_deterministic(engine_factory: Callable[[], Engine]) -> None:
    snapshots = {}
    for label, seed in (("a", 42), ("b", 42), ("c", 43)):
        with Session(engine_factory()) as s:
            seed_mock_store(s, seed=seed, now=NOW)
            s.commit()
            snapshots[label] = _snapshot(s)
    assert snapshots["a"] == snapshots["b"]
    assert snapshots["a"] != snapshots["c"]


def test_fixture_customers_and_orders_exist(session: Session, seeded: SeedResult) -> None:
    assert len(FIXTURE_ORDERS) == 8
    ids = set(session.scalars(select(Order.id)).all())
    assert set(FIXTURE_ORDER_IDS) <= ids
    aarav = session.get(Customer, "C-0001")
    assert aarav is not None and aarav.name == "Aarav Sharma"


def test_fixture_order_scenarios(session: Session, seeded: SeedResult) -> None:
    high_value = _order(session, "KRT-100001")
    assert high_value.customer_id == "C-0001"
    assert high_value.status is OrderStatus.DELIVERED
    assert high_value.total_paise == 289_800  # above the INR 1,000 auto-refund threshold

    in_transit = _order(session, "KRT-100002")
    assert in_transit.shipments[0].status is ShipmentStatus.IN_TRANSIT
    eta = in_transit.shipments[0].estimated_delivery
    assert eta is not None and _aware(eta) > NOW

    unshipped = _order(session, "KRT-100003")
    assert unshipped.status is OrderStatus.PLACED
    assert unshipped.customer_id == "C-0002"  # not C-0001: used for cross-customer tests
    assert unshipped.shipments == []

    old = _order(session, "KRT-100004")
    assert old.delivered_at is not None
    assert NOW - _aware(old.delivered_at) >= timedelta(days=45)

    final_sale = _order(session, "KRT-100005")
    assert any(i.final_sale for i in final_sale.items)

    delayed = _order(session, "KRT-100006")
    assert delayed.shipments[0].status is ShipmentStatus.EXCEPTION
    delayed_eta = delayed.shipments[0].estimated_delivery
    assert delayed_eta is not None and _aware(delayed_eta) < NOW

    small = _order(session, "KRT-100007")
    assert small.total_paise == 59_900  # under the INR 1,000 threshold

    partly_refunded = _order(session, "KRT-100008")
    assert partly_refunded.total_paise - partly_refunded.refunded_paise == 199_900


def test_seeding_twice_is_refused(session: Session, seeded: SeedResult) -> None:
    with pytest.raises(SeedError):
        seed_mock_store(session, now=NOW)


def test_minimal_seed_contains_only_fixtures(engine_factory: Callable[[], Engine]) -> None:
    with Session(engine_factory()) as s:
        result = seed_mock_store(s, now=NOW, n_customers=4, n_orders=8)
        assert (result.customers, result.orders) == (4, 8)
        assert set(s.scalars(select(Order.id)).all()) == set(FIXTURE_ORDER_IDS)


def test_too_small_seed_is_rejected(session: Session) -> None:
    with pytest.raises(ValueError):
        seed_mock_store(session, now=NOW, n_orders=3)
    with pytest.raises(ValueError):
        seed_mock_store(session, now=NOW, n_customers=2)


def test_reset_clears_data_but_never_the_audit_log(session: Session, seeded: SeedResult) -> None:
    ticket = Ticket(customer_id="C-0001")
    run = Run(mode=RunMode.SHADOW)
    run.steps = [RunStep(idx=0, kind=StepKind.STAGE, name="intake")]
    ticket.runs = [run]
    ticket.messages = [Message(role=MessageRole.CUSTOMER, content="hi")]
    session.add(ticket)
    session.add(AuditLog(actor_type=ActorType.SYSTEM, action="test"))
    session.commit()

    reset_mock_store(session)
    session.commit()

    for model in (Customer, Order, OrderItem, Shipment, Ticket, Message, Run, RunStep, Action):
        assert _count(session, model) == 0, model
    assert _count(session, AuditLog) == 1

    seed_mock_store(session, now=NOW)  # can be seeded again after a reset
    session.commit()
    assert _count(session, Order) == 200
