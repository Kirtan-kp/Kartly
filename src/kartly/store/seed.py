"""Deterministic synthetic data for the mock store.

    python -m kartly.store.seed --reset

loads ~50 customers and ~200 orders in varied states. The first four customers and eight orders
are hand-crafted fixtures with stable IDs so eval cases can refer to them (see FIXTURE_ORDERS);
everything else is generated from a seeded random generator, so the same seed and clock always
give the same data. All data is synthetic (emails use the reserved example.com domain).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from kartly.store.db import make_engine, make_session_factory
from kartly.store.models import (
    Action,
    Customer,
    CustomerTier,
    Message,
    Order,
    OrderItem,
    OrderStatus,
    PaymentMethod,
    Run,
    RunStep,
    Shipment,
    ShipmentStatus,
    Ticket,
)

# ------------------------------------------------------------------ reference data


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    price_paise: int
    final_sale: bool = False


PRODUCTS: tuple[Product, ...] = (
    Product("SKU-EL-001", "Wireless Earbuds Pro", "electronics", 249_900),
    Product("SKU-EL-002", "Bluetooth Speaker Mini", "electronics", 199_900),
    Product("SKU-EL-003", "USB-C Fast Charger 65W", "electronics", 149_900),
    Product("SKU-EL-004", "Smartwatch Lite", "electronics", 349_900),
    Product("SKU-EL-005", "Phone Case Clear", "electronics", 39_900),
    Product("SKU-AP-001", "Cotton Crew T-Shirt", "apparel", 79_900),
    Product("SKU-AP-002", "Slim Fit Jeans", "apparel", 189_900),
    Product("SKU-AP-003", "Running Shoes", "apparel", 329_900),
    Product("SKU-AP-004", "Hooded Sweatshirt", "apparel", 149_900),
    Product("SKU-AP-005", "Innerwear Pack of 3", "apparel", 59_900, final_sale=True),
    Product("SKU-HM-001", "Ceramic Dinner Set", "home", 299_900),
    Product("SKU-HM-002", "Cotton Bedsheet Set", "home", 129_900),
    Product("SKU-HM-003", "LED Desk Lamp", "home", 89_900),
    Product("SKU-HM-004", "Steel Water Bottle", "home", 59_900),
    Product("SKU-BT-001", "Vitamin C Serum", "beauty", 69_900, final_sale=True),
    Product("SKU-BT-002", "Matte Lipstick", "beauty", 49_900, final_sale=True),
    Product("SKU-BT-003", "Face Wash 100ml", "beauty", 29_900, final_sale=True),
    Product("SKU-BK-001", "Paperback Novel", "books", 39_900),
    Product("SKU-BK-002", "Notebook Set of 4", "books", 34_900),
    Product("SKU-TY-001", "Building Blocks 500pc", "toys", 129_900),
)
PRODUCT_BY_SKU: dict[str, Product] = {p.sku: p for p in PRODUCTS}

CITIES: tuple[tuple[str, str, str], ...] = (
    ("Bhopal", "Madhya Pradesh", "462001"),
    ("Indore", "Madhya Pradesh", "452001"),
    ("Mumbai", "Maharashtra", "400001"),
    ("Pune", "Maharashtra", "411001"),
    ("Bengaluru", "Karnataka", "560001"),
    ("Hyderabad", "Telangana", "500001"),
    ("Chennai", "Tamil Nadu", "600001"),
    ("Delhi", "Delhi", "110001"),
    ("Kolkata", "West Bengal", "700001"),
    ("Ahmedabad", "Gujarat", "380001"),
    ("Jaipur", "Rajasthan", "302001"),
    ("Lucknow", "Uttar Pradesh", "226001"),
)
STREETS = ("MG Road", "Station Road", "Park Street", "Gandhi Nagar", "Lake View Colony")
WAREHOUSES = ("Mumbai", "Bengaluru", "Delhi")
CARRIERS: tuple[tuple[str, str], ...] = (
    ("BlueDart", "BD"),
    ("Delhivery", "DL"),
    ("Ecom Express", "EE"),
    ("India Post", "IP"),
)
FIRST_NAMES = (
    "Aarav", "Vihaan", "Aditya", "Ishaan", "Kabir", "Arjun", "Rohan", "Karan", "Dev", "Neel",
    "Diya", "Ananya", "Aisha", "Meera", "Saanvi", "Kavya", "Riya", "Isha", "Tara", "Nisha",
)  # fmt: skip
LAST_NAMES = (
    "Sharma", "Patel", "Mehta", "Iyer", "Reddy", "Nair", "Gupta", "Singh", "Das", "Joshi",
    "Kulkarni", "Verma", "Menon", "Bose", "Chopra",
)  # fmt: skip

STATUS_WEIGHTS: tuple[tuple[OrderStatus, int], ...] = (
    (OrderStatus.PLACED, 8),
    (OrderStatus.CONFIRMED, 6),
    (OrderStatus.PACKED, 6),
    (OrderStatus.SHIPPED, 14),
    (OrderStatus.OUT_FOR_DELIVERY, 6),
    (OrderStatus.DELIVERED, 52),
    (OrderStatus.CANCELLED, 8),
)
TIER_WEIGHTS: tuple[tuple[CustomerTier, int], ...] = (
    (CustomerTier.STANDARD, 70),
    (CustomerTier.SILVER, 20),
    (CustomerTier.GOLD, 10),
)

# ------------------------------------------------------------------------ fixtures


@dataclass(frozen=True)
class FixtureCustomer:
    id: str
    name: str
    email: str
    tier: CustomerTier


@dataclass(frozen=True)
class FixtureOrder:
    """A hand-crafted order. Times are expressed as days before "now" (or after, for the ETA)."""

    id: str
    customer_id: str
    status: OrderStatus
    items: tuple[tuple[str, int], ...]  # (sku, quantity)
    placed_days_ago: float
    shipped_days_ago: float | None = None
    delivered_days_ago: float | None = None
    eta_days_from_now: float | None = None
    shipment_status: ShipmentStatus | None = None
    refunded_paise: int = 0
    payment: PaymentMethod = PaymentMethod.PREPAID
    note: str = ""


FIXTURE_CUSTOMERS: tuple[FixtureCustomer, ...] = (
    FixtureCustomer("C-0001", "Aarav Sharma", "aarav.sharma@example.com", CustomerTier.STANDARD),
    FixtureCustomer("C-0002", "Diya Patel", "diya.patel@example.com", CustomerTier.GOLD),
    FixtureCustomer("C-0003", "Rohan Mehta", "rohan.mehta@example.com", CustomerTier.STANDARD),
    FixtureCustomer("C-0004", "Ananya Iyer", "ananya.iyer@example.com", CustomerTier.SILVER),
)

FIXTURE_ORDERS: tuple[FixtureOrder, ...] = (
    FixtureOrder(
        "KRT-100001", "C-0001", OrderStatus.DELIVERED,
        items=(("SKU-EL-001", 1), ("SKU-EL-005", 1)),
        placed_days_ago=6, shipped_days_ago=5, delivered_days_ago=3,
        shipment_status=ShipmentStatus.DELIVERED,
        note="Recent delivery, high value: a full refund exceeds INR 1,000 and needs approval.",
    ),
    FixtureOrder(
        "KRT-100002", "C-0001", OrderStatus.SHIPPED,
        items=(("SKU-AP-003", 1),),
        placed_days_ago=3, shipped_days_ago=2, eta_days_from_now=2,
        shipment_status=ShipmentStatus.IN_TRANSIT,
        note="In transit and on schedule: good for order-status and tracking questions.",
    ),
    FixtureOrder(
        "KRT-100003", "C-0002", OrderStatus.PLACED,
        items=(("SKU-HM-002", 1), ("SKU-HM-004", 2)),
        placed_days_ago=0.5,
        note="Not shipped yet: cancellation and address change are allowed.",
    ),
    FixtureOrder(
        "KRT-100004", "C-0002", OrderStatus.DELIVERED,
        items=(("SKU-AP-002", 1),),
        placed_days_ago=50, shipped_days_ago=48, delivered_days_ago=45,
        shipment_status=ShipmentStatus.DELIVERED,
        note="Delivered 45 days ago: outside any normal return window.",
    ),
    FixtureOrder(
        "KRT-100005", "C-0003", OrderStatus.DELIVERED,
        items=(("SKU-BT-001", 1), ("SKU-AP-001", 2)),
        placed_days_ago=8, shipped_days_ago=7, delivered_days_ago=5,
        shipment_status=ShipmentStatus.DELIVERED,
        note="Contains a final-sale item that cannot be returned.",
    ),
    FixtureOrder(
        "KRT-100006", "C-0003", OrderStatus.SHIPPED,
        items=(("SKU-HM-003", 1),),
        placed_days_ago=9, shipped_days_ago=8, eta_days_from_now=-3,
        shipment_status=ShipmentStatus.EXCEPTION,
        note="Delayed: past its ETA with a carrier exception.",
    ),
    FixtureOrder(
        "KRT-100007", "C-0004", OrderStatus.DELIVERED,
        items=(("SKU-HM-004", 1),),
        placed_days_ago=13, shipped_days_ago=12, delivered_days_ago=10,
        shipment_status=ShipmentStatus.DELIVERED,
        note="INR 599 order: a refund is under the INR 1,000 auto-approval threshold.",
    ),
    FixtureOrder(
        "KRT-100008", "C-0004", OrderStatus.DELIVERED,
        items=(("SKU-EL-004", 1),),
        placed_days_ago=15, shipped_days_ago=14, delivered_days_ago=12,
        shipment_status=ShipmentStatus.DELIVERED,
        refunded_paise=150_000,
        note="Already partly refunded: INR 3,499 total, INR 1,500 refunded, INR 1,999 remaining.",
    ),
)  # fmt: skip
FIXTURE_ORDER_IDS: tuple[str, ...] = tuple(f.id for f in FIXTURE_ORDERS)

# ---------------------------------------------------------------------- generation


class SeedError(RuntimeError):
    """Raised when seeding cannot proceed (for example, the store already has data)."""


@dataclass(frozen=True)
class SeedResult:
    customers: int
    orders: int
    order_items: int
    shipments: int


@dataclass(frozen=True)
class Timeline:
    placed_at: datetime
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    eta: datetime | None = None
    shipment_status: ShipmentStatus | None = None


T = TypeVar("T")


def _weighted(rng: random.Random, options: tuple[tuple[T, int], ...]) -> T:
    return rng.choices([o for o, _ in options], weights=[w for _, w in options], k=1)[0]


def _address(rng: random.Random, name: str) -> dict[str, Any]:
    city, state, pincode = rng.choice(CITIES)
    return {
        "name": name,
        "line1": f"{rng.randint(1, 250)}, {rng.choice(STREETS)}",
        "city": city,
        "state": state,
        "pincode": pincode,
    }


def _events(
    *,
    status: ShipmentStatus,
    shipped_at: datetime,
    delivered_at: datetime | None,
    now: datetime,
    origin: str,
    destination: str,
) -> list[dict[str, Any]]:
    """Build a plausible tracking history; no event is ever later than "now"."""
    cap = now - timedelta(hours=1)

    def event(ts: datetime, location: str, status: str, **extra: str) -> dict[str, Any]:
        return {"ts": ts.isoformat(), "location": location, "status": status, **extra}

    events = [
        event(shipped_at - timedelta(hours=2), origin, "label_created"),
        event(shipped_at, origin, "picked_up"),
        event(min(shipped_at + timedelta(hours=12), cap), "Regional Hub", "in_transit"),
    ]
    if status is ShipmentStatus.EXCEPTION:
        exception_at = min(shipped_at + timedelta(days=2), cap)
        events.append(event(exception_at, "Regional Hub", "exception", detail="Delivery delayed"))
    if status in (ShipmentStatus.OUT_FOR_DELIVERY, ShipmentStatus.DELIVERED):
        out_at = delivered_at - timedelta(hours=6) if delivered_at else cap
        events.append(event(out_at, destination, "out_for_delivery"))
    if status is ShipmentStatus.DELIVERED and delivered_at is not None:
        events.append(event(delivered_at, destination, "delivered"))
    return events


def _random_timeline(status: OrderStatus, now: datetime, rng: random.Random) -> Timeline:
    def hours(lo: int, hi: int) -> timedelta:
        return timedelta(hours=rng.randint(lo, hi))

    if status in (OrderStatus.PLACED, OrderStatus.CONFIRMED, OrderStatus.PACKED):
        lo, hi = {
            OrderStatus.PLACED: (1, 30),
            OrderStatus.CONFIRMED: (6, 48),
            OrderStatus.PACKED: (12, 72),
        }[status]
        return Timeline(placed_at=now - hours(lo, hi))
    if status is OrderStatus.CANCELLED:
        return Timeline(placed_at=now - timedelta(days=rng.randint(1, 10)) - hours(0, 23))
    if status is OrderStatus.SHIPPED:
        shipped = now - timedelta(days=rng.randint(1, 4)) - hours(0, 12)
        exception = rng.random() < 0.15
        return Timeline(
            placed_at=shipped - hours(12, 36),
            shipped_at=shipped,
            eta=shipped + timedelta(days=rng.randint(3, 6)),
            shipment_status=ShipmentStatus.EXCEPTION if exception else ShipmentStatus.IN_TRANSIT,
        )
    if status is OrderStatus.OUT_FOR_DELIVERY:
        shipped = now - timedelta(days=rng.randint(2, 5)) - hours(0, 12)
        return Timeline(
            placed_at=shipped - hours(12, 36),
            shipped_at=shipped,
            eta=now + hours(1, 8),
            shipment_status=ShipmentStatus.OUT_FOR_DELIVERY,
        )
    delivered = now - timedelta(days=rng.randint(1, 55)) - hours(0, 23)
    shipped = delivered - timedelta(days=rng.randint(2, 5)) - hours(0, 12)
    return Timeline(
        placed_at=shipped - hours(12, 36),
        shipped_at=shipped,
        delivered_at=delivered,
        eta=shipped + timedelta(days=rng.randint(3, 6)),
        shipment_status=ShipmentStatus.DELIVERED,
    )


def _fixture_timeline(f: FixtureOrder, now: datetime) -> Timeline:
    def ago(days: float | None) -> datetime | None:
        return None if days is None else now - timedelta(days=days)

    placed = ago(f.placed_days_ago)
    assert placed is not None
    return Timeline(
        placed_at=placed,
        shipped_at=ago(f.shipped_days_ago),
        delivered_at=ago(f.delivered_days_ago),
        eta=None if f.eta_days_from_now is None else now + timedelta(days=f.eta_days_from_now),
        shipment_status=f.shipment_status,
    )


def _make_order(
    *,
    order_id: str,
    customer_id: str,
    address: dict[str, Any],
    status: OrderStatus,
    payment: PaymentMethod,
    lines: list[tuple[Product, int]],
    timeline: Timeline,
    refunded_paise: int,
    seq: int,
    now: datetime,
    rng: random.Random,
) -> Order:
    total = sum(p.price_paise * qty for p, qty in lines)
    if not 0 <= refunded_paise <= total:
        raise ValueError(f"{order_id}: refunded_paise {refunded_paise} not within 0..{total}")
    order = Order(
        id=order_id,
        customer_id=customer_id,
        status=status,
        payment_method=payment,
        placed_at=timeline.placed_at,
        delivered_at=timeline.delivered_at,
        shipping_address=address,
        currency="INR",
        total_paise=total,
        refunded_paise=refunded_paise,
    )
    order.items = [
        OrderItem(
            sku=p.sku,
            name=p.name,
            category=p.category,
            quantity=qty,
            unit_price_paise=p.price_paise,
            final_sale=p.final_sale,
        )
        for p, qty in lines
    ]
    if timeline.shipped_at is not None:
        if timeline.shipment_status is None:
            raise ValueError(f"{order_id}: shipped order needs a shipment status")
        carrier, code = rng.choice(CARRIERS)
        origin = rng.choice(WAREHOUSES)
        order.shipments = [
            Shipment(
                carrier=carrier,
                tracking_number=f"{code}{seq:07d}{rng.randint(100, 999)}",
                status=timeline.shipment_status,
                shipped_at=timeline.shipped_at,
                estimated_delivery=timeline.eta,
                delivered_at=timeline.delivered_at,
                events=_events(
                    status=timeline.shipment_status,
                    shipped_at=timeline.shipped_at,
                    delivered_at=timeline.delivered_at,
                    now=now,
                    origin=origin,
                    destination=address["city"],
                ),
            )
        ]
    return order


def seed_mock_store(
    session: Session,
    *,
    seed: int = 42,
    now: datetime | None = None,
    n_customers: int = 50,
    n_orders: int = 200,
) -> SeedResult:
    """Insert synthetic customers, orders, items and shipments. Does not commit."""
    if n_customers < len(FIXTURE_CUSTOMERS):
        raise ValueError(f"n_customers must be at least {len(FIXTURE_CUSTOMERS)}")
    if n_orders < len(FIXTURE_ORDERS):
        raise ValueError(f"n_orders must be at least {len(FIXTURE_ORDERS)}")
    if session.scalar(select(func.count()).select_from(Customer)):
        raise SeedError("the mock store already has customers; reset it first")

    now = (now or datetime.now(UTC)).replace(microsecond=0)
    rng = random.Random(seed)

    customers: list[Customer] = []
    addresses: dict[str, dict[str, Any]] = {}
    for fc in FIXTURE_CUSTOMERS:
        customers.append(
            Customer(
                id=fc.id,
                name=fc.name,
                email=fc.email,
                tier=fc.tier,
                created_at=now - timedelta(days=rng.randint(60, 400)),
            )
        )
        addresses[fc.id] = _address(rng, fc.name)
    for n in range(len(FIXTURE_CUSTOMERS) + 1, n_customers + 1):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        cid = f"C-{n:04d}"
        customers.append(
            Customer(
                id=cid,
                name=f"{first} {last}",
                email=f"{first}.{last}{n}@example.com".lower(),
                tier=_weighted(rng, TIER_WEIGHTS),
                created_at=now - timedelta(days=rng.randint(30, 700)),
            )
        )
        addresses[cid] = _address(rng, f"{first} {last}")

    orders: list[Order] = []
    seq = 0
    for f in FIXTURE_ORDERS:
        seq += 1
        orders.append(
            _make_order(
                order_id=f.id,
                customer_id=f.customer_id,
                address=addresses[f.customer_id],
                status=f.status,
                payment=f.payment,
                lines=[(PRODUCT_BY_SKU[sku], qty) for sku, qty in f.items],
                timeline=_fixture_timeline(f, now),
                refunded_paise=f.refunded_paise,
                seq=seq,
                now=now,
                rng=rng,
            )
        )

    first_random_number = 100_000 + len(FIXTURE_ORDERS) + 1
    for i in range(n_orders - len(FIXTURE_ORDERS)):
        seq += 1
        customer = rng.choice(customers)
        status = _weighted(rng, STATUS_WEIGHTS)
        payment = PaymentMethod.PREPAID if rng.random() < 0.7 else PaymentMethod.COD
        lines = [
            (rng.choice(PRODUCTS), 1 if rng.random() < 0.8 else 2)
            for _ in range(rng.choice((1, 1, 1, 2, 2, 3)))
        ]
        total = sum(p.price_paise * qty for p, qty in lines)
        refunded = 0
        if status is OrderStatus.CANCELLED and payment is PaymentMethod.PREPAID:
            refunded = total
        elif status is OrderStatus.DELIVERED and rng.random() < 0.08:
            refunded = (total * rng.randint(20, 60) // 100) // 100 * 100
        address = _address(rng, customer.name) if rng.random() < 0.1 else addresses[customer.id]
        orders.append(
            _make_order(
                order_id=f"KRT-{first_random_number + i}",
                customer_id=customer.id,
                address=address,
                status=status,
                payment=payment,
                lines=lines,
                timeline=_random_timeline(status, now, rng),
                refunded_paise=refunded,
                seq=seq,
                now=now,
                rng=rng,
            )
        )

    session.add_all(customers)
    session.add_all(orders)
    session.flush()
    return SeedResult(
        customers=len(customers),
        orders=len(orders),
        order_items=sum(len(o.items) for o in orders),
        shipments=sum(len(o.shipments) for o in orders),
    )


def reset_mock_store(session: Session) -> None:
    """Delete store and support data in dependency order. The append-only audit_log is untouched."""
    for model in (RunStep, Action, Run, Message, Ticket, Shipment, OrderItem, Order, Customer):
        session.execute(delete(model))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the Kartly mock store with synthetic data.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing store and support data first (audit_log is never touched)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--orders", type=int, default=200)
    args = parser.parse_args(argv)

    engine = make_engine()
    with make_session_factory(engine)() as session:
        if args.reset:
            reset_mock_store(session)
        try:
            result = seed_mock_store(
                session, seed=args.seed, n_customers=args.customers, n_orders=args.orders
            )
        except SeedError as exc:
            print(f"error: {exc} (re-run with --reset)", file=sys.stderr)
            return 1
        session.commit()

    print(
        f"Seeded {result.customers} customers, {result.orders} orders, "
        f"{result.order_items} items, {result.shipments} shipments."
    )
    print("Fixture orders for evals: " + ", ".join(FIXTURE_ORDER_IDS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
