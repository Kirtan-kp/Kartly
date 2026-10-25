"""Read-only order tools (risk tier 0): ``get_order`` and ``get_tracking``.

What the model sees is deliberately small and pre-computed:
- amounts are rupee strings ("2898.00"), so the model never does paise arithmetic;
- dates are ISO 8601 UTC, and "delayed" is computed here, not guessed by the model;
- the shipping address is reduced to city, state, and pincode (no name or street line);
- lists are capped so one large order cannot flood the context window.

Carrier event text is third-party data and therefore untrusted. It is length-limited here, and
the context builder (Week 3) will also wrap all tool output as data, never as instructions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.base import ExecutableOption

from kartly.sanitize import clean_str
from kartly.store.models import Order, Shipment, ShipmentStatus
from kartly.tools.base import RiskTier, Tool, ToolEnv, ToolError

MAX_ITEMS = 20
MAX_SHIPMENTS = 5
MAX_EVENTS = 5

OrderId = Annotated[
    str, Field(pattern=r"^KRT-\d{6,10}$", description="Order ID, for example KRT-100001")
]


class OrderArgs(BaseModel):
    """Arguments for any tool that works on one order. Note: there is no customer field."""

    model_config = ConfigDict(extra="forbid")

    order_id: OrderId

    @field_validator("order_id", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


def inr(paise: int) -> str:
    """Format integer paise as rupees with two decimals: 289800 -> '2898.00'."""
    rupees, remainder = divmod(abs(paise), 100)
    return f"{'-' if paise < 0 else ''}{rupees}.{remainder:02d}"


def utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes; treat them as UTC."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def iso(value: datetime | None) -> str | None:
    return None if value is None else utc(value).isoformat()


def _load_owned_order(env: ToolEnv, order_id: str, *options: ExecutableOption) -> Order:
    """Load an order only if it belongs to the caller.

    The ownership test is part of the query, so another customer's order is never loaded on the
    normal path. A foreign order and a nonexistent order produce the same error and message, so
    the model (and anyone steering it) cannot tell which order IDs exist. Only the audit log
    records the difference.
    """
    order = env.session.scalars(
        select(Order)
        .where(Order.id == order_id, Order.customer_id == env.ctx.customer_id)
        .options(*options)
    ).first()
    if order is not None:
        return order
    exists = env.session.scalar(select(Order.id).where(Order.id == order_id)) is not None
    raise ToolError(
        "order_not_found",
        f"No order {order_id} was found for this customer.",
        reason="not_owner" if exists else "no_such_order",
        security_event="cross_customer_access" if exists else None,
    )


def get_order(env: ToolEnv, args: OrderArgs) -> dict[str, Any]:
    order = _load_owned_order(
        env, args.order_id, selectinload(Order.items), selectinload(Order.shipments)
    )
    items = sorted(order.items, key=lambda item: item.id)
    address = order.shipping_address
    return {
        "order_id": order.id,
        "status": order.status.value,
        "placed_at": iso(order.placed_at),
        "delivered_at": iso(order.delivered_at),
        "payment_method": order.payment_method.value,
        "total_inr": inr(order.total_paise),
        "refunded_inr": inr(order.refunded_paise),
        "ship_to": {
            "city": address.get("city"),
            "state": address.get("state"),
            "pincode": address.get("pincode"),
        },
        "items": [
            {
                "sku": item.sku,
                "name": item.name,
                "category": item.category,
                "quantity": item.quantity,
                "unit_price_inr": inr(item.unit_price_paise),
                "final_sale": item.final_sale,
            }
            for item in items[:MAX_ITEMS]
        ],
        "items_truncated": len(items) > MAX_ITEMS,
        "has_shipment": bool(order.shipments),
    }


def _event_view(event: dict[str, Any]) -> dict[str, str]:
    return {
        key: clean_str(str(event[key]), 120)
        for key in ("ts", "location", "status", "detail")
        if event.get(key) is not None
    }


def _shipment_view(shipment: Shipment, now: datetime) -> dict[str, Any]:
    eta = utc(shipment.estimated_delivery) if shipment.estimated_delivery else None
    delayed = shipment.status is not ShipmentStatus.DELIVERED and eta is not None and eta < now
    events = sorted(shipment.events, key=lambda event: str(event.get("ts", "")))
    return {
        "carrier": shipment.carrier,
        "tracking_number": shipment.tracking_number,
        "status": shipment.status.value,
        "shipped_at": iso(shipment.shipped_at),
        "estimated_delivery": iso(shipment.estimated_delivery),
        "delivered_at": iso(shipment.delivered_at),
        "delayed": delayed,
        "recent_events": [_event_view(event) for event in events[-MAX_EVENTS:]],
    }


def get_tracking(env: ToolEnv, args: OrderArgs) -> dict[str, Any]:
    order = _load_owned_order(env, args.order_id, selectinload(Order.shipments))
    shipments = sorted(order.shipments, key=lambda s: (utc(s.shipped_at), s.id))
    return {
        "order_id": order.id,
        "order_status": order.status.value,
        "shipped": bool(shipments),
        "shipments": [_shipment_view(s, env.now) for s in shipments[:MAX_SHIPMENTS]],
    }


GET_ORDER = Tool(
    name="get_order",
    tier=RiskTier.READ_ONLY,
    description=(
        "Look up one of this customer's orders: status, items, amounts, placed and delivered "
        "dates, and destination city. Use it for any question about what an order contains or "
        "where it stands."
    ),
    args_model=OrderArgs,
    handler=get_order,
)

GET_TRACKING = Tool(
    name="get_tracking",
    tier=RiskTier.READ_ONLY,
    description=(
        "Get shipment tracking for one of this customer's orders: carrier, status, estimated "
        "delivery date, whether it is delayed, and recent scan events."
    ),
    args_model=OrderArgs,
    handler=get_tracking,
)

ORDER_TOOLS: tuple[Tool, ...] = (GET_ORDER, GET_TRACKING)
