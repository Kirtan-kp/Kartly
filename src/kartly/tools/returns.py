"""``check_return_eligibility`` (risk tier 0): can this order, or which of its items, be returned?

The model is bad at date arithmetic and at combining an order's facts with a policy, so the
verdict is computed here, from the order, the customer's membership tier and the fixed return
rules, and the model only explains it. The numbers below mirror ``kb/returns.md``; a test fails
if the knowledge base text and these constants disagree.

It reads only; it does not start a return. Items marked final sale are never returnable; the
other exclusions in the policy (opened personal care, innerwear, gift cards, perishables) depend
on facts the store does not record, so they are left to the policy text and a human.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy.orm import selectinload

from kartly.store.models import CustomerTier, Order, OrderStatus
from kartly.tools.base import RiskTier, Tool, ToolEnv
from kartly.tools.orders import OrderArgs, _load_owned_order, utc

RETURN_WINDOW_DAYS = 7
GOLD_WINDOW_DAYS = 10
MAX_ITEMS = 20


def window_days(tier: CustomerTier) -> int:
    return GOLD_WINDOW_DAYS if tier is CustomerTier.GOLD else RETURN_WINDOW_DAYS


def check_return_eligibility(env: ToolEnv, args: OrderArgs) -> dict[str, Any]:
    order = _load_owned_order(env, args.order_id, selectinload(Order.items))
    tier = order.customer.tier
    window = window_days(tier)
    base: dict[str, Any] = {
        "order_id": order.id,
        "order_status": order.status.value,
        "membership": tier.value,
        "window_days": window,
    }
    items = sorted(order.items, key=lambda item: item.id)[:MAX_ITEMS]

    if order.status is not OrderStatus.DELIVERED or order.delivered_at is None:
        return {
            **base,
            "delivered_on": None,
            "days_since_delivery": None,
            "window_ends_on": None,
            "eligible": False,
            "reason": "not_delivered",
            "items": [
                {"sku": i.sku, "name": i.name, "returnable": False, "reason": "not_delivered"}
                for i in items
            ],
        }

    delivered = utc(order.delivered_at).date()
    days = (env.now.date() - delivered).days
    ends = delivered + timedelta(days=window)
    within = days <= window

    def item_view(sku: str, name: str, final_sale: bool) -> dict[str, Any]:
        reason = "final_sale" if final_sale else None if within else "outside_window"
        return {"sku": sku, "name": name, "returnable": reason is None, "reason": reason}

    views = [item_view(i.sku, i.name, i.final_sale) for i in items]
    eligible = any(v["returnable"] for v in views)
    reason = None if eligible else ("outside_window" if not within else "all_items_final_sale")
    return {
        **base,
        "delivered_on": delivered.isoformat(),
        "days_since_delivery": days,
        "window_ends_on": ends.isoformat(),
        "eligible": eligible,
        "reason": reason,
        "items": views,
    }


RETURN_TOOLS: tuple[Tool, ...] = (
    Tool(
        name="check_return_eligibility",
        tier=RiskTier.READ_ONLY,
        description=(
            "Check whether the customer's order can be returned now: the return window for their "
            "membership, days since delivery, and which items cannot be returned and why. Use it "
            "whenever the customer asks if they can still return, exchange or get a refund for "
            "an order they named. It does not start a return."
        ),
        args_model=OrderArgs,
        handler=check_return_eligibility,
    ),
)
