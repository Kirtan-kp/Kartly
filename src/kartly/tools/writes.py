"""Write tools. They change the store, so they only ever run through the Tool Gateway's write path.

``cancel_order`` is risk tier 1. The gateway decides whether a call runs now or becomes a proposal
for a human (the tier's mode), and gives every call an idempotency key. The tool decides only
whether the cancellation is allowed, and does it. The rule matches kb/cancellation.md: an order
can be cancelled until it is packed.
"""

from __future__ import annotations

from typing import Any

from kartly.store.models import Order, OrderStatus, PaymentMethod
from kartly.tools.base import RiskTier, Tool, ToolEnv, ToolError
from kartly.tools.orders import OrderArgs, _load_owned_order, inr

# kb/cancellation.md: "Orders show 'Packed' or 'Shipped' once they are no longer cancellable."
CANCELLABLE = (OrderStatus.PLACED, OrderStatus.CONFIRMED)


def _cancellable_order(env: ToolEnv, args: OrderArgs) -> Order:
    order = _load_owned_order(env, args.order_id)
    if order.status == OrderStatus.CANCELLED:
        raise ToolError(
            "already_cancelled",
            "This order is already cancelled.",
            reason="already_cancelled",
        )
    if order.status not in CANCELLABLE:
        raise ToolError(
            "not_cancellable",
            f"This order is '{order.status.value}', so it can no longer be cancelled. The "
            "customer can refuse the delivery or return it after it arrives.",
            reason=f"status_{order.status.value}",
        )
    return order


def _prepaid_amount(order: Order) -> str | None:
    if order.payment_method == PaymentMethod.COD:
        return None  # nothing was paid, so there is nothing to give back
    return inr(order.total_paise - order.refunded_paise)


def preview_cancel(env: ToolEnv, args: OrderArgs) -> dict[str, Any]:
    order = _cancellable_order(env, args)
    return {
        "action": "cancel_order",
        "order_id": order.id,
        "current_status": order.status.value,
        "items": len(order.items),
        "payment_method": order.payment_method.value,
        "refund_due_inr": _prepaid_amount(order),
    }


def cancel_order(env: ToolEnv, args: OrderArgs) -> dict[str, Any]:
    order = _cancellable_order(env, args)
    order.status = OrderStatus.CANCELLED
    env.session.flush()
    return {
        "order_id": order.id,
        "status": "cancelled",
        "payment_method": order.payment_method.value,
        "refund_due_inr": _prepaid_amount(order),
        "note": (
            "The order is cancelled. Any refund is made separately; do not promise a date "
            "unless the policy text gives one."
        ),
    }


CANCEL_ORDER = Tool(
    name="cancel_order",
    tier=RiskTier.LOW_RISK_WRITE,
    description=(
        "Cancel one of this customer's orders that has not been packed or shipped yet. Use it "
        "only when the customer clearly asks to cancel a whole order. It cannot cancel part of "
        "an order. The result may say the cancellation is waiting for a human to approve it; "
        "then it has not happened yet."
    ),
    args_model=OrderArgs,
    handler=cancel_order,
    preview=preview_cancel,
)

WRITE_TOOLS: tuple[Tool, ...] = (CANCEL_ORDER,)
