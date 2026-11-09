"""Write tools. They change the store, so they only ever run through the Tool Gateway's write path.

``cancel_order`` is risk tier 1. The gateway decides whether a call runs now or becomes a proposal
for a human (the tier's mode), and gives every call an idempotency key. The tool decides only
whether the cancellation is allowed, and does it. The rule matches kb/cancellation.md: an order
can be cancelled until it is packed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import Field, field_validator
from sqlalchemy import select

from kartly.store.models import Action, ActionStatus, Order, OrderStatus, PaymentMethod, Ticket
from kartly.tools.base import RiskTier, Tool, ToolEnv, ToolError
from kartly.tools.orders import OrderArgs, _load_owned_order, inr

# kb/cancellation.md: "Orders show 'Packed' or 'Shipped' once they are no longer cancellable."
CANCELLABLE = (OrderStatus.PLACED, OrderStatus.CONFIRMED)


def _cancellable_order(env: ToolEnv, args: OrderArgs, lock: bool = False) -> Order:
    order = _load_owned_order(env, args.order_id, lock=lock)
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
    order = _cancellable_order(env, args, lock=True)
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


# --------------------------------------------------------------------------------- issue_refund

DEFAULT_AUTO_LIMIT_INR = 1_000  # kb/refunds.md: "Refunds up to INR 1,000 on an order ..."
DEFAULT_DAILY_CAP_INR = 5_000  # per customer, over the last 24 hours
REFUNDABLE = (OrderStatus.DELIVERED, OrderStatus.CANCELLED)


class RefundArgs(OrderArgs):
    """Order, amount in rupees (at most two decimals) and why. There is no customer field."""

    amount_inr: float = Field(gt=0, le=1_000_000)
    reason_code: Literal[
        "damaged", "defective", "wrong_item", "return_received", "order_cancelled", "other"
    ]

    @field_validator("amount_inr")
    @classmethod
    def _whole_paise(cls, value: float) -> float:
        if abs(value * 100 - round(value * 100)) > 1e-6:
            raise ValueError("amount_inr can have at most two decimal places")
        return value

    @property
    def amount_paise(self) -> int:
        return round(self.amount_inr * 100)


def _refunded_last_day(env: ToolEnv, customer_id: str) -> int:
    """Paise this customer was refunded in the 24 hours up to now, by refunds that ran."""
    since = env.now - timedelta(days=1)
    rows = env.session.scalars(
        select(Action.result)
        .join(Ticket, Action.ticket_id == Ticket.id)
        .where(
            Ticket.customer_id == customer_id,
            Action.tool_name == "issue_refund",
            Action.status == ActionStatus.SUCCEEDED,
        )
    )
    total = 0
    for result in rows:
        if not isinstance(result, dict):
            continue
        at = result.get("refunded_at")
        if isinstance(at, str) and datetime.fromisoformat(at) >= since:
            total += int(result.get("refunded_paise", 0))
    return total


def make_refund_tool(
    auto_limit_inr: int = DEFAULT_AUTO_LIMIT_INR, daily_cap_inr: int = DEFAULT_DAILY_CAP_INR
) -> Tool:
    """``issue_refund`` with its two money limits (both settings; see ADR-047)."""
    auto_limit = auto_limit_inr * 100
    daily_cap = daily_cap_inr * 100

    def check(env: ToolEnv, args: RefundArgs, lock: bool = False) -> Order:
        order = _load_owned_order(env, args.order_id, lock=lock)
        if order.status not in REFUNDABLE:
            raise ToolError(
                "not_refundable",
                f"This order is '{order.status.value}'. A refund is possible once it is delivered. "
                "Before that the customer can cancel it, or refuse the delivery.",
                reason=f"status_{order.status.value}",
            )
        if order.status == OrderStatus.CANCELLED and order.payment_method == PaymentMethod.COD:
            raise ToolError(
                "nothing_to_refund",
                "This cash on delivery order was cancelled before any payment was taken.",
                reason="cod_cancelled",
            )
        remaining = order.total_paise - order.refunded_paise
        if args.amount_paise > remaining:
            raise ToolError(
                "exceeds_refundable",
                f"Only INR {inr(remaining)} of this order can still be refunded.",
                reason="exceeds_refundable",
            )
        return order

    def preview(env: ToolEnv, args: RefundArgs) -> dict[str, Any]:
        order = check(env, args)
        return {
            "action": "issue_refund",
            "order_id": order.id,
            "amount_inr": inr(args.amount_paise),
            "reason_code": args.reason_code,
            "refunded_so_far_inr": inr(order.refunded_paise),
            "refundable_after_inr": inr(
                order.total_paise - order.refunded_paise - args.amount_paise
            ),
        }

    def needs_approval(env: ToolEnv, args: RefundArgs) -> str | None:
        order = _load_owned_order(env, args.order_id)
        # The limit is per order and counts earlier refunds, so splitting one large refund into
        # several small ones does not get around it.
        if order.refunded_paise + args.amount_paise > auto_limit:
            return "over_auto_limit"
        if _refunded_last_day(env, env.ctx.customer_id) + args.amount_paise > daily_cap:
            return "daily_cap"
        return None

    def issue(env: ToolEnv, args: RefundArgs) -> dict[str, Any]:
        order = check(env, args, lock=True)
        order.refunded_paise += args.amount_paise
        env.session.flush()
        return {
            "order_id": order.id,
            "refunded_inr": inr(args.amount_paise),
            "refunded_paise": args.amount_paise,
            "refunded_at": env.now.isoformat(),
            "total_refunded_inr": inr(order.refunded_paise),
            "remaining_refundable_inr": inr(order.total_paise - order.refunded_paise),
            "payment_method": order.payment_method.value,
            "note": (
                "The refund is recorded. Do not promise a date or a destination unless the "
                "policy text gives one."
            ),
        }

    return Tool(
        name="issue_refund",
        tier=RiskTier.MONEY,
        description=(
            "Refund part or all of what a customer paid on one of their delivered or cancelled "
            "orders. Use it only after the policy and the order's facts say the customer is "
            "owed this money (for example check_return_eligibility, or damaged goods). Give the "
            "amount in rupees. Refunds above the limit wait for a human to approve them; the "
            "result then says it has not happened yet."
        ),
        args_model=RefundArgs,
        handler=issue,
        preview=preview,
        needs_approval=needs_approval,
    )


def write_tools(
    auto_limit_inr: int = DEFAULT_AUTO_LIMIT_INR, daily_cap_inr: int = DEFAULT_DAILY_CAP_INR
) -> tuple[Tool, ...]:
    return (CANCEL_ORDER, make_refund_tool(auto_limit_inr, daily_cap_inr))


WRITE_TOOLS: tuple[Tool, ...] = write_tools()
