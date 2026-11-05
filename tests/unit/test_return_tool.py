"""check_return_eligibility: the verdict is computed in code from the order, tier and window."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.store.models import AuditLog, Customer, CustomerTier, Order, OrderItem, OrderStatus
from kartly.tools import RiskTier, default_tools
from kartly.tools.returns import (
    GOLD_WINDOW_DAYS,
    RETURN_TOOLS,
    RETURN_WINDOW_DAYS,
    window_days,
)
from tests.support import FIXED_NOW, make_ctx

ROOT = Path(__file__).resolve().parents[2]


def gateway_at(factory: sessionmaker[Session], days_later: int = 0) -> ToolGateway:
    now = FIXED_NOW + timedelta(days=days_later)
    return ToolGateway(factory, [*default_tools(), *RETURN_TOOLS], clock=lambda: now)


def check(gateway: ToolGateway, customer: str, order_id: str) -> dict[str, Any]:
    result = gateway.call(make_ctx(customer), "check_return_eligibility", {"order_id": order_id})
    assert result.ok, result.message
    assert result.data is not None
    return result.data


def test_a_recent_delivery_is_eligible_with_the_standard_window(
    seeded_factory: sessionmaker[Session],
) -> None:
    data = check(gateway_at(seeded_factory), "C-0001", "KRT-100001")  # delivered 3 days ago
    assert data["eligible"] is True and data["reason"] is None
    assert (data["membership"], data["window_days"]) == ("standard", RETURN_WINDOW_DAYS)
    assert data["days_since_delivery"] == 3
    assert data["delivered_on"] == "2026-10-20" and data["window_ends_on"] == "2026-10-27"
    assert [(i["sku"], i["returnable"], i["reason"]) for i in data["items"]] == [
        ("SKU-EL-001", True, None),
        ("SKU-EL-005", True, None),
    ]


def test_an_old_delivery_is_outside_the_window_and_gold_gets_ten_days(
    seeded_factory: sessionmaker[Session],
) -> None:
    data = check(gateway_at(seeded_factory), "C-0002", "KRT-100004")  # 45 days, a Gold member
    assert data["eligible"] is False and data["reason"] == "outside_window"
    assert (data["membership"], data["window_days"]) == ("gold", GOLD_WINDOW_DAYS)
    assert data["days_since_delivery"] == 45 and data["window_ends_on"] == "2026-09-18"
    assert [(i["returnable"], i["reason"]) for i in data["items"]] == [(False, "outside_window")]


@pytest.mark.parametrize(
    ("customer", "order", "delivered_days_ago", "window"),
    [("C-0001", "KRT-100001", 3, 7), ("C-0002", "KRT-100004", 45, 10)],
)
def test_the_last_day_of_the_window_counts_and_the_next_day_does_not(
    seeded_factory: sessionmaker[Session],
    customer: str,
    order: str,
    delivered_days_ago: int,
    window: int,
) -> None:
    last = check(gateway_at(seeded_factory, window - delivered_days_ago), customer, order)
    assert last["days_since_delivery"] == window and last["eligible"] is True
    after = check(gateway_at(seeded_factory, window - delivered_days_ago + 1), customer, order)
    assert after["days_since_delivery"] == window + 1 and after["eligible"] is False


def test_final_sale_items_are_listed_but_do_not_make_the_whole_order_ineligible(
    seeded_factory: sessionmaker[Session],
) -> None:
    data = check(gateway_at(seeded_factory), "C-0003", "KRT-100005")  # serum (final sale) + 2 tees
    assert data["eligible"] is True
    by_sku = {i["sku"]: i for i in data["items"]}
    assert (
        by_sku["SKU-BT-001"]["returnable"] is False
        and by_sku["SKU-BT-001"]["reason"] == "final_sale"
    )
    assert by_sku["SKU-AP-001"]["returnable"] is True


def test_an_order_of_only_final_sale_items_is_not_eligible(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        session.execute(
            update(OrderItem).where(OrderItem.order_id == "KRT-100005").values(final_sale=True)
        )
        session.commit()
    data = check(gateway_at(seeded_factory), "C-0003", "KRT-100005")
    assert data["eligible"] is False and data["reason"] == "all_items_final_sale"


def test_final_sale_beats_outside_window_as_the_reason_for_an_item(
    seeded_factory: sessionmaker[Session],
) -> None:
    data = check(gateway_at(seeded_factory, 30), "C-0003", "KRT-100005")
    reasons = {i["sku"]: i["reason"] for i in data["items"]}
    assert reasons == {"SKU-BT-001": "final_sale", "SKU-AP-001": "outside_window"}
    assert data["eligible"] is False and data["reason"] == "outside_window"


@pytest.mark.parametrize(
    ("customer", "order", "status"),
    [("C-0002", "KRT-100003", "placed"), ("C-0001", "KRT-100002", "shipped")],
)
def test_an_order_that_has_not_been_delivered_cannot_be_returned_yet(
    seeded_factory: sessionmaker[Session], customer: str, order: str, status: str
) -> None:
    data = check(gateway_at(seeded_factory), customer, order)
    assert data["eligible"] is False and data["reason"] == "not_delivered"
    assert data["order_status"] == status
    assert data["delivered_on"] is None and data["days_since_delivery"] is None
    assert all(i["reason"] == "not_delivered" and not i["returnable"] for i in data["items"])


def test_a_delivered_status_without_a_delivery_time_is_treated_as_not_delivered(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        session.execute(update(Order).where(Order.id == "KRT-100001").values(delivered_at=None))
        session.commit()
    assert check(gateway_at(seeded_factory), "C-0001", "KRT-100001")["reason"] == "not_delivered"


def test_another_customers_order_is_not_found_and_audited(
    seeded_factory: sessionmaker[Session],
) -> None:
    result = gateway_at(seeded_factory).call(
        make_ctx("C-0002"), "check_return_eligibility", {"order_id": "KRT-100001"}
    )
    assert not result.ok and result.error_code == "order_not_found"
    with seeded_factory() as session:
        row = session.scalars(select(AuditLog)).one()
        assert (
            row.details is not None and row.details.get("security_event") == "cross_customer_access"
        )


def test_the_tool_is_read_only_and_takes_no_identity() -> None:
    [tool] = RETURN_TOOLS
    assert tool.tier is RiskTier.READ_ONLY and tool.name == "check_return_eligibility"
    assert "customer_id" not in tool.args_model.model_json_schema()["properties"]


def test_the_membership_tiers_map_to_the_windows() -> None:
    assert window_days(CustomerTier.GOLD) == GOLD_WINDOW_DAYS == 10
    assert window_days(CustomerTier.SILVER) == window_days(CustomerTier.STANDARD) == 7


def test_a_standard_member_who_becomes_gold_gets_the_longer_window(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        session.execute(
            update(Customer).where(Customer.id == "C-0001").values(tier=CustomerTier.GOLD)
        )
        session.commit()
    data = check(gateway_at(seeded_factory), "C-0001", "KRT-100001")
    assert data["window_days"] == 10 and data["window_ends_on"] == "2026-10-30"


def test_the_constants_match_the_knowledge_base_text() -> None:
    text = (ROOT / "kb" / "returns.md").read_text(encoding="utf-8")
    window = re.search(
        r"returned within (\d+) days of delivery\. Gold members get (\d+) days", text
    )
    assert window is not None
    assert (int(window[1]), int(window[2])) == (RETURN_WINDOW_DAYS, GOLD_WINDOW_DAYS)
    assert "Final sale" in text  # the one exclusion this tool decides on
    assert OrderStatus.DELIVERED.value == "delivered"


def test_a_reply_quoting_the_tools_day_count_passes_the_case_and_the_grounding_guard() -> None:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_case
    from kartly.retrieval import Chunk, Hit
    from kartly.tools.policy import policy_tool
    from tests.llm_support import ScriptedClient, make_response, tool_call

    chunk = Chunk(id="returns#return-window", doc="returns", title="R", heading="W", text="7 days.")

    class One:
        name = "one"

        def search(self, query: str, k: int = 5) -> list[Hit]:
            return [Hit(chunk, 0.9)]

    case = next(
        c
        for c in load_cases(ROOT / "evals" / "golden_policy")
        if c.id == "policy-order-outside-window"
    )
    script = [
        make_response(None, [tool_call("search_policy", {"question": "return window"})]),
        make_response(None, [tool_call("check_return_eligibility", {"order_id": "KRT-100004"})]),
        make_response(
            "Order KRT-100004 was delivered 45 days ago, outside your 10 days return window, so "
            "it can't be returned. [returns#return-window]"
        ),
    ]
    tools = [*default_tools(), *RETURN_TOOLS, policy_tool(One(), min_score=0.5)]
    result = run_case(case, ScriptedClient(script), model_name="m", tools=tools)
    assert result.status == "pass", result.detail
