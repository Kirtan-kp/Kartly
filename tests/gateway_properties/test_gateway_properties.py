"""Property tests for the Tool Gateway's central guarantee.

    A customer can only ever read their own orders, whatever the model asks for.

The first test is exhaustive over the seeded store; the rest use Hypothesis to throw arbitrary and
hostile arguments at the gateway.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from kartly.gateway import ToolGateway, ToolResult
from kartly.store.models import Order
from tests.support import FIXED_NOW, make_ctx, make_sqlite_engine, seeded_session_factory

READ_TOOLS = ("get_order", "get_tracking")
CUSTOMERS = [f"C-{n:04d}" for n in range(1, 51)]
ORDER_IDS = [f"KRT-{100_000 + n}" for n in range(1, 201)]
PROPERTY = settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@dataclass
class World:
    gateway: ToolGateway
    owner_of: dict[str, str]  # order_id -> customer_id


@pytest.fixture(scope="module")
def world() -> Iterator[World]:
    engine = make_sqlite_engine()
    factory = seeded_session_factory(engine)
    with factory() as session:
        owner_of = {o.id: o.customer_id for o in session.scalars(select(Order))}
    yield World(ToolGateway(factory, clock=lambda: FIXED_NOW), owner_of)
    engine.dispose()


def test_seed_matches_the_ids_these_tests_assume(world: World) -> None:
    assert sorted(world.owner_of) == ORDER_IDS
    assert set(world.owner_of.values()) <= set(CUSTOMERS)


def test_only_the_owner_can_read_an_order(world: World) -> None:
    """Every order, both tools: the owner succeeds, other customers get nothing."""
    for order_id, owner in sorted(world.owner_of.items()):
        position = CUSTOMERS.index(owner)
        strangers = [CUSTOMERS[(position + step) % len(CUSTOMERS)] for step in (1, 7, 23)]
        for tool in READ_TOOLS:
            mine = world.gateway.call(make_ctx(owner), tool, {"order_id": order_id})
            assert mine.ok and mine.data is not None and mine.data["order_id"] == order_id
            for stranger in strangers:
                theirs = world.gateway.call(make_ctx(stranger), tool, {"order_id": order_id})
                assert not theirs.ok, (tool, order_id, stranger)
                assert theirs.data is None
                assert theirs.error_code == "order_not_found"
                assert "total_inr" not in theirs.to_model_content()


# ----------------------------------------------------------- hostile arguments

json_scalars = st.none() | st.booleans() | st.integers() | st.floats() | st.text(max_size=60)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=12), children, max_size=3)
    ),
    max_leaves=8,
)
any_args = json_values | st.dictionaries(
    st.sampled_from(["order_id", "customer_id", "note"]) | st.text(max_size=10),
    json_values,
    max_size=4,
)


@PROPERTY
@given(st.sampled_from(CUSTOMERS), st.sampled_from(READ_TOOLS), any_args)
def test_arbitrary_arguments_never_crash_or_leak(
    world: World, customer: str, tool: str, args: Any
) -> None:
    result = world.gateway.call(make_ctx(customer), tool, args)
    assert isinstance(result, ToolResult)
    json.loads(result.to_model_content())  # always valid JSON for the model
    if result.ok:
        assert result.data is not None
        assert world.owner_of[result.data["order_id"]] == customer


# (text sent, does it normalise to the real order ID?)
VARIANTS = (
    (lambda i: i, True),
    (lambda i: i.lower(), True),
    (lambda i: f"  {i}  ", True),
    (lambda i: i + "\n", True),
    (lambda i: i + "; DROP TABLE orders", False),
    (lambda i: i[:-1], False),  # one digit short
    (lambda i: i + "0", False),  # a different (nonexistent) order
    (lambda i: i.replace("-", ""), False),
)


@PROPERTY
@given(
    st.sampled_from(CUSTOMERS),
    st.sampled_from(ORDER_IDS),
    st.sampled_from(range(len(VARIANTS))),
    st.sampled_from(READ_TOOLS),
)
def test_variants_of_order_ids_follow_the_ownership_rule(
    world: World, customer: str, order_id: str, variant: int, tool: str
) -> None:
    make_text, normalises = VARIANTS[variant]
    result = world.gateway.call(make_ctx(customer), tool, {"order_id": make_text(order_id)})
    expected_ok = normalises and world.owner_of[order_id] == customer
    assert result.ok == expected_ok


@PROPERTY
@given(
    st.sampled_from(CUSTOMERS),
    st.sampled_from(ORDER_IDS),
    st.dictionaries(st.text(min_size=1, max_size=10), json_values, min_size=1, max_size=3),
)
def test_extra_arguments_are_always_rejected(
    world: World, customer: str, order_id: str, extras: dict[str, Any]
) -> None:
    """Unknown arguments, a customer_id above all, can never be smuggled in next to a valid one."""
    extras = {k: v for k, v in extras.items() if k != "order_id"}
    if not extras:
        return
    result = world.gateway.call(make_ctx(customer), "get_order", {"order_id": order_id, **extras})
    assert not result.ok
    assert result.error_code == "invalid_arguments"
