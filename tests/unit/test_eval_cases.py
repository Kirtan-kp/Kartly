"""The case schema, the loader, and the golden set itself."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from kartly.evals import CaseFileError, load_cases
from kartly.evals.grader import mentions
from kartly.store.seed import FIXTURE_CUSTOMERS, FIXTURE_ORDERS
from tests.support import FIXED_NOW, make_ctx, make_sqlite_engine, seeded_session_factory

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden"


def minimal(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "a-case",
        "description": "d",
        "tier": 0,
        "intent": "tracking",
        "customer_id": "C-0001",
        "message": "hi",
        "expect": {"outcome": "drafted"},
    }
    return {**case, **overrides}


def write(tmp_path: Path, cases: Any, name: str = "cases.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cases), encoding="utf-8")
    return path


# ------------------------------------------------------------------------- the golden set


def test_the_golden_set_has_fourteen_valid_cases_with_unique_ids() -> None:
    cases = load_cases(GOLDEN)
    assert len(cases) == 14
    assert len({c.id for c in cases}) == 14


def test_tier_zero_cases_come_first_and_every_case_is_documented() -> None:
    cases = load_cases(GOLDEN)
    assert [c.tier for c in cases][:7] == [0] * 7
    assert all(c.description and c.notes for c in cases)


def test_cases_use_real_fixture_customers_and_orders() -> None:
    customers = {c.id for c in FIXTURE_CUSTOMERS}
    orders = {o.id for o in FIXTURE_ORDERS}
    for case in load_cases(GOLDEN):
        assert case.customer_id in customers, case.id
        for match in [*case.expect.calls, *case.expect.forbidden_calls]:
            order = match.args.get("order_id")
            if order is not None:
                assert order in orders or order == "KRT-999999", (case.id, order)


def test_the_orders_named_in_a_message_are_the_ones_the_case_expects() -> None:
    """A case that expects a lookup of KRT-100002 must actually ask about KRT-100002."""
    for case in load_cases(GOLDEN):
        for match in case.expect.calls:
            order = match.args.get("order_id")
            if order:
                assert order.lower() in case.message.lower(), case.id


def test_every_fixture_the_cases_rely_on_still_says_what_the_cases_assume() -> None:
    """The reply checks encode facts about the fixtures; fail here if the seed drifts."""
    from kartly.gateway import ToolGateway

    gateway = ToolGateway(seeded_session_factory(make_sqlite_engine()), clock=lambda: FIXED_NOW)

    def data(customer: str, tool: str, order: str) -> dict[str, Any]:
        result = gateway.call(make_ctx(customer), tool, {"order_id": order})
        assert result.ok and result.data is not None
        return result.data

    transit = data("C-0001", "get_tracking", "KRT-100002")["shipments"][0]
    assert transit["status"] == "in_transit" and transit["estimated_delivery"].startswith(
        "2026-10-25"
    )
    assert data("C-0001", "get_order", "KRT-100001")["total_inr"] == "2898.00"
    assert data("C-0003", "get_tracking", "KRT-100006")["shipments"][0]["delayed"] is True
    assert data("C-0002", "get_tracking", "KRT-100003")["shipped"] is False


def test_leak_phrases_are_real_facts_about_the_protected_order() -> None:
    """A leak check that names things the foreign order never contained would pass anything."""
    from kartly.gateway import ToolGateway

    factory = seeded_session_factory(make_sqlite_engine())
    gateway = ToolGateway(factory, clock=lambda: FIXED_NOW)
    protected = {  # case id -> (owner, order) whose details must not reach the reply
        "idor-friends-order": ("C-0001", "KRT-100001"),
        "injection-in-message": ("C-0002", "KRT-100003"),
    }
    names = {c.id: c.name for c in FIXTURE_CUSTOMERS}
    cases = {c.id: c for c in load_cases(GOLDEN)}
    for case_id, (owner, order) in protected.items():
        facts = " ".join(
            json.dumps(gateway.call(make_ctx(owner), tool, {"order_id": order}).data)
            for tool in ("get_order", "get_tracking")
        )
        facts += " " + names[owner]
        for phrase in cases[case_id].expect.answer.must_not_leak:
            # amounts are stored as "2898.00" but a reply may write "2,898"
            real = mentions(facts, phrase) or mentions(facts, phrase.replace(",", ""))
            assert real, f"{case_id}: {phrase!r} is not a fact about {order}"


# ------------------------------------------------------------------------- schema and loader


def test_a_minimal_case_loads(tmp_path: Path) -> None:
    [case] = load_cases(write(tmp_path, [minimal()]))
    assert case.expect.calls == [] and case.expect.security_events == [] and case.tags == []


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (minimal(typo_field=1), "typo_field"),
        (minimal(expect={"outcome": "drafted", "colour": "red"}), "colour"),
        (minimal(expect={"outcome": "maybe"}), "outcome"),
        (minimal(tier=4), "tier"),
        (minimal(intent="gossip"), "intent"),
        (minimal(id="Bad Id"), "lower-case"),
        (minimal(message=""), "message"),
        (minimal(expect={"outcome": "drafted", "max_tool_calls": -1}), "max_tool_calls"),
        (
            minimal(expect={"outcome": "drafted", "answer": {"must_mention": [[]]}}),
            "non-blank alternatives",
        ),
        (
            minimal(expect={"outcome": "drafted", "answer": {"must_mention": [[" "]]}}),
            "non-blank alternatives",
        ),
        (
            minimal(expect={"outcome": "drafted", "answer": {"must_not_leak": [""]}}),
            "blank phrases",
        ),
        (
            minimal(expect={"outcome": "drafted", "forbidden_calls": [{}]}),
            "must name a tool or an argument",
        ),
        (
            minimal(expect={"outcome": "escalated", "answer": {"must_mention": [["x"]]}}),
            "no reply",
        ),
    ],
)
def test_invalid_cases_are_rejected_with_the_case_id_and_the_reason(
    tmp_path: Path, case: dict[str, Any], message: str
) -> None:
    with pytest.raises(CaseFileError, match=message) as error:
        load_cases(write(tmp_path, [case]))
    assert "cases.yaml" in str(error.value)


def test_a_date_that_yaml_turns_into_a_date_object_is_caught(tmp_path: Path) -> None:
    path = tmp_path / "dates.yaml"
    path.write_text(
        "- id: x\n  description: d\n  tier: 0\n  intent: tracking\n  customer_id: C-0001\n"
        "  message: hi\n  expect:\n    outcome: drafted\n"
        "    answer: {must_mention: [[2026-10-25]]}\n",
        encoding="utf-8",
    )
    with pytest.raises(CaseFileError, match="must_mention"):
        load_cases(path)


def test_duplicate_ids_across_files_are_rejected(tmp_path: Path) -> None:
    write(tmp_path, [minimal()], "a.yaml")
    write(tmp_path, [minimal()], "b.yaml")
    with pytest.raises(CaseFileError, match="duplicate case ids: a-case"):
        load_cases(tmp_path)


@pytest.mark.parametrize("content", ["", "[]", "{}", "- just a string", ": : :"])
def test_empty_or_wrongly_shaped_files_are_rejected(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CaseFileError):
        load_cases(path)


def test_a_missing_file_or_an_empty_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(CaseFileError, match="cannot read"):
        load_cases(tmp_path / "nope.yaml")
    with pytest.raises(CaseFileError, match="no \\*.yaml"):
        load_cases(tmp_path)


def test_files_in_a_directory_load_in_name_order(tmp_path: Path) -> None:
    write(tmp_path, [minimal(id="second")], "20_b.yaml")
    write(tmp_path, [minimal(id="first")], "10_a.yaml")
    assert [c.id for c in load_cases(tmp_path)] == ["first", "second"]
