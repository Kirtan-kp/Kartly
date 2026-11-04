"""The grounding guard: pure checks, then through the ticket runner."""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.guards import check_grounding
from kartly.guards.grounding import _normal
from kartly.orchestrator import TicketRunner, format_trace, load_trace
from kartly.retrieval import Chunk, Hit
from kartly.store.models import (
    ActorType,
    AuditLog,
    Message,
    MessageRole,
    Run,
    RunStatus,
    Ticket,
    TicketStatus,
)
from kartly.tools import default_tools
from kartly.tools.policy import policy_tool
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW


def search_result(*chunks: tuple[str, str], found: bool = True) -> dict[str, Any]:
    results = [
        {"chunk_id": cid, "title": "T", "section": "S", "text": text} for cid, text in chunks
    ]
    data = {"found": found, "results": results}
    return {"role": "tool", "content": json.dumps({"ok": True, "data": data})}


RETURNS = search_result(("returns#return-window", "Most items can be returned within 7 days."))


# ------------------------------------------------------------------------------ pure checks


def test_a_cited_reply_with_supported_numbers_passes_and_loses_its_citation() -> None:
    r = check_grounding("You have 7 days to return it. [returns#return-window]", [RETURNS])
    assert r.ok and r.applicable
    assert r.citations == ("returns#return-window",)
    assert r.text == "You have 7 days to return it."


def test_a_citation_that_no_search_returned_is_a_problem() -> None:
    r = check_grounding("Try this. [search_policy#price-matching]", [RETURNS])
    assert not r.ok
    assert r.problems == (
        "cites [search_policy#price-matching], which no policy search returned in this run",
    )


def test_a_citation_without_any_search_is_a_problem() -> None:
    r = check_grounding("The window is 7 days. [returns#return-window]", [])
    assert r.applicable and not r.ok  # a model that never looked cannot cite


def test_a_number_that_no_tool_or_customer_message_contains_is_a_problem() -> None:
    r = check_grounding("The window is 30 days. [returns#return-window]", [RETURNS])
    assert r.problems == ("states '30 days', which is in no policy text or tool result",)


@pytest.mark.parametrize(
    ("reply", "bad"),
    [
        ("Refunds take 24 hours.", ["24 hours"]),
        ("It costs INR 99.", ["INR 99"]),
        ("It costs Rs. 99 extra.", ["Rs. 99"]),
        ("Refunds are 100% sure.", ["100%"]),
        ("About 3 business days.", ["3 business days"]),
        ("Within 2 weeks.", ["2 weeks"]),
        ("That is ₹250 off.", ["₹250"]),
    ],
)
def test_units_currencies_and_percentages_are_checked(reply: str, bad: list[str]) -> None:
    r = check_grounding(reply + " [returns#return-window]", [RETURNS])
    assert [p.split("'")[1] for p in r.problems] == bad


@pytest.mark.parametrize(
    "reply",
    [
        "You have 7 days. [returns#return-window]",
        "Within 7 days.",
        "That is 7 days from delivery",
    ],
)
def test_supported_numbers_pass(reply: str) -> None:
    assert check_grounding(reply, [RETURNS]).ok


def test_numbers_compare_by_value_not_by_formatting() -> None:
    cod = search_result(("payments#cash-on-delivery-limit", "COD is up to INR 5,000."))
    assert check_grounding("Up to INR 5000 works.", [cod]).ok
    assert check_grounding("Up to INR 5,000.00 works.", [cod]).ok
    assert not check_grounding("Up to INR 50,000 works.", [cod]).ok
    order = {"role": "tool", "content": json.dumps({"ok": True, "data": {"total_inr": "2898.00"}})}
    assert check_grounding("You paid INR 2,898 [xx#yy]", [order, RETURNS]).problems == (
        "cites [xx#yy], which no policy search returned in this run",
    )


def test_a_number_the_customer_wrote_counts_as_evidence() -> None:
    cod = search_result(("payments#cash-on-delivery-limit", "COD is up to INR 5,000."))
    reply = "An order of 8000 rupees is above the limit."
    assert not check_grounding(reply, [cod]).ok
    assert check_grounding(reply, [cod], customer_text="I want to buy 8000 rupees of goods").ok


def test_replies_that_neither_cite_nor_follow_a_search_are_not_checked() -> None:
    r = check_grounding("Your parcel arrives in 2 days.", [])
    assert not r.applicable and r.ok and r.text == "Your parcel arrives in 2 days."
    order = {"role": "tool", "content": json.dumps({"ok": True, "data": {"order_id": "KRT-1"}})}
    assert not check_grounding("It should arrive in 5 days.", [order]).applicable


def test_a_search_that_found_nothing_still_checks_the_reply() -> None:
    nothing = search_result(found=False)
    assert check_grounding("A colleague will follow up with you.", [nothing]).ok
    assert not check_grounding("Returns take 30 days.", [nothing]).ok


def test_ordinary_brackets_and_malformed_tool_messages_are_left_alone() -> None:
    junk: list[dict[str, Any]] = [
        {"role": "tool", "content": "not json"},
        {"role": "tool", "content": 5},
        {"role": "tool", "content": json.dumps([1, 2])},
        {"role": "assistant", "content": "x"},
    ]
    r = check_grounding("Order [KRT-100001 is fine] and (see below).", junk)
    assert not r.applicable and r.text == "Order [KRT-100001 is fine] and (see below)."


def test_duplicate_problems_and_citations_are_reported_once() -> None:
    r = check_grounding("30 days. Again 30 days. [a-b] [a-b]", [RETURNS])
    assert r.citations == ("a-b",) and len(r.problems) == 2  # one citation, one quantity


def test_a_reply_that_is_only_a_citation_keeps_the_original_text() -> None:
    r = check_grounding("[returns#return-window]", [RETURNS])
    assert r.ok and r.text == "[returns#return-window]"


# ----------------------------------------------------------------------- through the runner


def runner_with(
    factory: sessionmaker[Session], script: list[Any], chunk_text: str = "Returns take 7 days."
) -> TicketRunner:
    chunk = Chunk(
        id="returns#return-window", doc="returns", title="R", heading="W", text=chunk_text
    )

    class One:
        name = "one"

        def search(self, query: str, k: int = 5) -> list[Hit]:
            return [Hit(chunk, 0.9)]

    tool = policy_tool(One(), min_score=0.5)
    gateway = ToolGateway(factory, [*default_tools(), tool], clock=lambda: FIXED_NOW)
    return TicketRunner(factory, ScriptedClient(script), gateway, model_tier="scripted")


def ask(runner: TicketRunner, text: str = "How long do I have to return?") -> Any:
    return runner.run(runner.create_ticket("C-0001", text))


SEARCH = make_response(None, [tool_call("search_policy", {"question": "return window"})])


def test_a_grounded_reply_is_stored_without_its_citation_and_traced(
    seeded_factory: sessionmaker[Session],
) -> None:
    runner = runner_with(seeded_factory, [SEARCH, make_response("7 days. [returns#return-window]")])
    summary = ask(runner)
    assert summary.outcome == "drafted" and summary.answer == "7 days."
    with seeded_factory() as session:
        agent = session.scalars(select(Message).where(Message.role == MessageRole.AGENT)).one()
        assert agent.content == "7 days."
        trace = load_trace(session, summary.run_id)
    assert trace is not None
    assert "grounding: ok (returns#return-window)" in format_trace(trace)


def test_an_ungrounded_reply_is_held_not_drafted(seeded_factory: sessionmaker[Session]) -> None:
    runner = runner_with(
        seeded_factory, [SEARCH, make_response("30 days. [returns#return-window]")]
    )
    summary = ask(runner)
    assert summary.outcome == "escalated" and summary.answer is None
    assert summary.stop_reason.value == "ungrounded"
    with seeded_factory() as session:
        assert session.scalars(select(Message).where(Message.role == MessageRole.AGENT)).all() == []
        run = session.scalars(select(Run)).one()
        assert run.status is RunStatus.SUCCEEDED and run.outcome == "escalated"
        assert run.error is not None and "30 days" in run.error
        assert session.scalars(select(Ticket)).one().status is TicketStatus.ESCALATED
        [audit] = session.scalars(
            select(AuditLog).where(AuditLog.action == "guard.ungrounded_reply")
        )
        assert audit.actor_type is ActorType.SYSTEM and audit.actor_id == "grounding"
        assert audit.details is not None and audit.details["citations"] == ["returns#return-window"]
        trace = load_trace(session, summary.run_id)
    assert trace is not None
    assert "grounding: held for a human (states '30 days'" in format_trace(trace)
    assert run.tokens_in > 0  # the run still reports what it cost


def test_an_invented_citation_with_no_search_is_held(seeded_factory: sessionmaker[Session]) -> None:
    runner = runner_with(seeded_factory, [make_response("Do this. [returns#made-up]")])
    summary = ask(runner)
    assert summary.outcome == "escalated" and summary.stop_reason.value == "ungrounded"


def test_ordinary_replies_get_no_grounding_step(seeded_factory: sessionmaker[Session]) -> None:
    runner = runner_with(seeded_factory, [make_response("Hello, it arrives in 2 days.")])
    summary = ask(runner, "hi")
    assert summary.outcome == "drafted"
    with seeded_factory() as session:
        trace = load_trace(session, summary.run_id)
    assert trace is not None and "grounding" not in format_trace(trace)


def test_the_guard_can_be_switched_off(seeded_factory: sessionmaker[Session]) -> None:
    chunk = Chunk(id="a#b", doc="a", title="A", heading="B", text="x")

    class One:
        name = "one"

        def search(self, query: str, k: int = 5) -> list[Hit]:
            return [Hit(chunk, 0.9)]

    gateway = ToolGateway(
        seeded_factory, [policy_tool(One(), min_score=0.1)], clock=lambda: FIXED_NOW
    )
    client = ScriptedClient([make_response("30 days. [nope#nope]")])
    runner = TicketRunner(seeded_factory, client, gateway, grounding=None)
    summary = runner.run(runner.create_ticket("C-0001", "q"))
    assert summary.outcome == "drafted" and summary.answer == "30 days. [nope#nope]"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("5,000", "5000"), ("5000.00", "5000"), ("007", "7"), ("0", "0"), ("0.0", "0"),
     ("0.50", "0.5"), ("10.0", "10"), ("100", "100"), ("1,234.50", "1234.5")],
)  # fmt: skip
def test_number_normalisation(raw: str, expected: str) -> None:
    assert _normal(raw) == expected
