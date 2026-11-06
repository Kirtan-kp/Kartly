"""The context builder: the token budget, thread trimming and the breakdown in the trace."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.context import OMITTED_NOTE, ContextBuilder, estimate_tokens
from kartly.context.builder import MIN_NEWEST_CHARS
from kartly.gateway import ToolGateway
from kartly.orchestrator import TicketRunner, format_trace, load_trace
from kartly.orchestrator.prompts import system_prompt
from kartly.store.models import Message, MessageRole, RunStep, StepKind
from tests.llm_support import ScriptedClient, make_response
from tests.support import FIXED_NOW

C, A = MessageRole.CUSTOMER, MessageRole.AGENT
SPECS = [{"type": "function", "function": {"name": "t", "description": "x" * 400}}]


def turn(role: MessageRole, n: int) -> tuple[MessageRole, str]:
    return (role, f"m{n} " + "word " * 40)  # about 50 tokens each


def thread_of(count: int) -> list[tuple[MessageRole, str]]:
    return [turn(C if i % 2 == 0 else A, i) for i in range(count)]


# ------------------------------------------------------------------------------ estimates


@pytest.mark.parametrize(("text", "tokens"), [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2)])
def test_token_estimate_rounds_up_at_four_characters_per_token(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


def test_a_short_thread_is_untouched_and_the_breakdown_adds_up() -> None:
    built = ContextBuilder(6000).build([(C, "where is my order?")], tool_specs=SPECS)
    b = built.breakdown
    assert [m["role"] for m in built.messages] == ["system", "user"]
    assert built.messages[1]["content"] == "where is my order?"
    assert b.system == estimate_tokens(system_prompt())
    assert b.tools > 100 and b.thread == estimate_tokens("where is my order?")
    assert b.total == b.system + b.tools + b.thread and b.budget == 6000
    assert (b.omitted_messages, b.truncated_newest, b.over_budget) == (0, False, False)
    assert b.to_dict()["total"] == b.total


def test_policy_rules_and_tool_definitions_are_counted() -> None:
    plain = ContextBuilder().build([(C, "hi")])
    policy = ContextBuilder().build([(C, "hi")], policy=True, policy_prompt="p4")
    assert policy.breakdown.system > plain.breakdown.system
    assert policy.messages[0]["content"] == system_prompt(True, "p4")
    assert ContextBuilder().build([(C, "hi")], tool_specs=SPECS).breakdown.tools > 0
    assert plain.breakdown.tools == estimate_tokens("[]")


def test_agent_and_human_messages_keep_their_chat_roles_and_system_entries_are_hidden() -> None:
    thread = [(C, "a"), (A, "b"), (MessageRole.HUMAN_AGENT, "c"), (MessageRole.SYSTEM, "secret")]
    roles = [m["role"] for m in ContextBuilder().build(thread).messages]
    assert roles == ["system", "user", "assistant", "assistant"]


# ------------------------------------------------------------------------------ trimming


def test_a_long_thread_keeps_the_first_and_newest_messages_and_says_what_it_left_out() -> None:
    thread = thread_of(20)
    fixed = ContextBuilder(10_000).build([thread[0]]).breakdown
    budget = fixed.system + 5 * 55  # room for about five thread messages
    built = ContextBuilder(budget).build(thread)
    texts = [m["content"] for m in built.messages[1:]]
    b = built.breakdown
    assert texts[0] == thread[0][1] and texts[-1] == thread[-1][1]
    assert texts.count(OMITTED_NOTE) == 1 and texts[1] == OMITTED_NOTE
    kept_real = len(texts) - 1  # without the note
    assert b.omitted_messages == 20 - kept_real and 0 < b.omitted_messages < 20
    assert b.total <= budget and not b.over_budget and not b.truncated_newest
    # what is kept after the note is a contiguous run of the newest messages
    tail = texts[2:]
    assert tail == [t for _, t in thread[20 - len(tail) :]]


def test_no_note_is_added_when_nothing_is_left_out() -> None:
    thread = thread_of(4)
    built = ContextBuilder(10_000).build(thread)
    assert OMITTED_NOTE not in [m["content"] for m in built.messages]
    assert built.breakdown.omitted_messages == 0


def test_exactly_fitting_threads_are_not_trimmed_and_one_token_less_is() -> None:
    thread = thread_of(6)
    full = ContextBuilder(10_000).build(thread).breakdown
    assert ContextBuilder(full.total).build(thread).breakdown.omitted_messages == 0
    trimmed = ContextBuilder(full.total - 1).build(thread).breakdown
    assert trimmed.omitted_messages > 0 and trimmed.total <= full.total - 1


def test_a_single_message_larger_than_the_room_is_cut_with_an_ellipsis() -> None:
    big = "long " * 4000  # the builder's own 4,000-character message cap leaves about 1,000 tokens
    built = ContextBuilder(400).build([(C, big)])
    text = built.messages[1]["content"]
    assert text.endswith("…") and len(text) < 4000
    assert built.breakdown.truncated_newest and not built.breakdown.over_budget


def test_the_newest_message_is_cut_before_the_first_one_is_dropped() -> None:
    thread = [(C, "original request"), (A, "ok"), (C, "x " * 5000)]
    built = ContextBuilder(600).build(thread)
    texts = [m["content"] for m in built.messages[1:]]
    assert texts[0] == "original request" and texts[1] == OMITTED_NOTE
    assert texts[-1].endswith("…") and built.breakdown.truncated_newest
    assert built.breakdown.omitted_messages == 1


def test_an_impossible_budget_is_flagged_and_the_newest_message_keeps_a_minimum() -> None:
    thread = [(C, "first"), (A, "second"), (C, "z " * 3000)]
    built = ContextBuilder(1).build(thread, tool_specs=SPECS)
    newest = built.messages[-1]["content"]
    assert built.breakdown.over_budget and newest.startswith("z z") and newest.endswith("…")
    assert len(newest) == MIN_NEWEST_CHARS  # the ellipsis counts inside the minimum


def test_an_empty_thread_and_a_bad_budget() -> None:
    built = ContextBuilder(100).build([])
    assert [m["role"] for m in built.messages] == ["system"] and built.breakdown.thread == 0
    for bad in (0, -5):
        with pytest.raises(ValueError, match="budget_tokens"):
            ContextBuilder(bad)


# --------------------------------------------------------------------- through the runner


def add_thread(factory: sessionmaker[Session], ticket_id: Any, extra: int) -> None:
    with factory() as session:
        start = datetime.now(UTC) + timedelta(seconds=1)
        for i in range(extra):
            role = A if i % 2 == 0 else C
            session.add(
                Message(
                    ticket_id=ticket_id,
                    role=role,
                    content=turn(role, i + 1)[1],
                    created_at=start + timedelta(seconds=i),
                )
            )
        session.commit()


def runner_with_budget(factory: sessionmaker[Session], budget: int) -> tuple[TicketRunner, Any]:
    client = ScriptedClient([make_response("Done.")])
    gateway = ToolGateway(factory, clock=lambda: FIXED_NOW)
    return TicketRunner(factory, client, gateway, context_budget=budget), client


def test_the_first_model_call_records_the_breakdown_and_the_trace_shows_est_vs_billed(
    seeded_factory: sessionmaker[Session],
) -> None:
    runner, client = runner_with_budget(seeded_factory, 6000)
    summary = runner.run(runner.create_ticket("C-0001", "where is KRT-100002?"))
    with seeded_factory() as session:
        step = session.scalars(select(RunStep).where(RunStep.kind == StepKind.LLM_CALL)).one()
        assert step.input is not None
        assert step.input["context"]["budget"] == 6000
        assert step.input["context"]["total"] > 200
        trace = load_trace(session, summary.run_id)
    assert trace is not None
    line = next(ln for ln in format_trace(trace).splitlines() if "replies" in ln)
    assert "[prompt ~" in line and "tools " in line and "billed]" in line
    assert "left out" not in line and "OVER BUDGET" not in line
    assert client.calls[0]["messages"][0]["role"] == "system"


def test_a_long_ticket_is_trimmed_before_the_model_sees_it_and_the_trace_says_so(
    seeded_factory: sessionmaker[Session],
) -> None:
    runner, client = runner_with_budget(seeded_factory, 1000)
    ticket_id = runner.create_ticket("C-0001", "the original question")
    add_thread(seeded_factory, ticket_id, 30)
    summary = runner.run(ticket_id)
    sent = [m["content"] for m in client.calls[0]["messages"][1:]]
    assert sent[0] == "the original question" and OMITTED_NOTE in sent
    assert len(sent) < 31
    with seeded_factory() as session:
        trace = load_trace(session, summary.run_id)
    assert trace is not None and "message(s) left out" in format_trace(trace)


def test_only_the_first_model_call_carries_the_breakdown(
    seeded_factory: sessionmaker[Session],
) -> None:
    from tests.llm_support import tool_call

    client = ScriptedClient(
        [
            make_response(None, [tool_call("get_order", {"order_id": "KRT-100001"})]),
            make_response("Done."),
        ]
    )
    gateway = ToolGateway(seeded_factory, clock=lambda: FIXED_NOW)
    runner = TicketRunner(seeded_factory, client, gateway)
    runner.run(runner.create_ticket("C-0001", "order KRT-100001?"))
    with seeded_factory() as session:
        steps = list(
            session.scalars(
                select(RunStep).where(RunStep.kind == StepKind.LLM_CALL).order_by(RunStep.idx)
            )
        )
    assert steps[0].input is not None and steps[1].input is not None
    assert "context" in steps[0].input and "context" not in steps[1].input


def test_a_middle_message_that_exactly_fits_the_room_is_kept() -> None:
    thread = thread_of(5)  # first, three middle messages, newest
    cost = estimate_tokens  # alias for readability
    fixed = ContextBuilder(100_000).build([thread[0]]).breakdown
    first = cost(thread[0][1])
    note = cost(OMITTED_NOTE)
    newest, m3 = cost(thread[4][1]), cost(thread[3][1])
    room = first + note + newest + m3  # exactly the first, the note, the newest and one more
    built = ContextBuilder(fixed.system + fixed.tools + room).build(thread)
    texts = [m["content"] for m in built.messages[1:]]
    assert texts == [thread[0][1], OMITTED_NOTE, thread[3][1], thread[4][1]]
    assert built.breakdown.omitted_messages == 2
    one_less = ContextBuilder(fixed.system + fixed.tools + room - 1).build(thread)
    assert [m["content"] for m in one_less.messages[1:]] == [
        thread[0][1],
        OMITTED_NOTE,
        thread[4][1],
    ]


def test_cutting_only_the_newest_message_adds_no_note_when_nothing_else_is_left_out() -> None:
    thread = [(C, "first"), (A, ""), (C, "y " * 3000)]  # the empty middle message costs nothing
    fixed = ContextBuilder(100_000).build([thread[0]]).breakdown
    built = ContextBuilder(fixed.system + fixed.tools + 700).build(thread)
    texts = [m["content"] for m in built.messages[1:]]
    assert OMITTED_NOTE not in texts and texts[:2] == ["first", ""]
    assert built.breakdown.omitted_messages == 0 and built.breakdown.truncated_newest
