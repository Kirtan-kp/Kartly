"""Property tests for the agent loop's central guarantees.

Whatever the model does (valid calls, garbage, made-up tools, injected identities, empty or cut-off
replies, provider errors, never stopping), the loop must:

- end, without raising, inside its step and tool-call caps,
- give a stop reason and an answer only together,
- run only what the gateway allows, for the ticket's own customer,
- leave a complete, contiguous, storage-safe trace whose totals add up.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.llm import LLMTimeoutError, RateLimitError, ToolUseFailedError
from kartly.llm.types import ChatResponse, ToolCall
from kartly.orchestrator import AgentLoop, RunBudget, StopReason
from kartly.orchestrator.prompts import build_messages
from kartly.store.models import AuditLog, MessageRole, Order, StepKind
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW, make_ctx, make_sqlite_engine, seeded_session_factory

PROPERTY = settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
CUSTOMER = "C-0001"
MESSAGES = build_messages([(MessageRole.CUSTOMER, "Where is my order?")])

ORDER_IDS = st.sampled_from(
    ["KRT-100001", "KRT-100002", "KRT-100003", "KRT-100006", "KRT-999999", "krt-100002 ", "nope"]
)
TOOL_NAMES = st.sampled_from(
    ["get_order", "get_tracking", "get_order", "issue_refund", "drop_table"]
)
ARGUMENTS = st.one_of(
    st.builds(lambda o: {"order_id": o}, ORDER_IDS),
    st.builds(lambda o: {"order_id": o, "customer_id": "C-0002"}, ORDER_IDS),
    st.just({}),
    st.dictionaries(
        st.text(max_size=8), st.one_of(st.integers(), st.text(max_size=20), st.none()), max_size=4
    ),
)


@st.composite
def calls(draw: st.DrawFn) -> ToolCall:
    name = draw(TOOL_NAMES)
    call_id = draw(st.sampled_from(["a", "b", "c", "d", "e", ""]))
    if draw(st.integers(0, 5)) == 0:  # broken arguments
        return tool_call(
            name, None, raw=draw(st.text(max_size=30)), parse_error="broken", call_id=call_id
        )
    return tool_call(name, draw(ARGUMENTS), call_id=call_id)


@st.composite
def responses(draw: st.DrawFn) -> ChatResponse | Exception:
    kind = draw(st.integers(0, 9))
    if kind <= 4:
        return make_response(None, draw(st.lists(calls(), min_size=1, max_size=4)))
    if kind == 5:
        return make_response(draw(st.text(min_size=1, max_size=50)))
    if kind == 6:
        return make_response(draw(st.sampled_from(["", "   ", None])))
    if kind == 7:
        cut = make_response(draw(st.text(max_size=20)))
        object.__setattr__(cut, "finish_reason", "length")
        return cut
    if kind == 8:
        return draw(
            st.sampled_from(
                [
                    ToolUseFailedError("HTTP 400", model="m"),
                    RateLimitError("HTTP 429", model="m", status=429),
                    LLMTimeoutError("slow", model="m"),
                ]
            )
        )
    return make_response(
        draw(st.text(max_size=20)),
        [tool_call("get_order", {"order_id": "KRT-100002"})],
        prompt_tokens=draw(st.integers(0, 12_000)),
        completion_tokens=draw(st.integers(0, 3_000)),
        cost=draw(st.sampled_from(["0", "0.00001", "0.2"])),
    )


class EndlessClient(ScriptedClient):
    """After the script runs out, the model asks for a new tool call forever."""

    def __init__(self, script: list[ChatResponse | Exception]) -> None:
        super().__init__(script)
        self._n = 0

    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        if not self._queue:
            self._n += 1
            self._queue.append(
                make_response(
                    None,
                    [
                        tool_call(
                            "get_order", {"order_id": f"KRT-8{self._n:05d}"}, call_id=str(self._n)
                        )
                    ],
                )
            )
        return super().chat(messages, **kwargs)


@dataclass
class World:
    gateway: ToolGateway
    factory: sessionmaker[Session]
    owned: set[str]


@pytest.fixture(scope="module")
def world() -> Iterator[World]:
    engine = make_sqlite_engine()
    factory = seeded_session_factory(engine)
    with factory() as session:
        owned = {o.id for o in session.scalars(select(Order)) if o.customer_id == CUSTOMER}
    yield World(ToolGateway(factory, clock=lambda: FIXED_NOW), factory, owned)
    engine.dispose()


@PROPERTY
@given(
    script=st.lists(responses(), max_size=14),
    max_steps=st.integers(1, 7),
    max_tool_calls=st.integers(1, 9),
    max_repairs=st.integers(0, 2),
)
def test_the_loop_always_ends_inside_its_limits_and_leaves_a_clean_trace(
    world: World,
    script: list[ChatResponse | Exception],
    max_steps: int,
    max_tool_calls: int,
    max_repairs: int,
) -> None:
    budget = RunBudget(max_steps=max_steps, max_tool_calls=max_tool_calls, max_repairs=max_repairs)
    client = EndlessClient(script)
    ctx = make_ctx(CUSTOMER)
    result = AgentLoop(client, world.gateway, budget=budget).run(ctx, MESSAGES)  # must not raise

    event(f"stop={result.stop_reason}")  # shown by --hypothesis-show-statistics

    # it ended, inside the caps, and used exactly the requests it reports
    assert result.stop_reason in set(StopReason)
    assert result.llm_calls <= max_steps and result.llm_calls == len(client.calls)
    assert result.tool_calls <= max_tool_calls

    # an answer exists exactly when the run was answered
    assert (result.answer is not None) == (result.stop_reason is StopReason.ANSWERED)
    assert result.needs_human == (result.answer is None)
    if result.answer is not None:
        assert result.answer.strip() == result.answer and result.answer

    # the gateway ran every tool call that was counted, for this run and this customer only
    with world.factory() as session:
        audit = list(session.scalars(select(AuditLog).where(AuditLog.run_id == ctx.run_id)))
    assert len(audit) == result.tool_calls
    assert all(row.actor_id == f"run:{ctx.run_id}" for row in audit)
    for row in audit:
        if row.action == "tool.succeeded":
            assert row.target_id in world.owned  # no one else's order was ever returned

    # the trace is complete, contiguous, safe to store, and adds up
    assert [s.idx for s in result.steps] == list(range(len(result.steps)))
    llm_steps = [s for s in result.steps if s.kind is StepKind.LLM_CALL]
    assert len(llm_steps) == result.llm_calls
    assert sum(s.tokens_in for s in result.steps) == result.tokens_in
    assert sum(s.tokens_out for s in result.steps) == result.tokens_out
    assert (
        sum((s.cost_usd for s in result.steps), start=type(result.cost_usd)(0)) == result.cost_usd
    )
    for step in result.steps:
        json.dumps([step.input, step.output], allow_nan=False).encode("utf-8")
        assert len(step.name) <= 80

    # tool messages in the history always answer a call the model actually made
    asked = {
        c["id"]
        for m in result.messages
        if m["role"] == "assistant"
        for c in m.get("tool_calls", [])
    }
    answered = [m["tool_call_id"] for m in result.messages if m["role"] == "tool"]
    assert set(answered) <= asked
