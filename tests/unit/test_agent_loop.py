from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from kartly.llm import LLMTimeoutError, RateLimitError, ToolUseFailedError
from kartly.llm.types import ChatResponse
from kartly.orchestrator import AgentLoop, LoopResult, RunBudget, StepRecord, StopReason
from kartly.orchestrator.prompts import build_messages
from kartly.store.models import AuditLog, MessageRole, StepKind
from kartly.tools import RiskTier, Tool
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import make_ctx

CUSTOMER = "C-0001"  # owns KRT-100001 and KRT-100002
MESSAGES = build_messages([(MessageRole.CUSTOMER, "Where is my order KRT-100002?")])


def ask(
    order_id: str = "KRT-100002", tool: str = "get_order", call_id: str = "call_1"
) -> ChatResponse:
    return make_response(None, [tool_call(tool, {"order_id": order_id}, call_id=call_id)])


def unknown_order(n: int) -> ChatResponse:
    """A valid call that is new every time, so duplicate detection never fires."""
    return ask(f"KRT-9{n:05d}", call_id=f"call_{n}")


def run_loop(
    gateway: ToolGateway,
    script: Sequence[ChatResponse | Exception],
    *,
    customer: str = CUSTOMER,
    allowed: list[str] | None = None,
    **budget: Any,
) -> tuple[LoopResult, ScriptedClient]:
    client = ScriptedClient(script)
    loop = AgentLoop(client, gateway, budget=RunBudget(**budget), allowed_tools=allowed)
    return loop.run(make_ctx(customer), MESSAGES), client


def all_audit(factory: sessionmaker[Session]) -> list[AuditLog]:
    with factory() as session:
        return list(session.scalars(select(AuditLog).order_by(AuditLog.id)))


def strings(value: Any) -> Iterator[str]:
    """Every string anywhere in a JSON-like structure, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def kinds(result: LoopResult) -> list[str]:
    return [f"{s.kind.value}:{s.name}" for s in result.steps]


# -------------------------------------------------------------------- the happy path


def test_a_tool_call_then_an_answer(gateway: ToolGateway) -> None:
    result, client = run_loop(
        gateway,
        [
            ask(),
            make_response("Your order is on its way.", prompt_tokens=300, completion_tokens=30),
        ],
    )
    assert result.stop_reason is StopReason.ANSWERED and result.answered
    assert result.answer == "Your order is on its way."
    assert not result.needs_human and result.detail == ""
    assert (result.llm_calls, result.tool_calls) == (2, 1)
    assert kinds(result) == ["llm_call:chat", "tool_call:get_order", "llm_call:chat"]
    assert [s.idx for s in result.steps] == [0, 1, 2]

    # the model saw the tool result, tied to its own call id
    second_prompt = client.calls[1]["messages"]
    assert second_prompt[-2]["tool_calls"][0]["id"] == "call_1"
    assert second_prompt[-1]["role"] == "tool" and second_prompt[-1]["tool_call_id"] == "call_1"
    assert json.loads(second_prompt[-1]["content"])["data"]["order_id"] == "KRT-100002"


def test_totals_are_the_sum_of_the_steps(gateway: ToolGateway) -> None:
    result, _ = run_loop(
        gateway,
        [
            make_response(
                None,
                [tool_call("get_order", {"order_id": "KRT-100002"})],
                prompt_tokens=400,
                completion_tokens=25,
                cost="0.0001",
            ),
            make_response("done", prompt_tokens=600, completion_tokens=40, cost="0.0002"),
        ],
    )
    assert (result.tokens_in, result.tokens_out) == (1000, 65)
    assert result.total_tokens == 1065
    assert result.cost_usd == Decimal("0.0003")
    assert sum(s.tokens_in for s in result.steps) == result.tokens_in
    assert sum(s.cost_usd for s in result.steps) == result.cost_usd


def test_an_answer_with_no_tool_call_is_fine(gateway: ToolGateway) -> None:
    result, client = run_loop(gateway, [make_response("  Hello!  ")])
    assert result.answer == "Hello!"
    assert (result.llm_calls, result.tool_calls) == (1, 0)
    assert len(client.calls) == 1


def test_the_model_is_given_only_the_allowed_tools_and_a_bounded_output(
    gateway: ToolGateway,
) -> None:
    _, client = run_loop(gateway, [make_response("hi")], allowed=["get_tracking"])
    names = [t["function"]["name"] for t in client.calls[0]["tools"]]
    assert names == ["get_tracking"]
    assert 128 <= client.calls[0]["max_tokens"] <= 1024


def test_the_callers_message_list_is_not_modified(gateway: ToolGateway) -> None:
    before = json.dumps(MESSAGES)
    run_loop(gateway, [ask(), make_response("done")])
    assert json.dumps(MESSAGES) == before


def test_steps_are_reported_as_they_happen(gateway: ToolGateway) -> None:
    seen: list[StepRecord] = []
    client = ScriptedClient([ask(), make_response("done")])
    loop = AgentLoop(client, gateway)
    result = loop.run(make_ctx(CUSTOMER), MESSAGES, on_step=seen.append)
    assert tuple(seen) == result.steps


def test_unknown_allowed_tools_fail_at_construction(gateway: ToolGateway) -> None:
    with pytest.raises(ValueError, match="nonsense"):
        AgentLoop(ScriptedClient([]), gateway, allowed_tools=["get_order", "nonsense"])


# --------------------------------------------------------------------------- the caps


def test_step_limit(gateway: ToolGateway) -> None:
    result, client = run_loop(gateway, [unknown_order(n) for n in range(10)], max_steps=3)
    assert result.stop_reason is StopReason.STEP_LIMIT
    assert result.llm_calls == 3 and len(client.calls) == 3
    assert result.needs_human and result.answer is None


def test_tool_call_limit_stops_mid_response(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    three = make_response(
        None,
        [tool_call("get_order", {"order_id": f"KRT-90000{i}"}, call_id=f"c{i}") for i in range(3)],
    )
    result, _ = run_loop(gateway, [three, make_response("never reached")], max_tool_calls=2)
    assert result.stop_reason is StopReason.TOOL_LIMIT
    assert result.tool_calls == 2
    assert len(all_audit(seeded_factory)) == 2  # the third call never reached the gateway


def test_token_budget_is_checked_before_the_first_call(gateway: ToolGateway) -> None:
    result, client = run_loop(gateway, [make_response("never asked")], max_total_tokens=300)
    assert result.stop_reason is StopReason.TOKEN_LIMIT
    assert client.calls == [] and result.llm_calls == 0  # no request was spent


def test_token_budget_is_checked_after_each_call(gateway: ToolGateway) -> None:
    expensive = make_response(
        None,
        [tool_call("get_order", {"order_id": "KRT-100002"})],
        prompt_tokens=9_000,
        completion_tokens=900,
    )
    result, client = run_loop(
        gateway, [expensive, make_response("never reached")], max_total_tokens=10_000
    )
    assert result.stop_reason is StopReason.TOKEN_LIMIT
    assert result.llm_calls == 1 and len(client.calls) == 1
    assert result.tool_calls == 1  # the tool the model asked for still ran


def test_output_cap_shrinks_to_what_is_left_of_the_budget(gateway: ToolGateway) -> None:
    first = make_response(
        None,
        [tool_call("get_order", {"order_id": "KRT-100002"})],
        prompt_tokens=2_000,
        completion_tokens=100,
    )
    _, client = run_loop(gateway, [first, make_response("ok")], max_total_tokens=3_000)
    first_cap, second_cap = (call["max_tokens"] for call in client.calls)
    assert first_cap > second_cap >= 128
    assert second_cap <= 3_000 - 2_100  # never asks for more than the budget has left


def test_cost_limit(gateway: ToolGateway) -> None:
    pricey = make_response(None, [tool_call("get_order", {"order_id": "KRT-100002"})], cost="0.5")
    result, client = run_loop(
        gateway, [pricey, make_response("never")], max_cost_usd=Decimal("0.3")
    )
    assert result.stop_reason is StopReason.COST_LIMIT
    assert len(client.calls) == 1


def test_timeout(gateway: ToolGateway) -> None:
    now = [0.0]

    class SlowClient(ScriptedClient):
        def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
            now[0] += 100.0  # each model call "takes" 100 s
            return super().chat(messages, **kwargs)

    client = SlowClient([unknown_order(n) for n in range(5)])
    loop = AgentLoop(client, gateway, budget=RunBudget(timeout_s=150), clock=lambda: now[0])
    result = loop.run(make_ctx(CUSTOMER), MESSAGES)
    assert result.stop_reason is StopReason.TIMEOUT
    assert result.llm_calls == 2  # 0 s, 100 s fine; 200 s is over


@pytest.mark.parametrize(
    "bad",
    [
        {"max_steps": 0},
        {"max_tool_calls": -1},
        {"max_total_tokens": 0},
        {"timeout_s": 0},
        {"max_cost_usd": Decimal(0)},
        {"max_repairs": -1},
        {"max_duplicate_calls": -1},
        {"min_output_tokens": 2000},
    ],
)
def test_nonsense_budgets_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RunBudget(**bad)


# -------------------------------------------------------------------- broken output


def test_broken_arguments_are_not_run_and_the_model_gets_one_repair(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    broken = make_response(
        None, [tool_call("get_order", None, raw='{"order_id": "KRT-1', parse_error="invalid JSON")]
    )
    result, client = run_loop(gateway, [broken, ask(call_id="call_2"), make_response("done")])
    assert result.answered
    assert result.tool_calls == 1  # only the repaired call reached the gateway
    assert len(all_audit(seeded_factory)) == 1
    told = json.loads(client.calls[1]["messages"][-1]["content"])
    assert told["error"]["code"] == "invalid_arguments"
    assert client.calls[1]["messages"][-1]["tool_call_id"] == "call_1"
    assert kinds(result)[:2] == ["llm_call:chat", "tool_call:get_order"]
    assert result.steps[1].error == "invalid_arguments"


def test_a_second_broken_call_ends_the_run(gateway: ToolGateway) -> None:
    def broken(call_id: str) -> ChatResponse:
        return make_response(
            None, [tool_call("get_order", None, raw="{oops", parse_error="x", call_id=call_id)]
        )

    result, _ = run_loop(gateway, [broken("a"), broken("b"), make_response("never")])
    assert result.stop_reason is StopReason.MALFORMED_OUTPUT
    assert result.tool_calls == 0


def test_two_broken_calls_in_one_response_use_only_one_repair(gateway: ToolGateway) -> None:
    both = make_response(
        None,
        [
            tool_call("get_order", None, raw="{", parse_error="x", call_id="a"),
            tool_call("get_tracking", None, raw="[", parse_error="y", call_id="b"),
        ],
    )
    result, _ = run_loop(gateway, [both, make_response("recovered")])
    assert result.answered


def test_no_repairs_allowed(gateway: ToolGateway) -> None:
    broken = make_response(None, [tool_call("get_order", None, raw="{", parse_error="x")])
    result, _ = run_loop(gateway, [broken, make_response("never")], max_repairs=0)
    assert result.stop_reason is StopReason.MALFORMED_OUTPUT


def test_a_provider_rejected_tool_call_is_repaired_with_a_nudge(gateway: ToolGateway) -> None:
    rejected = ToolUseFailedError("HTTP 400", model="groq_small", failed_generation="<function=")
    result, client = run_loop(gateway, [rejected, make_response("sorry, here is your answer")])
    assert result.answered and result.llm_calls == 2
    nudge = client.calls[1]["messages"][-1]
    assert nudge["role"] == "user" and "not valid" in nudge["content"]
    failed_step = result.steps[0]
    assert failed_step.error is not None and failed_step.error.startswith("ToolUseFailedError")
    assert failed_step.tokens_in == 0


def test_repeated_provider_rejections_end_the_run(gateway: ToolGateway) -> None:
    rejected = ToolUseFailedError("HTTP 400", model="m")
    result, _ = run_loop(gateway, [rejected, rejected, make_response("never")])
    assert result.stop_reason is StopReason.MALFORMED_OUTPUT
    assert result.llm_calls == 2


def test_an_empty_reply_is_repaired_once(gateway: ToolGateway) -> None:
    result, client = run_loop(gateway, [make_response(""), make_response("Here you go.")])
    assert result.answered and result.answer == "Here you go."
    assert "empty" in client.calls[1]["messages"][-1]["content"]

    again, _ = run_loop(gateway, [make_response(None), make_response("   "), make_response("x")])
    assert again.stop_reason is StopReason.MALFORMED_OUTPUT


def test_a_reply_cut_off_by_the_output_cap_is_never_sent(gateway: ToolGateway) -> None:
    cut = make_response("Your order will arrive on Mon")
    object.__setattr__(cut, "finish_reason", "length")
    result, _ = run_loop(gateway, [cut, make_response("never")])
    assert result.stop_reason is StopReason.TRUNCATED_REPLY
    assert result.answer is None and result.needs_human


# ---------------------------------------------------------------------- failures


def test_a_provider_error_stops_the_run_without_retrying(gateway: ToolGateway) -> None:
    result, client = run_loop(gateway, [RateLimitError("HTTP 429", model="m", status=429)])
    assert result.stop_reason is StopReason.LLM_ERROR
    assert "RateLimitError" in result.detail
    assert len(client.calls) == 1  # retries belong to the resilient client, not the loop
    assert result.steps[0].error is not None and result.steps[0].kind is StepKind.LLM_CALL


def test_an_error_after_some_work_keeps_the_earlier_steps(gateway: ToolGateway) -> None:
    result, _ = run_loop(gateway, [ask(), LLMTimeoutError("slow", model="m")])
    assert result.stop_reason is StopReason.LLM_ERROR
    assert kinds(result) == ["llm_call:chat", "tool_call:get_order", "llm_call:chat"]
    assert result.tool_calls == 1


def test_the_gateway_failing_closed_stops_the_run(
    gateway: ToolGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise OperationalError("INSERT audit_log", {}, Exception("disk full"))

    monkeypatch.setattr(gateway, "call", boom)
    result, client = run_loop(gateway, [ask(), make_response("never")])
    assert result.stop_reason is StopReason.GATEWAY_ERROR
    assert "OperationalError" in result.detail
    assert len(client.calls) == 1
    assert result.steps[-1].kind is StepKind.TOOL_CALL and result.steps[-1].error is not None


# --------------------------------------------------------------- no progress


def test_a_repeated_call_is_not_run_again(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    result, client = run_loop(
        gateway, [ask(call_id="a"), ask(call_id="b"), make_response("Thanks, got it.")]
    )
    assert result.answered
    assert result.tool_calls == 1 and len(all_audit(seeded_factory)) == 1
    told = json.loads(client.calls[2]["messages"][-1]["content"])
    assert told["error"]["code"] == "duplicate_call"
    assert result.steps[-2].error == "duplicate_call"


def test_the_same_arguments_in_a_different_key_order_still_count_as_a_repeat(
    gateway: ToolGateway,
) -> None:
    first = make_response(None, [tool_call("get_order", {"order_id": "KRT-100002"}, call_id="a")])
    again = make_response(None, [tool_call("get_order", {"order_id": "KRT-100002"}, call_id="b")])
    result, _ = run_loop(gateway, [first, again, again, make_response("never")])
    assert result.stop_reason is StopReason.NO_PROGRESS


def test_the_third_identical_call_ends_the_run(gateway: ToolGateway) -> None:
    script = [ask(call_id=c) for c in "abcde"]
    result, client = run_loop(gateway, script)
    assert result.stop_reason is StopReason.NO_PROGRESS
    assert len(client.calls) == 3 and result.tool_calls == 1


def test_a_different_tool_on_the_same_order_is_not_a_repeat(gateway: ToolGateway) -> None:
    result, _ = run_loop(
        gateway,
        [
            ask(tool="get_order", call_id="a"),
            ask(tool="get_tracking", call_id="b"),
            make_response("ok"),
        ],
    )
    assert result.answered and result.tool_calls == 2


# ------------------------------------------------------ the gateway stays in charge


def test_a_made_up_tool_is_rejected_by_the_gateway_and_the_run_continues(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    made_up = make_response(None, [tool_call("issue_refund", {"order_id": "KRT-100002"})])
    result, client = run_loop(
        gateway, [made_up, make_response("I can't refund, a colleague will.")]
    )
    assert result.answered
    told = json.loads(client.calls[1]["messages"][-1]["content"])
    assert told["error"]["code"] == "tool_unavailable"
    assert "get_order" in told["error"]["message"]  # tells the model what does exist
    assert [row.action for row in all_audit(seeded_factory)] == ["tool.rejected"]


def test_a_tool_outside_the_allowed_set_is_refused(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    result, _ = run_loop(
        gateway, [ask(tool="get_order"), make_response("done")], allowed=["get_tracking"]
    )
    assert result.answered
    [row] = all_audit(seeded_factory)
    assert row.details is not None and row.details["security_event"] == "disallowed_tool_call"


def test_identity_comes_from_the_ticket_not_from_the_model(
    gateway: ToolGateway, seeded_factory: sessionmaker[Session]
) -> None:
    """A customer asks for someone else's order, and the model even claims to be the owner."""
    injected = make_response(
        None,
        [
            tool_call(
                "get_order", {"order_id": "KRT-100001", "customer_id": "C-0001"}, call_id="a"
            ),
            tool_call("get_order", {"order_id": "KRT-100001"}, call_id="b"),
        ],
    )
    result, client = run_loop(
        gateway, [injected, make_response("I could not find that order.")], customer="C-0002"
    )
    assert result.answered
    replies = [json.loads(m["content"]) for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert [r["error"]["code"] for r in replies] == ["invalid_arguments", "order_not_found"]
    assert all("data" not in r for r in replies)  # nothing about the real order leaked
    events = {
        row.details["security_event"]
        for row in all_audit(seeded_factory)
        if row.details and "security_event" in row.details
    }
    assert events == {"identity_argument_supplied", "cross_customer_access"}


def test_an_oversized_tool_result_is_replaced_not_forwarded(
    seeded_factory: sessionmaker[Session],
) -> None:
    class NoArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")

    huge = Tool(
        name="huge",
        tier=RiskTier.READ_ONLY,
        description="returns far too much",
        args_model=NoArgs,
        handler=lambda env, args: {"blob": "x" * 50_000},
    )
    gateway = ToolGateway(seeded_factory, tools=[huge])
    result, client = run_loop(
        gateway, [make_response(None, [tool_call("huge", {})]), make_response("done")]
    )
    sent = client.calls[1]["messages"][-1]["content"]
    assert len(sent) < 500 and json.loads(sent)["error"]["code"] == "result_too_large"
    assert result.answered


# ----------------------------------------------------------------------- the trace


def test_each_llm_step_records_only_the_new_messages(gateway: ToolGateway) -> None:
    result, _ = run_loop(gateway, [ask(), make_response("done")])
    first, last = (s for s in result.steps if s.kind is StepKind.LLM_CALL)
    assert [m["role"] for m in first.input["new_messages"]] == ["system", "user"]
    assert [m["role"] for m in last.input["new_messages"]] == ["assistant", "tool"]
    assert first.input["tools"] == ["get_order", "get_tracking"]


def test_the_tool_step_records_what_was_asked_and_what_came_back(gateway: ToolGateway) -> None:
    result, _ = run_loop(gateway, [ask(), make_response("done")])
    tool_step = result.steps[1]
    assert tool_step.input == {"call_id": "call_1", "arguments": {"order_id": "KRT-100002"}}
    assert tool_step.output is not None and tool_step.output["ok"] is True
    assert tool_step.output["data"]["order_id"] == "KRT-100002"
    assert tool_step.latency_ms is not None and tool_step.error is None


def test_hostile_model_output_is_safe_to_store(gateway: ToolGateway) -> None:
    nasty_args = '{"order_id": "KRT-100002", "note": "nul\\u0000 surrogate \\ud800", "n": 1e999}'
    hostile = make_response(
        "reply with \x00 nul",
        [
            tool_call(
                "get_order",
                {"order_id": "KRT-100002", "note": "nul\x00 surrogate \ud800", "n": float("nan")},
                raw=nasty_args,
                call_id="id\x00" + "x" * 500,
            )
        ],
    )
    result, _ = run_loop(gateway, [hostile, make_response("fine")])
    assert result.answered
    for step in result.steps:
        text = json.dumps([step.input, step.output], allow_nan=False)  # raises on NaN or Infinity
        text.encode("utf-8")  # raises on lone surrogates
        assert not any("\x00" in value for value in strings([step.input, step.output]))
        assert len(step.name) <= 80


def test_a_very_long_tool_name_cannot_overflow_the_step_name_column(gateway: ToolGateway) -> None:
    long_name = make_response(None, [tool_call("a" * 500, {})])
    result, _ = run_loop(gateway, [long_name, make_response("done")])
    assert all(len(s.name) <= 80 for s in result.steps)


def test_repeat_detection_survives_arguments_no_encoder_accepts() -> None:
    from kartly.llm.types import ToolCall
    from kartly.orchestrator.loop import _call_key

    huge = ToolCall("a", "get_order", {"n": 10**5000}, "RAW", None)
    assert _call_key(huge) == "get_order:RAW"  # falls back to the raw text instead of raising
    first = ToolCall("a", "get_order", {"x": 1, "y": 2}, '{"x":1,"y":2}', None)
    second = ToolCall("b", "get_order", {"y": 2, "x": 1}, '{"y": 2, "x": 1}', None)
    assert _call_key(first) == _call_key(second)  # key order does not matter
