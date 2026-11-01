"""The bounded tool-calling loop: the only place an LLM runs in the resolve stage.

The model may ask for tool calls; the Tool Gateway decides what happens. The loop's job is to
make sure it always ends, always within budget, and always says why:

- at most ``max_steps`` model calls and ``max_tool_calls`` gateway calls,
- a token budget and a shadow-cost cap, checked *before* each model call using a size estimate,
- a wall-clock timeout,
- one repair re-prompt for a broken tool call or an empty reply, then give up,
- identical repeated tool calls are not run again, and a second repeat ends the run.

Every way of not producing an answer is a distinct ``StopReason`` and means "hand to a human".
Nothing here retries a failed provider call: that belongs to the resilient client (Week 5).
Each step is reported through ``on_step`` as it happens, so a crash still leaves a partial trace.
"""

from __future__ import annotations

import enum
import json
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from kartly.gateway import ToolGateway
from kartly.llm import ChatClient, ChatResponse, LLMError, ToolCall, ToolUseFailedError
from kartly.llm.pricing import estimate_prompt_tokens
from kartly.orchestrator.budget import RunBudget
from kartly.orchestrator.prompts import EMPTY_REPLY_NUDGE, REPAIR_NUDGE
from kartly.sanitize import TRACE_LIMITS, clean_str, safe_json
from kartly.store.models import StepKind
from kartly.tools.base import CallContext

MAX_TOOL_CONTENT_CHARS = 8000
MAX_TOOL_NAME_STORED = 60  # run_steps.name is varchar(80); clean_str may add one character
MAX_ID_STORED = 80


class StopReason(enum.StrEnum):
    ANSWERED = "answered"
    STEP_LIMIT = "step_limit"
    TOOL_LIMIT = "tool_limit"
    TOKEN_LIMIT = "token_limit"
    COST_LIMIT = "cost_limit"
    TIMEOUT = "timeout"
    NO_PROGRESS = "no_progress"  # the model keeps repeating the same tool call
    MALFORMED_OUTPUT = "malformed_output"  # broken tool calls or empty replies, repairs used up
    TRUNCATED_REPLY = "truncated_reply"  # the reply was cut off by the output cap
    LLM_ERROR = "llm_error"
    GATEWAY_ERROR = "gateway_error"  # the gateway failed closed (for example, audit write failed)
    SCREENED = "screened"  # the injection screen sent the ticket to a human; no model call was made
    ROUTED_TIER3 = "routed_tier3"  # a hard rule marked it never-autonomous; no model call was made


@dataclass(frozen=True)
class StepRecord:
    """One traced step. Values are already bounded and safe to store (NUL, NaN etc. removed)."""

    idx: int
    kind: StepKind
    name: str
    input: dict[str, Any]
    output: dict[str, Any] | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int | None = None
    cost_usd: Decimal = Decimal(0)
    error: str | None = None


@dataclass(frozen=True)
class LoopResult:
    stop_reason: StopReason
    detail: str  # why the loop stopped, in a sentence (empty when answered)
    answer: str | None
    steps: tuple[StepRecord, ...]
    messages: tuple[dict[str, Any], ...]  # the final history, for debugging and replay tests
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal

    @property
    def answered(self) -> bool:
        return self.stop_reason is StopReason.ANSWERED

    @property
    def needs_human(self) -> bool:
        return not self.answered

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


StepSink = Callable[[StepRecord], None]


@dataclass(frozen=True)
class _Stop:
    reason: StopReason
    detail: str = ""


class AgentLoop:
    def __init__(
        self,
        client: ChatClient,
        gateway: ToolGateway,
        *,
        budget: RunBudget | None = None,
        allowed_tools: Collection[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client
        self.gateway = gateway
        self.budget = budget or RunBudget()
        self.allowed: frozenset[str] | None = (
            None if allowed_tools is None else frozenset(allowed_tools)
        )
        self.specs = gateway.tool_specs(self.allowed)  # raises early on an unknown tool name
        self.clock = clock

    def run(
        self,
        ctx: CallContext,
        messages: Sequence[dict[str, Any]],
        *,
        on_step: StepSink | None = None,
    ) -> LoopResult:
        """Run to completion. Never raises for model or tool trouble; those become stop reasons."""
        return _Execution(self, ctx, [dict(m) for m in messages], on_step).execute()


def _call_key(call: ToolCall) -> str:
    """Identifies a tool call for repeat detection: same tool, same arguments in any key order."""
    try:
        arguments = json.dumps(call.arguments, sort_keys=True, default=str)
    except (ValueError, TypeError, RecursionError):  # arguments no encoder accepts
        arguments = call.raw_arguments
    return f"{call.name}:{arguments}"


def _error_content(code: str, message: str) -> str:
    payload = {"ok": False, "error": {"code": code, "message": message}}
    return json.dumps(payload, separators=(",", ":"))


class _Execution:
    """The state of one run. Created per call to ``AgentLoop.run``, so the loop is reusable."""

    def __init__(
        self,
        loop: AgentLoop,
        ctx: CallContext,
        messages: list[dict[str, Any]],
        on_step: StepSink | None,
    ) -> None:
        self.loop = loop
        self.budget = loop.budget
        self.ctx = ctx
        self.messages = messages
        self.on_step = on_step
        self.steps: list[StepRecord] = []
        self.llm_calls = 0
        self.tool_calls = 0
        self.repairs = 0
        self.duplicates = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost = Decimal(0)
        self.seen: set[str] = set()
        self.answer: str | None = None
        self.started = loop.clock()
        self.recorded_upto = 0  # messages already written into an llm_call step

    # ----------------------------------------------------------------------- driver

    def execute(self) -> LoopResult:
        while True:
            room = self._room()
            if isinstance(room, _Stop):
                return self._finish(room)
            stop = self._step(room)
            if stop is not None:
                return self._finish(stop)

    def _room(self) -> int | _Stop:
        """Check every limit before spending anything. Returns the output cap for the next call."""
        budget = self.budget
        if self.llm_calls >= budget.max_steps:
            return _Stop(StopReason.STEP_LIMIT, f"used all {budget.max_steps} model calls")
        elapsed = self.loop.clock() - self.started
        if elapsed >= budget.timeout_s:
            return _Stop(StopReason.TIMEOUT, f"ran for {elapsed:.0f}s of {budget.timeout_s:g}s")
        if self.cost >= budget.max_cost_usd:
            return _Stop(StopReason.COST_LIMIT, f"shadow cost reached ${self.cost:.4f}")
        prompt = estimate_prompt_tokens(self.messages, self.loop.specs)
        room = budget.max_total_tokens - self.tokens_in - self.tokens_out - prompt
        if room < budget.min_output_tokens:
            return _Stop(
                StopReason.TOKEN_LIMIT,
                f"no room left in the {budget.max_total_tokens}-token budget",
            )
        return min(budget.max_output_tokens, room)

    def _finish(self, stop: _Stop) -> LoopResult:
        answered = stop.reason is StopReason.ANSWERED
        return LoopResult(
            stop_reason=stop.reason,
            detail=stop.detail,
            answer=self.answer if answered else None,
            steps=tuple(self.steps),
            messages=tuple(self.messages),
            llm_calls=self.llm_calls,
            tool_calls=self.tool_calls,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=self.cost,
        )

    # ------------------------------------------------------------------ one step

    def _step(self, max_tokens: int) -> _Stop | None:
        """One model call and whatever it asked for. None means: go round again."""
        try:
            response = self.loop.client.chat(
                self.messages, tools=self.loop.specs or None, max_tokens=max_tokens
            )
        except ToolUseFailedError as exc:  # the provider rejected the model's own tool call
            self._record_failed_call(exc, max_tokens)
            return self._repair("the provider rejected a malformed tool call", REPAIR_NUDGE)
        except LLMError as exc:
            self._record_failed_call(exc, max_tokens)
            return _Stop(StopReason.LLM_ERROR, f"{type(exc).__name__}: {clean_str(str(exc), 200)}")

        self._record_llm_call(response, max_tokens)
        if response.tool_calls:
            self.messages.append(response.assistant_message())
            return self._run_tool_calls(response.tool_calls)
        if response.finish_reason == "length":
            return _Stop(StopReason.TRUNCATED_REPLY, "the reply was cut off by the output cap")
        text = (response.content or "").strip()
        if not text:
            return self._repair("the model returned an empty reply", EMPTY_REPLY_NUDGE)
        self.messages.append(response.assistant_message())
        self.answer = text
        return _Stop(StopReason.ANSWERED)

    def _repair(self, why: str, nudge: str) -> _Stop | None:
        """Spend one repair, or give up when they are used up."""
        self.repairs += 1
        if self.repairs > self.budget.max_repairs:
            return _Stop(StopReason.MALFORMED_OUTPUT, f"{why}; no repairs left")
        self.messages.append({"role": "user", "content": nudge})
        return None

    # ------------------------------------------------------------------ tool calls

    def _run_tool_calls(self, calls: Sequence[ToolCall]) -> _Stop | None:
        malformed = False
        for call in calls:
            if call.arguments is None:
                malformed = True
                self._answer_without_running(
                    call,
                    "invalid_arguments",
                    f"The arguments were not a valid JSON object ({call.parse_error}). "
                    "Call the tool again with valid JSON arguments.",
                )
                continue
            key = _call_key(call)
            if key in self.seen:
                self.duplicates += 1
                self._answer_without_running(
                    call,
                    "duplicate_call",
                    "You already made this exact call. Use the earlier result.",
                )
                if self.duplicates > self.budget.max_duplicate_calls:
                    return _Stop(StopReason.NO_PROGRESS, "the model repeats the same tool call")
                continue
            if self.tool_calls >= self.budget.max_tool_calls:
                return _Stop(
                    StopReason.TOOL_LIMIT, f"used all {self.budget.max_tool_calls} tool calls"
                )
            self.seen.add(key)
            self.tool_calls += 1
            stop = self._execute(call, call.arguments)
            if stop is not None:
                return stop
        if malformed:
            self.repairs += 1
            if self.repairs > self.budget.max_repairs:
                return _Stop(StopReason.MALFORMED_OUTPUT, "broken tool calls; no repairs left")
        return None

    def _execute(self, call: ToolCall, arguments: dict[str, Any]) -> _Stop | None:
        try:
            result = self.loop.gateway.call(
                self.ctx, call.name, arguments, allowed_tools=self.loop.allowed
            )
        except (
            Exception
        ) as exc:  # the gateway fails closed, e.g. when it cannot write the audit row
            error = f"{type(exc).__name__}: {clean_str(str(exc), 200)}"
            self._record_tool(call, arguments, output=None, latency_ms=None, error=error)
            return _Stop(StopReason.GATEWAY_ERROR, error)

        content = result.to_model_content()
        if len(content) > MAX_TOOL_CONTENT_CHARS:
            content = _error_content("result_too_large", "The result was too large to show.")
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        output = {
            "ok": result.ok,
            "error_code": result.error_code,
            "message": result.message,
            "data": result.data,
        }
        self._record_tool(
            call,
            arguments,
            output=output,
            latency_ms=result.latency_ms,
            error=None if result.ok else result.error_code,
        )
        return None

    def _answer_without_running(self, call: ToolCall, code: str, message: str) -> None:
        """Tell the model why a call was not run. It never reaches the gateway."""
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": _error_content(code, message)}
        )
        self._record_tool(
            call,
            call.arguments if call.arguments is not None else {"raw": call.raw_arguments},
            output={"ok": False, "error_code": code, "message": message},
            latency_ms=None,
            error=code,
        )

    # ------------------------------------------------------------------- recording

    def _add(self, step: StepRecord) -> None:
        self.steps.append(step)
        if self.on_step is not None:
            self.on_step(step)

    def _new_messages(self) -> list[Any]:
        """Messages added since the last model call. Together the steps hold the whole prompt."""
        fresh = safe_json(self.messages[self.recorded_upto :], TRACE_LIMITS)
        self.recorded_upto = len(self.messages)
        assert isinstance(fresh, list)
        return fresh

    def _record_llm_call(self, response: ChatResponse, max_tokens: int) -> None:
        self.llm_calls += 1
        self.tokens_in += response.usage.prompt_tokens
        self.tokens_out += response.usage.completion_tokens
        self.cost += response.cost_usd
        output = {
            "content": response.content,
            "tool_calls": [
                {"id": c.id, "name": c.name, "arguments": c.raw_arguments}
                for c in response.tool_calls
            ],
            "finish_reason": response.finish_reason,
            "provider_model": response.provider_model,
            "ignored_params": list(response.ignored_params),
            "usage_estimated": response.usage.estimated,
        }
        self._add(
            StepRecord(
                idx=len(self.steps),
                kind=StepKind.LLM_CALL,
                name="chat",
                input={
                    "new_messages": self._new_messages(),
                    "tools": [s["function"]["name"] for s in self.loop.specs],
                    "max_tokens": max_tokens,
                },
                output=safe_json(output, TRACE_LIMITS),
                model=response.model,
                tokens_in=response.usage.prompt_tokens,
                tokens_out=response.usage.completion_tokens,
                latency_ms=response.latency_ms,
                cost_usd=response.cost_usd,
            )
        )

    def _record_failed_call(self, exc: LLMError, max_tokens: int) -> None:
        self.llm_calls += 1  # a failed request still uses a step, so errors cannot loop forever
        self._add(
            StepRecord(
                idx=len(self.steps),
                kind=StepKind.LLM_CALL,
                name="chat",
                input={
                    "new_messages": self._new_messages(),
                    "tools": [s["function"]["name"] for s in self.loop.specs],
                    "max_tokens": max_tokens,
                },
                model=exc.model,
                error=f"{type(exc).__name__}: {clean_str(str(exc), 300)}",
            )
        )

    def _record_tool(
        self,
        call: ToolCall,
        arguments: dict[str, Any],
        *,
        output: dict[str, Any] | None,
        latency_ms: int | None,
        error: str | None,
    ) -> None:
        stored = safe_json(output, TRACE_LIMITS) if output is not None else None
        self._add(
            StepRecord(
                idx=len(self.steps),
                kind=StepKind.TOOL_CALL,
                name=clean_str(call.name, MAX_TOOL_NAME_STORED),
                input={
                    "call_id": clean_str(call.id, MAX_ID_STORED),
                    "arguments": safe_json(arguments, TRACE_LIMITS),
                },
                output=stored,
                latency_ms=latency_ms,
                error=error,
            )
        )
