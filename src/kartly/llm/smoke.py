"""Smoke-test one configured model against its real provider.

    python -m kartly.llm.smoke --list
    python -m kartly.llm.smoke local_small
    python -m kartly.llm.smoke groq_small

Three checks, each of which the agent loop depends on:
1. plain_completion   the endpoint answers and reports usage
2. tool_call          the model calls a tool, with arguments that pass the real tool's schema
3. tool_round_trip    the provider accepts our assistant tool-call message plus the tool result
                      and the model then answers in words

The tool result in step 3 is canned, so no database is needed. Real API keys are read from the
environment (or .env) and are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from kartly.config import get_settings, load_env
from kartly.gateway.gateway import build_tool_specs
from kartly.llm.errors import LLMConfigError, LLMError, ToolUseFailedError
from kartly.llm.registry import ModelRegistry
from kartly.llm.types import ChatClient, ChatResponse
from kartly.sanitize import clean_str
from kartly.tools import Tool, default_tools

SYSTEM_PROMPT = (
    "You are a customer support agent for Kartly, an online store. "
    "Use the tools to look up order information. Never guess order details."
)
QUESTION = "Hi, where is my order KRT-100002?"
EXPECTED_ORDER_ID = "KRT-100002"
# Reasoning models spend part of the cap on thinking, so a tight cap can leave the answer empty.
MAX_TOKENS = 512

_ORDER = {
    "order_id": EXPECTED_ORDER_ID,
    "status": "shipped",
    "placed_at": "2026-10-20T12:00:00+00:00",
    "delivered_at": None,
    "total_inr": "3299.00",
    "has_shipment": True,
}
_TRACKING = {
    "order_id": EXPECTED_ORDER_ID,
    "order_status": "shipped",
    "shipped": True,
    "shipments": [
        {
            "carrier": "Delhivery",
            "status": "in_transit",
            "estimated_delivery": "2026-10-25T12:00:00+00:00",
            "delayed": False,
            "recent_events": [
                {"ts": "2026-10-22T09:00:00+00:00", "location": "Mumbai", "status": "in_transit"}
            ],
        }
    ],
}
CANNED_RESULTS: dict[str, dict[str, Any]] = {
    "get_order": {"ok": True, "data": _ORDER},
    "get_tracking": {"ok": True, "data": _TRACKING},
}


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    latency_ms: int = 0
    tokens: int = 0
    cost_usd: Decimal = Decimal(0)
    skipped: bool = False


def _measured(name: str, ok: bool, detail: str, response: ChatResponse) -> StepResult:
    return StepResult(
        name, ok, detail, response.latency_ms, response.usage.total_tokens, response.cost_usd
    )


def _cut_off(response: ChatResponse) -> str:
    if response.finish_reason == "length":
        return " (cut off at max_tokens: a reasoning model may have spent it all thinking)"
    return ""


def _failed(name: str, exc: LLMError) -> StepResult:
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ToolUseFailedError) and exc.failed_generation:
        detail += f" | model output: {clean_str(exc.failed_generation, 160)!r}"
    return StepResult(name, False, detail)


def _check_tool_calls(response: ChatResponse, tools: Sequence[Tool]) -> tuple[bool, str]:
    by_name = {tool.name: tool for tool in tools}
    if not response.tool_calls:
        return False, f"answered without calling a tool: {clean_str(response.content or '', 100)!r}"
    notes: list[str] = []
    order_ids: set[str] = set()
    for call in response.tool_calls:
        tool = by_name.get(call.name)
        if tool is None:
            return False, f"called unknown tool {clean_str(call.name, 40)!r}"
        if call.arguments is None:
            return False, f"{call.name}: unparseable arguments ({call.parse_error})"
        try:
            parsed = tool.args_model.model_validate(call.arguments)
        except ValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            return False, f"{call.name}: arguments fail the tool schema ({first['msg']})"
        order_ids.add(str(getattr(parsed, "order_id", "")))
        notes.append(f"{call.name}({call.raw_arguments})")
    if EXPECTED_ORDER_ID not in order_ids:
        return False, f"valid calls but for the wrong order: {', '.join(notes)}"
    return True, "valid call: " + ", ".join(notes)


def run_smoke(client: ChatClient, tools: Sequence[Tool] | None = None) -> list[StepResult]:
    tool_list = tuple(default_tools() if tools is None else tools)
    specs = build_tool_specs(tool_list)
    results: list[StepResult] = []

    # 1. plain completion
    try:
        reply = client.chat(
            [
                {"role": "system", "content": "You are a terse assistant."},
                {"role": "user", "content": "Reply with exactly the single word OK."},
            ],
            max_tokens=MAX_TOKENS,
        )
    except LLMError as exc:
        results.append(_failed("plain_completion", exc))
        results += [
            StepResult(name, False, "skipped: the first step failed", skipped=True)
            for name in ("tool_call", "tool_round_trip")
        ]
        return results
    said_ok = "ok" in (reply.content or "").lower()
    estimate = " (token counts estimated: provider sent no usage)" if reply.usage.estimated else ""
    results.append(
        _measured(
            "plain_completion",
            said_ok,
            f"replied {clean_str(reply.content or '', 60)!r}{_cut_off(reply)}{estimate}",
            reply,
        )
    )

    # 2. tool call
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    try:
        first = client.chat(messages, tools=specs, max_tokens=MAX_TOKENS)
    except LLMError as exc:
        results.append(_failed("tool_call", exc))
        results.append(StepResult("tool_round_trip", False, "skipped: no tool call", skipped=True))
        return results
    ok, detail = _check_tool_calls(first, tool_list)
    if not ok:
        detail += _cut_off(first)
    results.append(_measured("tool_call", ok, detail, first))
    if not ok:
        results.append(
            StepResult("tool_round_trip", False, "skipped: no valid tool call", skipped=True)
        )
        return results

    # 3. tool round trip: send the tool results back and expect a written answer
    follow_up = [*messages, first.assistant_message()]
    for call in first.tool_calls:
        canned = CANNED_RESULTS.get(call.name, {"ok": False, "error": {"code": "unavailable"}})
        follow_up.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(canned)})
    try:
        final = client.chat(follow_up, tools=specs, max_tokens=MAX_TOKENS)
    except LLMError as exc:
        results.append(_failed("tool_round_trip", exc))
        return results
    answered = bool((final.content or "").strip())
    if answered:
        detail = f"answered: {clean_str(final.content or '', 100)!r}"
    elif final.tool_calls:
        detail = "asked for another tool call instead of answering"
    else:
        detail = "returned an empty reply" + _cut_off(final)
    results.append(_measured("tool_round_trip", answered, detail, final))
    return results


def format_report(model_name: str, results: Sequence[StepResult]) -> str:
    lines = [f"Model: {model_name}"]
    for step in results:
        status = "SKIP" if step.skipped else ("PASS" if step.ok else "FAIL")
        timing = f"{step.latency_ms / 1000:5.1f}s {step.tokens:5d} tok" if not step.skipped else ""
        lines.append(f"  [{status}] {step.name:<16} {timing}  {step.detail}")
    total = sum((step.cost_usd for step in results), Decimal(0))
    tokens = sum(step.tokens for step in results)
    lines.append(
        f"Total: {tokens} tokens, shadow cost ${total:.6f} at the reference prices in models.yaml"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a configured model.")
    parser.add_argument("model", nargs="?", help="model name from models.yaml")
    parser.add_argument("--config", help="path to models.yaml (default: config/models.yaml)")
    parser.add_argument("--list", action="store_true", help="list models and roles, then exit")
    args = parser.parse_args(argv)

    load_env()
    try:
        registry = ModelRegistry.from_yaml(args.config or get_settings().models_config_path)
        if args.list or not args.model:
            for name in registry.model_names:
                model = registry.model(name)
                print(f"{name:<14} {model.provider}/{model.model}")
            for role, chain in registry.roles.items():
                print(f"role {role:<9} {' -> '.join(chain)}")
            return 0 if args.list else 2
        with registry.client(args.model) as client:
            model = registry.model(args.model)
            print(f"Testing {args.model} ({model.provider}/{model.model}) ...", flush=True)
            results = run_smoke(client)
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(format_report(args.model, results))
    return 0 if all(step.ok for step in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
