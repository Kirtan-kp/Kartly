"""The Tool Gateway: the only way the agent can touch the store.

The model proposes a tool call; this module decides what actually happens. For every call it:

1. rejects tools that do not exist or are not allowed for this request,
2. validates the arguments strictly (unknown arguments, including ``customer_id``, are errors),
3. runs the tool as the *authenticated* customer from ``CallContext``, never from model output,
4. isolates the tool in a SAVEPOINT, so a failure undoes its writes,
5. writes one audit_log row, in the same transaction, whatever the outcome.

One call is one transaction. If the audit row cannot be written the whole call is rolled back
and the exception propagates: an action without an audit record must not happen (fail closed).
Expected failures never raise; they come back as a ToolResult the model can read and react to.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from kartly.sanitize import clean_str, safe_json
from kartly.store.models import ActorType, AuditLog
from kartly.tools import Tool, ToolEnv, ToolError, default_tools
from kartly.tools.base import CallContext

SessionFactory = Callable[[], Session]
MAX_TOOL_NAME_LOG = 60


@dataclass(frozen=True)
class ToolResult:
    """What a tool call returns. ``message`` is safe to show the model."""

    ok: bool
    tool: str
    data: dict[str, Any] | None = None
    error_code: str | None = None
    message: str | None = None
    latency_ms: int = 0

    def to_model_content(self) -> str:
        """The compact JSON string handed back to the model as the tool result."""
        payload: dict[str, Any]
        if self.ok:
            payload = {"ok": True, "data": self.data}
        else:
            payload = {"ok": False, "error": {"code": self.error_code, "message": self.message}}
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)


@dataclass
class _Outcome:
    result: ToolResult
    outcome: str  # succeeded | rejected | failed
    target_id: str | None = None
    reason: str | None = None
    security_event: str | None = None
    error_type: str | None = None
    error: str | None = None


def compact_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's ``title`` noise from a JSON schema to save prompt tokens."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, value in node.items():
                if key == "title" and isinstance(value, str):
                    continue
                if key == "properties" and isinstance(value, dict):
                    out[key] = {name: walk(prop) for name, prop in value.items()}
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    compacted = walk(schema)
    assert isinstance(compacted, dict)
    return compacted


def _describe_validation_error(exc: ValidationError) -> str:
    """A short, model-readable message that never echoes the offending input back."""
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    parts = []
    for error in errors[:5]:
        location = ".".join(clean_str(str(p), 40) for p in error["loc"]) or "arguments"
        parts.append(f"{location}: {clean_str(error['msg'], 120)}")
    return "Invalid arguments. " + "; ".join(parts)


class ToolGateway:
    def __init__(
        self,
        session_factory: SessionFactory,
        tools: Iterable[Tool] | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        tool_list = tuple(default_tools() if tools is None else tools)
        names = [tool.name for tool in tool_list]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {sorted(names)}")
        self._tools = {tool.name: tool for tool in tool_list}
        self._session_factory = session_factory
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools)

    def tool_specs(self, allowed: Collection[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI-style function specs for the model, limited to ``allowed`` (default: all).

        Order is stable (registration order) so the tool block can be prompt-cached.
        """
        if allowed is not None:
            unknown = sorted(set(allowed) - set(self._tools))
            if unknown:
                raise ValueError(f"unknown tools in allowed list: {unknown}")
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": compact_schema(tool.args_model.model_json_schema()),
                },
            }
            for name, tool in self._tools.items()
            if allowed is None or name in allowed
        ]

    def call(
        self,
        ctx: CallContext,
        tool_name: str,
        raw_args: Any,
        *,
        allowed_tools: Collection[str] | None = None,
    ) -> ToolResult:
        """Run one tool call. Raises only if the audit record cannot be written."""
        started = time.perf_counter()
        now = self._clock()
        with self._session_factory() as session:
            outcome = self._run(session, ctx, tool_name, raw_args, allowed_tools, now)
            latency_ms = int((time.perf_counter() - started) * 1000)
            result = replace(outcome.result, latency_ms=latency_ms)
            session.add(self._audit_row(ctx, tool_name, raw_args, outcome, latency_ms))
            session.commit()
        return result

    # ------------------------------------------------------------------ internals

    def _run(
        self,
        session: Session,
        ctx: CallContext,
        tool_name: str,
        raw_args: Any,
        allowed: Collection[str] | None,
        now: datetime,
    ) -> _Outcome:
        name = clean_str(tool_name, MAX_TOOL_NAME_LOG)
        tool = self._tools.get(tool_name)
        available = [n for n in self._tools if allowed is None or n in allowed]

        if tool is None or tool.name not in available:
            exists = tool is not None
            return _Outcome(
                result=ToolResult(
                    ok=False,
                    tool=name,
                    error_code="tool_unavailable",
                    message=(
                        f"Tool '{name}' is not available. "
                        f"Available tools: {', '.join(available) or 'none'}."
                    ),
                ),
                outcome="rejected",
                reason="not_allowed_for_request" if exists else "unknown_tool",
                security_event="disallowed_tool_call" if exists else None,
            )

        if not isinstance(raw_args, dict):
            return _Outcome(
                result=ToolResult(
                    ok=False,
                    tool=name,
                    error_code="invalid_arguments",
                    message="Invalid arguments. Arguments must be a JSON object.",
                ),
                outcome="rejected",
                reason="arguments_not_an_object",
            )

        # Identity comes from CallContext only. A model supplying one is worth an alert.
        security_event = "identity_argument_supplied" if "customer_id" in raw_args else None
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return _Outcome(
                result=ToolResult(
                    ok=False,
                    tool=name,
                    error_code="invalid_arguments",
                    message=_describe_validation_error(exc),
                ),
                outcome="rejected",
                reason="validation_failed",
                security_event=security_event,
            )

        order_id = getattr(args, "order_id", None)
        target_id = order_id if isinstance(order_id, str) else None

        try:
            with session.begin_nested():  # SAVEPOINT: a failing tool cannot leave partial writes
                data = tool.handler(ToolEnv(session=session, ctx=ctx, now=now), args)
        except ToolError as exc:
            return _Outcome(
                result=ToolResult(ok=False, tool=name, error_code=exc.code, message=exc.message),
                outcome="rejected",
                target_id=target_id,
                reason=exc.reason,
                security_event=exc.security_event,
            )
        except Exception as exc:
            return _Outcome(
                result=ToolResult(
                    ok=False,
                    tool=name,
                    error_code="internal_error",
                    message="The tool failed unexpectedly.",
                ),
                outcome="failed",
                target_id=target_id,
                error_type=type(exc).__name__,
                error=clean_str(str(exc)),
            )
        return _Outcome(
            result=ToolResult(ok=True, tool=name, data=data),
            outcome="succeeded",
            target_id=target_id,
        )

    @staticmethod
    def _audit_row(
        ctx: CallContext, tool_name: str, raw_args: Any, outcome: _Outcome, latency_ms: int
    ) -> AuditLog:
        details = {
            "tool": clean_str(tool_name, MAX_TOOL_NAME_LOG),
            "outcome": outcome.outcome,
            "error_code": outcome.result.error_code,
            "reason": outcome.reason,
            "security_event": outcome.security_event,
            "error_type": outcome.error_type,
            "error": outcome.error,
            "latency_ms": latency_ms,
            "args": safe_json(raw_args),
        }
        return AuditLog(
            actor_type=ActorType.AGENT,
            actor_id=f"run:{ctx.run_id}",
            action=f"tool.{outcome.outcome}",
            ticket_id=ctx.ticket_id,
            run_id=ctx.run_id,
            target_type="order" if outcome.target_id else None,
            target_id=outcome.target_id,
            details={key: value for key, value in details.items() if value is not None},
        )
