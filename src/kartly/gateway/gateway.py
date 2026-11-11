"""The Tool Gateway: the only way the agent can touch the store.

The model proposes a tool call; this module decides what actually happens. For every call it:

1. rejects tools that do not exist or are not allowed for this request,
2. validates the arguments strictly (unknown arguments, including ``customer_id``, are errors),
3. runs the tool as the *authenticated* customer from ``CallContext``, never from model output,
4. isolates the tool in a SAVEPOINT, so a failure undoes its writes,
5. writes one audit_log row, in the same transaction, whatever the outcome.

Tools that change the store (risk tier 1 and above) take a longer path, set per tier by a mode:
``off`` (the tool does not exist for the model), ``shadow`` (the call is checked and recorded as a
proposal that a human must approve; nothing changes) or ``auto`` (it runs now, unless the tool
says this call needs a human, as a large refund will). Every write has a server-made idempotency
key, so a repeated call gets the first call's result and cannot run twice.

One call is one transaction. If the audit row cannot be written the whole call is rolled back
and the exception propagates: an action without an audit record must not happen (fail closed).
Expected failures never raise; they come back as a ToolResult the model can read and react to.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
import uuid
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from kartly.controls import Controls
from kartly.sanitize import clean_str, safe_json
from kartly.store.models import (
    Action,
    ActionStatus,
    ActorType,
    Approval,
    ApprovalStatus,
    AuditLog,
    Ticket,
    TicketStatus,
)
from kartly.tools import Tool, ToolEnv, ToolError, default_tools
from kartly.tools.base import CallContext, RiskTier
from kartly.tools.orders import utc

SessionFactory = Callable[[], Session]
MAX_TOOL_NAME_LOG = 60
APPROVAL_TTL = timedelta(hours=24)
MAX_NOTE_CHARS = 300


class Mode(enum.StrEnum):
    """How much a tier may do on its own (DESIGN.md section 5, rollout modes)."""

    OFF = "off"
    SHADOW = "shadow"
    AUTO = "auto"


# Safe by default: until someone chooses otherwise, writes are proposals a human must approve.
DEFAULT_MODES: dict[RiskTier, Mode] = {
    RiskTier.LOW_RISK_WRITE: Mode.SHADOW,
    RiskTier.MONEY: Mode.SHADOW,
}

AWAITING_NOTE = (
    "This has NOT been done yet. A human colleague must approve it first. Tell the customer it "
    "is waiting for review; do not say it is done."
)


def modes_from_settings(tier1: str, tier2: str) -> dict[RiskTier, Mode]:
    return {RiskTier.LOW_RISK_WRITE: Mode(tier1), RiskTier.MONEY: Mode(tier2)}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def idempotency_key(run_id: uuid.UUID, tool: str, args: Mapping[str, Any]) -> str:
    """Made by the server from the run, the tool and the validated arguments, never by the model."""
    digest = hashlib.sha256(canonical_json(args).encode()).hexdigest()[:40]
    return f"{run_id}:{tool}:{digest}"


def payload_hash(tool: str, args: Mapping[str, Any], customer_id: str) -> str:
    """Fingerprint of the exact action a human is asked to approve."""
    body = canonical_json({"tool": tool, "args": args, "customer_id": customer_id})
    return hashlib.sha256(body.encode()).hexdigest()


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


def _settle_ticket(session: Session, ticket: Ticket) -> None:
    """Move a ticket out of ``awaiting_approval`` once none of its requests is pending.

    All requests done and carried out: ``drafted`` (a human sends the reply, which can now say it
    is done). Any rejected, expired, refused or failed: ``escalated``, so a person follows up
    rather than the customer being told something happened that did not.
    """
    if ticket.status != TicketStatus.AWAITING_APPROVAL:
        return
    session.flush()
    rows = session.execute(
        select(Approval.status, Action.status)
        .join(Action, Approval.action_id == Action.id)
        .where(Approval.ticket_id == ticket.id)
    ).all()
    if any(approval is ApprovalStatus.PENDING for approval, _ in rows):
        return
    # A rejected or expired request leaves its action rejected, so the action status says it.
    failed = any(action in (ActionStatus.FAILED, ActionStatus.REJECTED) for _, action in rows)
    ticket.status = TicketStatus.ESCALATED if failed else TicketStatus.DRAFTED


@dataclass(frozen=True)
class Decision:
    """The result of a human approving or rejecting a proposed action."""

    ok: bool
    status: str  # executed | not_executed | rejected | expired | already_decided | refused
    message: str
    data: dict[str, Any] | None = None


@dataclass
class _Outcome:
    result: ToolResult
    outcome: str  # succeeded | rejected | failed
    target_id: str | None = None
    reason: str | None = None
    security_event: str | None = None
    error_type: str | None = None
    error: str | None = None
    action_id: uuid.UUID | None = None


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


def build_tool_specs(
    tools: Iterable[Tool], allowed: Collection[str] | None = None
) -> list[dict[str, Any]]:
    """OpenAI-style function specs for ``tools``, limited to ``allowed`` (default: all).

    Needs no database, so scripts and tests can use it directly. Order follows ``tools``.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": compact_schema(tool.args_model.model_json_schema()),
            },
        }
        for tool in tools
        if allowed is None or tool.name in allowed
    ]


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
        modes: Mapping[RiskTier, Mode] | None = None,
        approval_ttl: timedelta = APPROVAL_TTL,
        controls: Controls | None = None,
    ) -> None:
        tool_list = tuple(default_tools() if tools is None else tools)
        names = [tool.name for tool in tool_list]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate tool names: {sorted(names)}")
        self._tools = {tool.name: tool for tool in tool_list}
        self._session_factory = session_factory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._modes = {**DEFAULT_MODES, **(modes or {})}
        self._approval_ttl = approval_ttl
        self._controls = controls

    def mode_for(self, tool: Tool) -> Mode:
        """Reads are always on. A write follows the mode set for its tier (default: shadow)."""
        if not tool.is_write:
            return Mode.AUTO
        mode = self._modes.get(tool.tier, Mode.SHADOW)
        if tool.tier == RiskTier.NEVER_AUTONOMOUS and mode == Mode.AUTO:
            return Mode.SHADOW  # by definition a human decides; no setting can change that
        return mode

    @property
    def tool_names(self) -> list[str]:
        """Tools the model may be offered: every tool except writes whose tier is switched off."""
        return [name for name, tool in self._tools.items() if self.mode_for(tool) != Mode.OFF]

    def tool_specs(self, allowed: Collection[str] | None = None) -> list[dict[str, Any]]:
        """OpenAI-style function specs for the model, limited to ``allowed`` (default: all).

        Order is stable (registration order) so the tool block can be prompt-cached.
        """
        if allowed is not None:
            unknown = sorted(set(allowed) - set(self._tools))
            if unknown:
                raise ValueError(f"unknown tools in allowed list: {unknown}")
        on = self.tool_names
        return build_tool_specs(
            (t for t in self._tools.values() if t.name in on),
            on if allowed is None else [n for n in allowed if n in on],
        )

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
        available = [
            n
            for n, t in self._tools.items()
            if (allowed is None or n in allowed) and self.mode_for(t) != Mode.OFF
        ]

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

        if tool.is_write:
            return self._run_write(session, ctx, tool, args, target_id, security_event, now)

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

    # ------------------------------------------------------------------- approvals

    def decide(
        self,
        approval_id: uuid.UUID,
        *,
        approve: bool,
        actor: str,
        note: str | None = None,
    ) -> Decision:
        """A human approves or rejects a proposed action. Approving runs it, once, now.

        Before anything runs the stored action is hashed again and compared with the hash taken
        when it was proposed, so what runs is what was shown. The tool's own checks run again too,
        because the order may have changed since the proposal. Everything, including a refused
        decision, leaves an audit row in the same transaction.
        """
        now = self._clock()
        with self._session_factory() as session:
            decision = self._decide(session, approval_id, approve, actor, note, now)
            session.commit()
        return decision

    def _decide(
        self,
        session: Session,
        approval_id: uuid.UUID,
        approve: bool,
        actor: str,
        note: str | None,
        now: datetime,
    ) -> Decision:
        # Lock the request, so two reviewers deciding at once are handled one after the other and
        # the second sees "already decided" instead of running the action again.
        approval = session.get(Approval, approval_id, with_for_update=True, populate_existing=True)
        if approval is None:
            return Decision(False, "refused", "No such approval request.")
        action = session.get(Action, approval.action_id)
        ticket = session.get(Ticket, approval.ticket_id)
        assert action is not None and ticket is not None

        def audit(name: str, **details: Any) -> None:
            session.add(
                AuditLog(
                    actor_type=ActorType.HUMAN,
                    actor_id=clean_str(actor, 64),
                    action=f"approval.{name}",
                    ticket_id=approval.ticket_id,
                    run_id=action.run_id,
                    target_type="approval",
                    target_id=str(approval.id),
                    details={"action_id": str(action.id), "tool": action.tool_name, **details},
                )
            )

        def close(status: ApprovalStatus) -> None:
            approval.status = status
            approval.decided_by = clean_str(actor, 64)
            approval.decided_at = now
            approval.decision_note = clean_str(note, MAX_NOTE_CHARS) if note else None

        if approval.status != ApprovalStatus.PENDING:
            audit("repeat_refused", was=approval.status.value)
            return Decision(False, "already_decided", f"Already {approval.status.value}.")
        if approve and self._controls is not None and not self._controls.writes_on(session):
            audit("blocked_writes_frozen")  # the request stays pending; rejecting still works
            return Decision(
                False, "refused", "Writes are frozen, so nothing can be approved right now."
            )
        if now >= utc(approval.expires_at):
            close(ApprovalStatus.EXPIRED)
            action.status = ActionStatus.REJECTED
            action.result = {"error_code": "approval_expired", "message": "Approval expired."}
            audit("expired")
            _settle_ticket(session, ticket)
            return Decision(False, "expired", "This request expired before it was decided.")
        if not approve:
            close(ApprovalStatus.REJECTED)
            action.status = ActionStatus.REJECTED
            action.result = {"error_code": "approval_rejected", "message": "A human rejected it."}
            audit("rejected", note=clean_str(note, MAX_NOTE_CHARS) if note else None)
            _settle_ticket(session, ticket)
            return Decision(True, "rejected", "Rejected. Nothing was changed.")

        tool = self._tools.get(action.tool_name)
        if payload_hash(action.tool_name, action.args, ticket.customer_id) != approval.payload_hash:
            close(ApprovalStatus.REJECTED)
            action.status = ActionStatus.REJECTED
            action.result = {"error_code": "payload_changed", "message": "The action changed."}
            audit("payload_mismatch", security_event="approval_payload_mismatch")
            _settle_ticket(session, ticket)
            return Decision(False, "refused", "The action no longer matches what was proposed.")
        if tool is None:
            close(ApprovalStatus.REJECTED)
            action.status = ActionStatus.FAILED
            audit("tool_missing")
            _settle_ticket(session, ticket)
            return Decision(False, "refused", "That tool is no longer available.")

        args = tool.args_model.model_validate(action.args)
        ctx = CallContext(ticket_id=ticket.id, run_id=action.run_id, customer_id=ticket.customer_id)
        order_id = getattr(args, "order_id", None)
        outcome = self._apply(
            session, tool, action, ctx, args, now, order_id if isinstance(order_id, str) else None
        )
        close(ApprovalStatus.APPROVED)
        audit("approved", outcome=outcome.outcome, error_code=outcome.result.error_code)
        _settle_ticket(session, ticket)
        if outcome.result.ok:
            return Decision(True, "executed", "Done.", outcome.result.data)
        return Decision(False, "not_executed", outcome.result.message or "It could not be done.")

    def expire_stale(self) -> int:
        """Mark every pending request past its expiry as expired; returns how many.

        Nothing runs on a timer yet, so the API calls this before it lists requests. An expired
        request can no longer be approved (``decide`` also checks), and its ticket goes to a human.
        """
        now = self._clock()
        count = 0
        with self._session_factory() as session:
            pending = session.scalars(
                select(Approval).where(Approval.status == ApprovalStatus.PENDING).with_for_update()
            ).all()
            for approval in pending:
                if now < utc(approval.expires_at):
                    continue
                action = session.get(Action, approval.action_id)
                ticket = session.get(Ticket, approval.ticket_id)
                assert action is not None and ticket is not None
                approval.status = ApprovalStatus.EXPIRED
                approval.decided_at = now
                action.status = ActionStatus.REJECTED
                action.result = {"error_code": "approval_expired", "message": "Approval expired."}
                session.add(
                    AuditLog(
                        actor_type=ActorType.SYSTEM,
                        actor_id="gateway",
                        action="approval.expired",
                        ticket_id=approval.ticket_id,
                        run_id=action.run_id,
                        target_type="approval",
                        target_id=str(approval.id),
                        details={"action_id": str(action.id), "tool": action.tool_name},
                    )
                )
                _settle_ticket(session, ticket)
                count += 1
            session.commit()
        return count

    # --------------------------------------------------------------- the write path

    def _run_write(
        self,
        session: Session,
        ctx: CallContext,
        tool: Tool,
        args: BaseModel,
        target_id: str | None,
        security_event: str | None,
        now: datetime,
    ) -> _Outcome:
        name = tool.name
        canonical = args.model_dump(mode="json")
        key = idempotency_key(ctx.run_id, name, canonical)
        existing = session.scalars(select(Action).where(Action.idempotency_key == key)).first()
        if existing is not None:
            return self._replay(existing, name, target_id)

        env = ToolEnv(session=session, ctx=ctx, now=now)
        try:
            with session.begin_nested():  # a preview must not write, and the savepoint enforces it
                assert tool.preview is not None
                preview = tool.preview(env, args)
                needs = tool.needs_approval(env, args) if tool.needs_approval else None
        except ToolError as exc:
            return _Outcome(
                result=ToolResult(ok=False, tool=name, error_code=exc.code, message=exc.message),
                outcome="rejected",
                target_id=target_id,
                reason=exc.reason,
                security_event=exc.security_event or security_event,
            )
        except Exception as exc:
            return self._failed(name, target_id, exc)

        mode = self.mode_for(tool)
        frozen = (
            mode == Mode.AUTO
            and self._controls is not None
            and not self._controls.writes_on(session)
        )
        reason = "writes_frozen" if frozen else "shadow_mode" if mode == Mode.SHADOW else needs
        if reason is not None:
            return self._propose(
                session, ctx, tool, canonical, preview, reason, key, target_id, now
            )

        action = self._claim(session, ctx, name, canonical, key, ActionStatus.PENDING, None)
        if action is None:
            return self._replay(self._existing(session, key), name, target_id)
        outcome = self._apply(session, tool, action, ctx, args, now, target_id)
        outcome.action_id = action.id
        return outcome

    @staticmethod
    def _existing(session: Session, key: str) -> Action:
        return session.scalars(select(Action).where(Action.idempotency_key == key)).one()

    @staticmethod
    def _claim(
        session: Session,
        ctx: CallContext,
        tool: str,
        args: dict[str, Any],
        key: str,
        status: ActionStatus,
        result: dict[str, Any] | None,
    ) -> Action | None:
        """Insert the action row. None means another call with the same key won the race."""
        action = Action(
            run_id=ctx.run_id,
            ticket_id=ctx.ticket_id,
            tool_name=tool,
            args=args,
            idempotency_key=key,
            status=status,
            result=result,
        )
        try:
            with session.begin_nested():
                session.add(action)
                session.flush()
        except IntegrityError:
            if session.scalars(select(Action.id).where(Action.idempotency_key == key)).first():
                return None
            raise  # a different integrity problem, for example a missing run: do not hide it
        return action

    def _propose(
        self,
        session: Session,
        ctx: CallContext,
        tool: Tool,
        canonical: dict[str, Any],
        preview: dict[str, Any],
        reason: str,
        key: str,
        target_id: str | None,
        now: datetime,
    ) -> _Outcome:
        action = self._claim(
            session,
            ctx,
            tool.name,
            canonical,
            key,
            ActionStatus.AWAITING_APPROVAL,
            {"preview": preview, "reason": reason},
        )
        if action is None:
            return self._replay(self._existing(session, key), tool.name, target_id)
        approval = Approval(
            action_id=action.id,
            ticket_id=ctx.ticket_id,
            payload_hash=payload_hash(tool.name, canonical, ctx.customer_id),
            reason=reason,
            expires_at=now + self._approval_ttl,
        )
        session.add(approval)
        session.flush()
        data = {
            "status": "awaiting_approval",
            "action_id": str(action.id),
            "approval_id": str(approval.id),
            "would_do": preview,
            "note": AWAITING_NOTE,
        }
        return _Outcome(
            result=ToolResult(ok=True, tool=tool.name, data=data),
            outcome="proposed",
            target_id=target_id,
            reason=reason,
            action_id=action.id,
        )

    @staticmethod
    def _failed(name: str, target_id: str | None, exc: Exception) -> _Outcome:
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

    def _apply(
        self,
        session: Session,
        tool: Tool,
        action: Action,
        ctx: CallContext,
        args: BaseModel,
        now: datetime,
        target_id: str | None,
    ) -> _Outcome:
        """Run the handler for a stored action and record how it ended on the action row."""
        name = tool.name
        try:
            with session.begin_nested():
                data = tool.handler(ToolEnv(session=session, ctx=ctx, now=now), args)
        except ToolError as exc:
            action.status = ActionStatus.REJECTED
            action.result = {"error_code": exc.code, "message": exc.message}
            action.error = clean_str(f"{exc.code}: {exc.message}")
            session.flush()
            return _Outcome(
                result=ToolResult(ok=False, tool=name, error_code=exc.code, message=exc.message),
                outcome="rejected",
                target_id=target_id,
                reason=exc.reason,
                security_event=exc.security_event,
            )
        except Exception as exc:
            action.status = ActionStatus.FAILED
            action.error = clean_str(f"{type(exc).__name__}: {exc}")
            session.flush()
            return self._failed(name, target_id, exc)
        action.status = ActionStatus.SUCCEEDED
        action.result = json.loads(canonical_json(data))
        session.flush()
        return _Outcome(
            result=ToolResult(ok=True, tool=name, data=data),
            outcome="succeeded",
            target_id=target_id,
        )

    @staticmethod
    def _replay(action: Action, name: str, target_id: str | None) -> _Outcome:
        """A repeated call gets the first call's result and does nothing."""
        stored = action.result or {}
        if action.status == ActionStatus.SUCCEEDED:
            result = ToolResult(ok=True, tool=name, data={**stored, "replayed": True})
        elif action.status == ActionStatus.AWAITING_APPROVAL:
            data = {
                "status": "awaiting_approval",
                "action_id": str(action.id),
                "would_do": stored.get("preview"),
                "note": AWAITING_NOTE,
                "replayed": True,
            }
            result = ToolResult(ok=True, tool=name, data=data)
        elif action.status == ActionStatus.PENDING:
            result = ToolResult(
                ok=False,
                tool=name,
                error_code="action_in_progress",
                message="This exact action is already being processed.",
            )
        else:
            result = ToolResult(
                ok=False,
                tool=name,
                error_code=str(stored.get("error_code", "internal_error")),
                message=str(stored.get("message", "This action already failed.")),
            )
        return _Outcome(result=result, outcome="replayed", target_id=target_id, action_id=action.id)

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
            "action_id": str(outcome.action_id) if outcome.action_id else None,
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
