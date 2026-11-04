"""Tickets in, stored runs out: wires the agent loop to the database.

A ticket is created with its first customer message. ``TicketRunner.run`` then opens a run,
runs the bounded loop with every step written to ``run_steps`` as it happens, and closes the run:

- the loop produced an answer  -> run ``succeeded``/``drafted``, ticket ``drafted`` (shadow mode:
  a human reviews and sends; the draft is stored as an agent message),
- the loop hit a limit or an error -> run ``failed``/``escalated``, ticket ``escalated``.

The stages before and after the loop (screening, routing, validation) arrive in later weeks, so
the ticket goes straight from ``received`` to ``resolving``.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kartly.gateway import ToolGateway
from kartly.guards import ScreenResult, screen_text
from kartly.guards.grounding import GroundingResult, check_grounding
from kartly.llm import ChatClient
from kartly.orchestrator.budget import RunBudget
from kartly.orchestrator.loop import AgentLoop, LoopResult, StepRecord, StopReason
from kartly.orchestrator.prompts import DEFAULT_POLICY_PROMPT, build_messages, prompt_version
from kartly.routing import (
    DEFAULT_VERSION,
    ClassifierResult,
    RuleVerdict,
    classify,
    decide_route,
    route_by_rules,
)
from kartly.sanitize import clean_str
from kartly.store.models import (
    ActorType,
    AuditLog,
    Customer,
    Message,
    MessageRole,
    Run,
    RunMode,
    RunStatus,
    RunStep,
    StepKind,
    Ticket,
    TicketStatus,
)
from kartly.tools.base import CallContext

SessionFactory = Callable[[], Session]
MAX_ANSWER_CHARS = 8000
MAX_ERROR_CHARS = 500


class UnknownCustomerError(LookupError):
    """The ticket names a customer that does not exist."""


@dataclass(frozen=True)
class RunSummary:
    run_id: uuid.UUID
    ticket_id: uuid.UUID
    outcome: str  # drafted | escalated
    stop_reason: StopReason
    detail: str
    answer: str | None
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal


class TicketRunner:
    def __init__(
        self,
        session_factory: SessionFactory,
        client: ChatClient,
        gateway: ToolGateway,
        *,
        budget: RunBudget | None = None,
        model_tier: str | None = None,
        allowed_tools: Collection[str] | None = None,
        screen: Callable[[str], ScreenResult] | None = screen_text,
        rules: Callable[[str], RuleVerdict] | None = route_by_rules,
        router: ChatClient | None = None,
        router_version: str = DEFAULT_VERSION,
        policy_prompt: str = DEFAULT_POLICY_PROMPT,
        grounding: Callable[[str, Sequence[Mapping[str, Any]], str], GroundingResult] | None = (
            check_grounding
        ),
    ) -> None:
        """``screen`` checks customer text for prompt injection before any model call (default
        on); a suspicious ticket goes straight to a human. ``rules`` is the same for Tier 3
        topics (legal threats, fraud, safety...). ``None`` turns either off. ``router`` is a
        model that labels the ticket (intent, tier) after the rules; a Tier 3 label sends it to a
        human. Off by default. If the router fails the ticket carries on to the resolver (it is
        recorded in the trace)."""
        self._sessions = session_factory
        self._screen = screen
        self._rules = rules
        self._router = router
        self._router_version = router_version
        self._model_tier = model_tier
        self._policy_prompt = policy_prompt
        self._grounding = grounding
        self._loop = AgentLoop(client, gateway, budget=budget, allowed_tools=allowed_tools)
        # The policy rules go into the prompt only when the model can actually call the tool.
        self._policy = any(spec["function"]["name"] == "search_policy" for spec in self._loop.specs)

    # ---------------------------------------------------------------------- tickets

    def create_ticket(
        self, customer_id: str, text: str, *, external_id: str | None = None
    ) -> uuid.UUID:
        with self._sessions() as session:
            if session.get(Customer, customer_id) is None:
                raise UnknownCustomerError(f"no customer {clean_str(customer_id, 40)!r}")
            ticket = Ticket(customer_id=customer_id, external_id=external_id)
            ticket.messages = [
                Message(
                    role=MessageRole.CUSTOMER,
                    content=clean_str(text, MAX_ANSWER_CHARS),
                    created_at=datetime.now(UTC),
                )
            ]
            session.add(ticket)
            session.commit()
            return ticket.id

    def discard_unstarted_ticket(self, ticket_id: uuid.UUID) -> bool:
        """Delete a ticket no run has touched (used when the system is too busy to start it)."""
        with self._sessions() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None or ticket.status is not TicketStatus.RECEIVED or ticket.runs:
                return False
            session.delete(ticket)
            session.commit()
            return True

    def find_ticket_by_external_id(self, external_id: str) -> uuid.UUID | None:
        with self._sessions() as session:
            return session.scalar(select(Ticket.id).where(Ticket.external_id == external_id))

    # -------------------------------------------------------------------------- run

    def run(self, ticket_id: uuid.UUID) -> RunSummary:
        run_id, customer_id, thread = self._open_run(ticket_id)
        verdict = self._screen_thread(thread)
        if verdict.suspicious:
            return self._close_screened(run_id, ticket_id, verdict)
        tier3 = self._route_thread(thread)
        if tier3.tier >= 3:
            return self._close_routed(run_id, ticket_id, tier3)
        offset = 0
        pre: ClassifierResult | None = None
        if self._router is not None:
            pre = self._classify_thread(run_id, thread)
            offset = 1
            decision = decide_route(RuleVerdict(0), pre)
            if decision.tier >= 3:
                return self._close_routed(
                    run_id,
                    ticket_id,
                    RuleVerdict(3, ()),
                    source="classifier",
                    intent=decision.intent,
                    pre=pre,
                    idx=1,
                )
        ctx = CallContext(ticket_id=ticket_id, run_id=run_id, customer_id=customer_id)
        try:
            result = self._loop.run(
                ctx,
                build_messages(thread, policy=self._policy, policy_prompt=self._policy_prompt),
                on_step=lambda step: self._write_step(run_id, step, offset),
            )
        except Exception as exc:
            self._close_after_crash(run_id, ticket_id, exc)
            raise
        if pre is not None:  # the label call is part of this run's cost
            result = replace(
                result,
                llm_calls=result.llm_calls + 1,
                tokens_in=result.tokens_in + pre.tokens_in,
                tokens_out=result.tokens_out + pre.tokens_out,
                cost_usd=result.cost_usd + pre.cost_usd,
            )
        check = None
        if self._grounding is not None and result.answered and result.answer is not None:
            customer_text = "\n".join(c for role, c in thread if role is MessageRole.CUSTOMER)
            verdict_g = self._grounding(result.answer, result.messages, customer_text)
            check = verdict_g if verdict_g.applicable else None
        return self._close_run(run_id, ticket_id, result, check, idx=len(result.steps) + offset)

    # -------------------------------------------------------------------- internals

    def _screen_thread(self, thread: list[tuple[MessageRole, str]]) -> ScreenResult:
        if self._screen is None:
            return ScreenResult(False)
        rules: list[str] = []
        for role, text in thread:
            if role is MessageRole.CUSTOMER:
                rules.extend(r for r in self._screen(text).rules if r not in rules)
        return ScreenResult(bool(rules), tuple(rules))

    def _classify_thread(
        self, run_id: uuid.UUID, thread: list[tuple[MessageRole, str]]
    ) -> ClassifierResult:
        """Label the customer's text with the router model and record it as trace step 0."""
        assert self._router is not None
        text = "\n\n".join(content for role, content in thread if role is MessageRole.CUSTOMER)
        result = classify(self._router, text, self._router_version)
        c = result.classification
        output: dict[str, object] = (
            {
                "intent": c.intent,
                "tier": c.effective_tier,
                "complexity": c.complexity,
                "sentiment": c.sentiment,
                "confidence": c.confidence,
            }
            if c is not None
            else {"failed": True, "error": result.error}
        )
        with self._sessions() as session:
            session.add(
                RunStep(
                    run_id=run_id,
                    idx=0,
                    kind=StepKind.STAGE,
                    name="classify",
                    input={"version": self._router_version},
                    output=output,
                    model=self._router.name,
                    tokens_in=result.tokens_in,
                    tokens_out=result.tokens_out,
                    latency_ms=result.latency_ms,
                    shadow_cost_usd=result.cost_usd,
                    error=clean_str(result.error, 200) if result.error else None,
                )
            )
            session.commit()
        return result

    def _route_thread(self, thread: list[tuple[MessageRole, str]]) -> RuleVerdict:
        if self._rules is None:
            return RuleVerdict(0)
        categories: list[str] = []
        for role, text in thread:
            if role is MessageRole.CUSTOMER:
                categories.extend(c for c in self._rules(text).categories if c not in categories)
        return RuleVerdict(3 if categories else 0, tuple(categories))

    def _close_screened(
        self, run_id: uuid.UUID, ticket_id: uuid.UUID, verdict: ScreenResult
    ) -> RunSummary:
        """Hand a suspicious ticket to a human. The run succeeded at its job: it did not fail."""
        detail = f"suspected prompt injection ({', '.join(verdict.rules)})"
        return self._close_held(
            run_id,
            ticket_id,
            stage="screen",
            stop=StopReason.SCREENED,
            detail=detail,
            step_input={"rules": list(verdict.rules)},
            audit_action="screen.blocked",
            audit_actor="screen",
            audit_details={
                "security_event": "prompt_injection_suspected",
                "rules": list(verdict.rules),
            },
        )

    def _close_routed(
        self,
        run_id: uuid.UUID,
        ticket_id: uuid.UUID,
        verdict: RuleVerdict,
        *,
        source: str = "rules",
        intent: str | None = None,
        pre: ClassifierResult | None = None,
        idx: int = 0,
    ) -> RunSummary:
        """Tier 3 is never autonomous: a human takes the ticket, no resolver model is involved."""
        why = ", ".join(verdict.categories) if source == "rules" else f"classifier: {intent}"
        detail = f"tier 3 by {source} ({why})"
        facts: dict[str, object] = {
            "tier": 3,
            "categories": list(verdict.categories),
            "source": source,
        }
        if intent is not None:
            facts["intent"] = intent
        return self._close_held(
            run_id,
            ticket_id,
            stage="route",
            stop=StopReason.ROUTED_TIER3,
            detail=detail,
            step_input=facts,
            audit_action="route.escalated",
            audit_actor="router",
            audit_details=facts,
            pre=pre,
            idx=idx,
        )

    def _close_held(
        self,
        run_id: uuid.UUID,
        ticket_id: uuid.UUID,
        *,
        stage: str,
        stop: StopReason,
        detail: str,
        step_input: dict[str, object],
        audit_action: str,
        audit_actor: str,
        audit_details: dict[str, object],
        pre: ClassifierResult | None = None,
        idx: int = 0,
    ) -> RunSummary:
        """Close a run that a stage before the resolver sent to a human: one trace step, one
        audit row, ticket escalated, no resolver or tool calls (``pre`` is the label call)."""
        with self._sessions() as session:
            run = session.get(Run, run_id)
            ticket = session.get(Ticket, ticket_id)
            assert run is not None and ticket is not None
            session.add(
                RunStep(
                    run_id=run_id,
                    idx=idx,
                    kind=StepKind.STAGE,
                    name=stage,
                    input=step_input,
                    output={"held": True},
                )
            )
            session.add(
                AuditLog(
                    actor_type=ActorType.SYSTEM,
                    actor_id=audit_actor,
                    action=audit_action,
                    ticket_id=ticket_id,
                    run_id=run_id,
                    details=audit_details,
                )
            )
            run.status = RunStatus.SUCCEEDED
            run.outcome = "escalated"
            run.error = clean_str(f"{stop}: {detail}", MAX_ERROR_CHARS)
            if pre is not None:
                run.tokens_in, run.tokens_out = pre.tokens_in, pre.tokens_out
                run.shadow_cost_usd = pre.cost_usd
            run.finished_at = datetime.now(UTC)
            ticket.status = TicketStatus.ESCALATED
            session.commit()
        return RunSummary(
            run_id=run_id,
            ticket_id=ticket_id,
            outcome="escalated",
            stop_reason=stop,
            detail=detail,
            answer=None,
            llm_calls=1 if pre is not None else 0,
            tool_calls=0,
            tokens_in=pre.tokens_in if pre else 0,
            tokens_out=pre.tokens_out if pre else 0,
            cost_usd=pre.cost_usd if pre else Decimal(0),
        )

    def _open_run(
        self, ticket_id: uuid.UUID
    ) -> tuple[uuid.UUID, str, list[tuple[MessageRole, str]]]:
        with self._sessions() as session:
            ticket = session.get(Ticket, ticket_id)
            if ticket is None:
                raise LookupError(f"no ticket {ticket_id}")
            thread = [
                (m.role, m.content)
                for m in session.scalars(
                    select(Message)
                    .where(Message.ticket_id == ticket_id)
                    .order_by(Message.created_at, Message.id)
                )
            ]
            run = Run(
                ticket_id=ticket_id,
                mode=RunMode.SHADOW,
                model_tier=self._model_tier,
                prompt_version=prompt_version(self._policy, self._policy_prompt),
            )
            ticket.status = TicketStatus.RESOLVING
            session.add(run)
            session.commit()
            return run.id, ticket.customer_id, thread

    def _write_step(self, run_id: uuid.UUID, step: StepRecord, offset: int = 0) -> None:
        """One short transaction per step, so a crash leaves every earlier step in place."""
        with self._sessions() as session:
            session.add(
                RunStep(
                    run_id=run_id,
                    idx=step.idx + offset,
                    kind=step.kind,
                    name=step.name,
                    input=step.input,
                    output=step.output,
                    model=step.model,
                    tokens_in=step.tokens_in,
                    tokens_out=step.tokens_out,
                    latency_ms=step.latency_ms,
                    shadow_cost_usd=step.cost_usd,
                    error=step.error,
                )
            )
            session.commit()

    def _close_run(
        self,
        run_id: uuid.UUID,
        ticket_id: uuid.UUID,
        result: LoopResult,
        check: GroundingResult | None = None,
        idx: int = 0,
    ) -> RunSummary:
        """``check`` is the grounding verdict for a policy reply: a clean one adds a trace step and
        stores the reply without its citations; a failed one holds the reply for a human."""
        held = check is not None and not check.ok
        if held:
            assert check is not None
            result = replace(
                result,
                stop_reason=StopReason.UNGROUNDED,
                detail="; ".join(check.problems),
                answer=None,
            )
        elif check is not None and result.answer is not None:
            result = replace(result, answer=check.text)
        with self._sessions() as session:
            run = session.get(Run, run_id)
            ticket = session.get(Ticket, ticket_id)
            assert run is not None and ticket is not None
            run.tokens_in = result.tokens_in
            run.tokens_out = result.tokens_out
            run.shadow_cost_usd = result.cost_usd
            run.finished_at = datetime.now(UTC)
            if check is not None:
                session.add(
                    RunStep(
                        run_id=run_id,
                        idx=idx,
                        kind=StepKind.STAGE,
                        name="grounding",
                        input={"citations": list(check.citations)},
                        output={"held": held, "problems": list(check.problems)},
                    )
                )
            if held:
                assert check is not None
                session.add(
                    AuditLog(
                        actor_type=ActorType.SYSTEM,
                        actor_id="grounding",
                        action="guard.ungrounded_reply",
                        ticket_id=ticket_id,
                        run_id=run_id,
                        details={
                            "problems": [clean_str(p, 200) for p in check.problems],
                            "citations": list(check.citations),
                        },
                    )
                )
            if result.answered and result.answer is not None:
                outcome = "drafted"
                run.status = RunStatus.SUCCEEDED
                ticket.status = TicketStatus.DRAFTED
                session.add(
                    Message(
                        ticket_id=ticket_id,
                        role=MessageRole.AGENT,
                        content=clean_str(result.answer, MAX_ANSWER_CHARS),
                        created_at=datetime.now(UTC),
                    )
                )
            else:
                outcome = "escalated"
                run.status = RunStatus.SUCCEEDED if held else RunStatus.FAILED
                run.error = clean_str(f"{result.stop_reason}: {result.detail}", MAX_ERROR_CHARS)
                ticket.status = TicketStatus.ESCALATED
            run.outcome = outcome
            session.commit()
        return RunSummary(
            run_id=run_id,
            ticket_id=ticket_id,
            outcome=outcome,
            stop_reason=result.stop_reason,
            detail=result.detail,
            answer=result.answer,
            llm_calls=result.llm_calls,
            tool_calls=result.tool_calls,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
        )

    def _close_after_crash(self, run_id: uuid.UUID, ticket_id: uuid.UUID, exc: Exception) -> None:
        """Best effort: never leave a run 'running' forever. The original error still propagates."""
        with contextlib.suppress(Exception), self._sessions() as session:
            run = session.get(Run, run_id)
            ticket = session.get(Ticket, ticket_id)
            if run is not None:
                run.status = RunStatus.FAILED
                run.outcome = "failed"
                run.error = clean_str(f"{type(exc).__name__}: {exc}", MAX_ERROR_CHARS)
                run.finished_at = datetime.now(UTC)
            if ticket is not None:
                ticket.status = TicketStatus.FAILED
            session.commit()
