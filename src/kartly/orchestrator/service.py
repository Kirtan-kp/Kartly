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
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from kartly.gateway import ToolGateway
from kartly.llm import ChatClient
from kartly.orchestrator.budget import RunBudget
from kartly.orchestrator.loop import AgentLoop, LoopResult, StepRecord, StopReason
from kartly.orchestrator.prompts import PROMPT_VERSION, build_messages
from kartly.sanitize import clean_str
from kartly.store.models import (
    Customer,
    Message,
    MessageRole,
    Run,
    RunMode,
    RunStatus,
    RunStep,
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
    ) -> None:
        self._sessions = session_factory
        self._model_tier = model_tier
        self._loop = AgentLoop(client, gateway, budget=budget, allowed_tools=allowed_tools)

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

    # -------------------------------------------------------------------------- run

    def run(self, ticket_id: uuid.UUID) -> RunSummary:
        run_id, customer_id, thread = self._open_run(ticket_id)
        ctx = CallContext(ticket_id=ticket_id, run_id=run_id, customer_id=customer_id)
        try:
            result = self._loop.run(
                ctx,
                build_messages(thread),
                on_step=lambda step: self._write_step(run_id, step),
            )
        except Exception as exc:
            self._close_after_crash(run_id, ticket_id, exc)
            raise
        return self._close_run(run_id, ticket_id, result)

    # -------------------------------------------------------------------- internals

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
                prompt_version=PROMPT_VERSION,
            )
            ticket.status = TicketStatus.RESOLVING
            session.add(run)
            session.commit()
            return run.id, ticket.customer_id, thread

    def _write_step(self, run_id: uuid.UUID, step: StepRecord) -> None:
        """One short transaction per step, so a crash leaves every earlier step in place."""
        with self._sessions() as session:
            session.add(
                RunStep(
                    run_id=run_id,
                    idx=step.idx,
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

    def _close_run(self, run_id: uuid.UUID, ticket_id: uuid.UUID, result: LoopResult) -> RunSummary:
        with self._sessions() as session:
            run = session.get(Run, run_id)
            ticket = session.get(Ticket, ticket_id)
            assert run is not None and ticket is not None
            run.tokens_in = result.tokens_in
            run.tokens_out = result.tokens_out
            run.shadow_cost_usd = result.cost_usd
            run.finished_at = datetime.now(UTC)
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
                run.status = RunStatus.FAILED
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
