"""The HTTP API (DESIGN.md section 8).

    POST /v1/tickets            create a ticket and start an agent run (202; ?wait=true waits)
    GET  /v1/tickets/{id}       status, messages (the draft reply is the last agent message), runs
    GET  /v1/runs/{id}/trace    every model and tool call of a run, plus the gateway's audit rows
    GET  /healthz               liveness and a database check (no key needed)

Security stance: every /v1 route needs the API key. The ``customer_id`` on a ticket is asserted by
the caller (the storefront backend that authenticated the customer), stored once, and from then
on is the only identity a tool call can use. Nothing the model or the customer's text says can
change it. Ticket text is data, never instructions.
"""

from __future__ import annotations

import hmac
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from kartly.api.executor import Busy, RunExecutor
from kartly.api.schemas import (
    Health,
    MessageOut,
    RunOut,
    TicketCreate,
    TicketCreated,
    TicketOut,
    TraceOut,
)
from kartly.orchestrator import TicketRunner, UnknownCustomerError, load_trace
from kartly.sanitize import clean_str
from kartly.store.models import Ticket

SessionFactory = Callable[[], Session]


@dataclass
class ApiState:
    session_factory: SessionFactory
    runner: TicketRunner
    executor: RunExecutor
    api_key: str
    wait_timeout_s: float = 150.0  # a little over the run's own 120 s limit


def _state(request: Request) -> ApiState:
    state = request.app.state.kartly
    assert isinstance(state, ApiState)
    return state


StateDep = Annotated[ApiState, Depends(_state)]


def require_api_key(state: StateDep, x_api_key: Annotated[str | None, Header()] = None) -> None:
    supplied = (x_api_key or "").encode()
    # Same answer for a missing and a wrong key; compare_digest avoids timing differences.
    if not x_api_key or not hmac.compare_digest(supplied, state.api_key.encode()):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _db(state: StateDep) -> Iterator[Session]:
    with state.session_factory() as session:
        yield session


DbDep = Annotated[Session, Depends(_db)]


def create_app(state: ApiState, *, on_shutdown: Callable[[], None] | None = None) -> FastAPI:
    if not state.api_key or len(state.api_key) < 8:
        raise ValueError("the API key must be at least 8 characters (set KARTLY_API_KEY)")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        state.executor.shutdown()  # waits for runs in progress
        if on_shutdown is not None:
            on_shutdown()

    app = FastAPI(title="Kartly Support Agent", version="0.1.0", lifespan=lifespan)
    app.state.kartly = state

    @app.exception_handler(RequestValidationError)
    def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 that names the field and the rule but never echoes the input back: the default
        handler repeats the offending value, which can be huge or not even encodable (a lone
        surrogate in a JSON string made it crash with a 500)."""
        problems = [
            {
                "loc": [
                    part if isinstance(part, int) else clean_str(str(part), 60) for part in e["loc"]
                ],
                "msg": clean_str(str(e["msg"]), 200),
                "type": clean_str(str(e["type"]), 60),
            }
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": problems})

    @app.get("/healthz", response_model=Health, tags=["ops"])
    def healthz(response: Response, db: DbDep) -> Health:
        try:
            db.execute(text("SELECT 1"))
        except SQLAlchemyError:
            response.status_code = 503
            return Health(status="database unavailable")
        return Health(status="ok")

    @app.post(
        "/v1/tickets",
        response_model=TicketCreated,
        status_code=202,
        dependencies=[Depends(require_api_key)],
        tags=["tickets"],
        responses={
            200: {"description": "The run finished (wait=true) or the ticket already existed."},
            404: {"description": "Unknown customer."},
            503: {"description": "Too many runs in progress; retry shortly."},
        },
    )
    def create_ticket(
        body: TicketCreate, response: Response, state: StateDep, wait: bool = False
    ) -> TicketCreated:
        runner = state.runner
        if body.external_id is not None:
            earlier = runner.find_ticket_by_external_id(body.external_id)
            if earlier is not None:
                return _duplicate(response, state, earlier)
        try:
            ticket_id = runner.create_ticket(
                body.customer_id, body.message, external_id=body.external_id
            )
        except UnknownCustomerError:
            raise HTTPException(status_code=404, detail="unknown customer") from None
        except IntegrityError:  # two requests with the same external_id raced; the other won
            earlier = (
                runner.find_ticket_by_external_id(body.external_id) if body.external_id else None
            )
            if earlier is None:
                raise
            return _duplicate(response, state, earlier)

        try:
            future = state.executor.submit(lambda: runner.run(ticket_id))
        except Busy:
            runner.discard_unstarted_ticket(ticket_id)  # nothing was started; a retry is clean
            raise HTTPException(
                status_code=503, detail="busy, try again shortly", headers={"Retry-After": "5"}
            ) from None

        if wait:
            try:
                summary = future.result(timeout=state.wait_timeout_s)
            except FutureTimeout:
                return TicketCreated(ticket_id=ticket_id, status="resolving")
            except Exception:
                raise HTTPException(
                    status_code=500, detail="the run crashed; see the ticket and its trace"
                ) from None
            response.status_code = 200
            return TicketCreated(
                ticket_id=ticket_id,
                status=summary.outcome,
                run_id=summary.run_id,
                outcome=summary.outcome,
                stop_reason=str(summary.stop_reason),
                answer=summary.answer,
            )
        return TicketCreated(ticket_id=ticket_id, status="received")

    @app.get(
        "/v1/tickets/{ticket_id}",
        response_model=TicketOut,
        dependencies=[Depends(require_api_key)],
        tags=["tickets"],
    )
    def get_ticket(ticket_id: uuid.UUID, db: DbDep) -> TicketOut:
        ticket = db.scalar(
            select(Ticket)
            .where(Ticket.id == ticket_id)
            .options(selectinload(Ticket.messages), selectinload(Ticket.runs))
        )
        if ticket is None:
            raise HTTPException(status_code=404, detail="ticket not found")
        runs = sorted(ticket.runs, key=lambda r: r.started_at)
        return TicketOut(
            id=ticket.id,
            customer_id=ticket.customer_id,
            external_id=ticket.external_id,
            channel=ticket.channel,
            status=ticket.status.value,
            created_at=ticket.created_at,
            messages=[
                MessageOut(role=m.role.value, content=m.content, created_at=m.created_at)
                for m in ticket.messages
            ],
            runs=[
                RunOut(
                    id=r.id,
                    status=r.status.value,
                    outcome=r.outcome,
                    model_tier=r.model_tier,
                    prompt_version=r.prompt_version,
                    tokens_in=r.tokens_in,
                    tokens_out=r.tokens_out,
                    shadow_cost_usd=f"{r.shadow_cost_usd:.10f}",
                    error=r.error,
                    started_at=r.started_at,
                    finished_at=r.finished_at,
                )
                for r in runs
            ],
        )

    @app.get(
        "/v1/runs/{run_id}/trace",
        response_model=TraceOut,
        dependencies=[Depends(require_api_key)],
        tags=["runs"],
    )
    def get_trace(run_id: uuid.UUID, db: DbDep) -> dict[str, Any]:
        trace = load_trace(db, run_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="run not found")
        return trace

    return app


def _duplicate(response: Response, state: ApiState, ticket_id: uuid.UUID) -> TicketCreated:
    response.status_code = 200
    with state.session_factory() as session:
        ticket = session.get(Ticket, ticket_id)
        status = ticket.status.value if ticket else "unknown"
    return TicketCreated(ticket_id=ticket_id, status=status, duplicate=True)
