"""The HTTP API (DESIGN.md section 8).

    POST /v1/tickets            create a ticket and start an agent run (202; ?wait=true waits)
    GET  /v1/tickets/{id}       status, messages (the draft reply is the last agent message), runs
    GET  /v1/runs/{id}/trace    every model and tool call of a run, plus the gateway's audit rows
    GET  /v1/approvals          requests for a human (?status=pending|approved|rejected|expired|all)
    GET  /v1/approvals/{id}     one request: what it would do, or what happened
    POST /v1/approvals/{id}/approve | /reject   a human decides (approving runs the action once)
    GET  /v1/controls           the operator switches (agent on/off, writes on/frozen)
    PUT  /v1/controls/{name}    flip one, with who and why (audited)
    GET  /healthz               liveness and a database check (no key needed)

Security stance: every /v1 route needs the API key. The ``customer_id`` on a ticket is asserted by
the caller (the storefront backend that authenticated the customer), stored once, and from then
on is the only identity a tool call can use. Nothing the model or the customer's text says can
change it. Ticket text is data, never instructions.
"""

from __future__ import annotations

import hmac
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Cookie, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import and_, or_, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from kartly.alerts import AlertLoop, AlertManager
from kartly.api.executor import Busy, RunExecutor
from kartly.api.schemas import (
    ActiveSet,
    AlertCheckOut,
    AlertList,
    AlertOut,
    ApprovalDecide,
    ApprovalList,
    ApprovalOut,
    ControlChanged,
    ControlList,
    ControlOut,
    ControlSet,
    DeadLetterList,
    DeadLetterOut,
    DecisionOut,
    Health,
    JobRequeue,
    LoginBody,
    MeOut,
    MessageOut,
    PasswordSet,
    ReviewerCreate,
    ReviewerList,
    ReviewerOut,
    RunOut,
    TicketCreate,
    TicketCreated,
    TicketList,
    TicketOut,
    TicketSummary,
    TraceOut,
)
from kartly.controls import ControlError, Controls, ControlState
from kartly.costs import build_report
from kartly.gateway import ToolGateway
from kartly.llm.config import ModelsFile
from kartly.metrics import collect, to_prometheus
from kartly.orchestrator import TicketRunner, UnknownCustomerError, load_trace
from kartly.queue import JobQueue
from kartly.reviewers import (
    AuthError,
    ReviewerError,
    ReviewerNotFoundError,
    Reviewers,
    ReviewerView,
)
from kartly.sanitize import clean_str
from kartly.store.models import (
    Action,
    Approval,
    ApprovalStatus,
    JobStatus,
    Run,
    Ticket,
    TicketStatus,
)
from kartly.worker import WorkerPool

SessionFactory = Callable[[], Session]


@dataclass
class ApiState:
    session_factory: SessionFactory
    runner: TicketRunner
    executor: RunExecutor
    api_key: str
    wait_timeout_s: float = 150.0  # a little over the run's own 120 s limit
    gateway: ToolGateway | None = None  # needed for the approval routes
    controls: Controls | None = None  # needed for the control routes
    queue: JobQueue | None = None  # set in queue mode: tickets become jobs for workers
    max_backlog: int = 20  # queue mode: jobs waiting or running before the API answers 503
    workers: WorkerPool | None = None  # embedded workers, started and stopped with the app
    alerts: AlertManager | None = None
    alert_loop: AlertLoop | None = None  # checks the alert rules on a timer
    reviewers: Reviewers | None = None  # admin accounts and sessions
    models: ModelsFile | None = None  # prices for the cost report's what-if section
    cookie_secure: bool = False  # mark the session cookie Secure (https only)
    session_ttl_s: float = 12 * 3600.0


def _state(request: Request) -> ApiState:
    state = request.app.state.kartly
    assert isinstance(state, ApiState)
    return state


StateDep = Annotated[ApiState, Depends(_state)]


SESSION_COOKIE = "kartly_session"
STATIC_DIR = Path(__file__).parent / "static"
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}
# A reviewer may stop things; only an admin (or the service API key) may start them again.
_SAFE_VALUES = {"agent": {"off"}, "writes": {"frozen"}}


@dataclass(frozen=True)
class Principal:
    """Who is calling: the shared API key (a service), or a signed-in reviewer or admin."""

    name: str
    role: str  # "service" | "reviewer" | "admin"
    via: str  # "key" | "session"
    csrf: str | None = None


def _authenticate(state: ApiState, x_api_key: str | None, cookie: str | None) -> Principal | None:
    if x_api_key is not None:  # a key was offered: it must be right; a cookie never rescues it
        if x_api_key and hmac.compare_digest(x_api_key.encode(), state.api_key.encode()):
            return Principal("api-key", "service", "key")
        return None
    if state.reviewers is None:
        return None
    info = state.reviewers.resolve(cookie)
    if info is None:
        return None
    return Principal(info.reviewer.username, info.reviewer.role, "session", info.csrf)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail="invalid or missing credentials",
        headers={"WWW-Authenticate": "ApiKey"},
    )


def read_access(
    state: StateDep,
    x_api_key: Annotated[str | None, Header()] = None,
    session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> Principal:
    principal = _authenticate(state, x_api_key, session)
    if principal is None:
        raise _unauthorized()
    return principal


def write_access(
    principal: Annotated[Principal, Depends(read_access)],
    x_csrf_token: Annotated[str | None, Header()] = None,
) -> Principal:
    """Like read access, and a browser session must also prove the request came from our page."""
    if principal.via == "session" and not (
        x_csrf_token
        and principal.csrf
        and hmac.compare_digest(x_csrf_token.encode(), principal.csrf.encode())
    ):
        raise HTTPException(status_code=403, detail="missing or wrong X-CSRF-Token")
    return principal


def admin_access(principal: Annotated[Principal, Depends(write_access)]) -> Principal:
    if principal.via != "session" or principal.role != "admin":
        raise HTTPException(status_code=403, detail="this needs a signed-in admin")
    return principal


def admin_read_access(principal: Annotated[Principal, Depends(read_access)]) -> Principal:
    if principal.via != "session" or principal.role != "admin":
        raise HTTPException(status_code=403, detail="this needs a signed-in admin")
    return principal


WriteDep = Annotated[Principal, Depends(write_access)]


def _actor_name(principal: Principal, supplied: str | None) -> str:
    """Who to record. A session decides for itself; the shared key must say who is acting."""
    if principal.via == "session":
        return principal.name
    if not supplied:
        raise HTTPException(status_code=422, detail="reviewer is required when using the API key")
    return supplied


access_log = logging.getLogger("kartly.access")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def require_api_key(state: StateDep, x_api_key: Annotated[str | None, Header()] = None) -> None:
    supplied = (x_api_key or "").encode()
    # Same answer for a missing and a wrong key; compare_digest avoids timing differences.
    if not x_api_key or not hmac.compare_digest(supplied, state.api_key.encode()):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing credentials",
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
        if state.workers is not None:
            state.workers.start()
        if state.alert_loop is not None:
            state.alert_loop.start()
        yield
        if state.alert_loop is not None:
            state.alert_loop.stop()
        if state.workers is not None:
            state.workers.stop()
        state.executor.shutdown()  # waits for runs in progress
        if on_shutdown is not None:
            on_shutdown()

    app = FastAPI(title="Kartly Support Agent", version="0.1.0", lifespan=lifespan)
    app.state.kartly = state

    @app.middleware("http")
    async def request_id_and_access_log(request: Request, call_next: Any) -> Response:
        """Every response carries ``X-Request-ID`` (the caller's, if it is a plain token, else a
        new one) and every request leaves one log line, so a customer report can be traced."""
        supplied = request.headers.get("x-request-id", "")
        request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
        started = time.perf_counter()
        status = 500
        try:
            response: Response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            access_log.info(
                "%s %s -> %d in %d ms request_id=%s",
                request.method,
                request.url.path,
                status,
                (time.perf_counter() - started) * 1000,
                request_id,
            )

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

    # ---------------------------------------------------------------- the admin page
    # Three static files. The page loads nothing from anywhere else, so the policy forbids it;
    # with no inline script or style, a customer-written string cannot become running code.

    def _asset(name: str, media_type: str) -> Callable[[], Response]:
        body = (STATIC_DIR / name).read_bytes()

        def serve() -> Response:
            return Response(body, media_type=media_type, headers=_PAGE_HEADERS)

        return serve

    for route, name, media in (
        ("/admin", "admin.html", "text/html; charset=utf-8"),
        ("/admin/admin.js", "admin.js", "text/javascript; charset=utf-8"),
        ("/admin/admin.css", "admin.css", "text/css; charset=utf-8"),
    ):
        app.add_api_route(route, _asset(name, media), methods=["GET"], include_in_schema=False)

    @app.get("/v1/costs", dependencies=[Depends(read_access)], tags=["ops"])
    def costs_report(
        state: StateDep,
        window_s: Annotated[float, Query(gt=0, le=365 * 86400)] = 7 * 86400.0,
    ) -> dict[str, Any]:
        return build_report(state.session_factory, state.models, window_s=window_s)

    @app.get("/v1/metrics", dependencies=[Depends(read_access)], tags=["ops"])
    def metrics_json(
        state: StateDep,
        window_s: Annotated[float, Query(gt=0, le=90 * 86400)] = 86400.0,
    ) -> dict[str, Any]:
        return collect(state.session_factory, window_s=window_s, controls=state.controls)

    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        dependencies=[Depends(read_access)],
        tags=["ops"],
    )
    def metrics_prometheus(
        state: StateDep,
        window_s: Annotated[float, Query(gt=0, le=90 * 86400)] = 3600.0,
    ) -> PlainTextResponse:
        body = to_prometheus(
            collect(state.session_factory, window_s=window_s, controls=state.controls)
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4")

    def _alerts(state: ApiState) -> AlertManager:
        if state.alerts is None:
            raise HTTPException(status_code=503, detail="alerts are not configured")
        return state.alerts

    @app.get(
        "/v1/alerts",
        response_model=AlertList,
        dependencies=[Depends(read_access)],
        tags=["ops"],
    )
    def list_alerts(state: StateDep) -> AlertList:
        return AlertList(
            alerts=[
                AlertOut(**{k: getattr(a, k) for k in AlertOut.model_fields})
                for a in _alerts(state).states()
            ]
        )

    @app.post(
        "/v1/alerts/check",
        response_model=AlertCheckOut,
        dependencies=[Depends(write_access)],
        tags=["ops"],
    )
    def check_alerts(state: StateDep) -> AlertCheckOut:
        result = _alerts(state).check()
        return AlertCheckOut(
            firing=[f.rule for f in result.firing],
            fired=[a.rule for a in result.fired],
            cleared=[a.rule for a in result.cleared],
        )

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

        if state.queue is not None:
            return _enqueue(state, ticket_id, response, wait)

        try:
            future = state.executor.submit(lambda: runner.run(ticket_id))
        except Busy:
            raise _busy(runner, ticket_id) from None

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
        dependencies=[Depends(read_access)],
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
        dependencies=[Depends(read_access)],
        tags=["runs"],
    )
    def get_trace(run_id: uuid.UUID, db: DbDep) -> dict[str, Any]:
        trace = load_trace(db, run_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="run not found")
        return trace

    def _gateway(state: ApiState) -> ToolGateway:
        if state.gateway is None:
            raise HTTPException(status_code=503, detail="approvals are not configured")
        return state.gateway

    def _approval_out(db: Session, approval: Approval) -> ApprovalOut:
        action = db.get(Action, approval.action_id)
        ticket = db.get(Ticket, approval.ticket_id)
        assert action is not None and ticket is not None
        result = action.result if isinstance(action.result, dict) else None
        waiting = approval.status is ApprovalStatus.PENDING
        preview = result.get("preview") if waiting and result else None
        return ApprovalOut(
            id=approval.id,
            ticket_id=approval.ticket_id,
            customer_id=ticket.customer_id,
            status=approval.status.value,
            reason=approval.reason,
            tool=action.tool_name,
            args=action.args,
            preview=preview if isinstance(preview, dict) else None,
            outcome=None if waiting else result,
            action_status=action.status.value,
            requested_at=approval.requested_at,
            expires_at=approval.expires_at,
            decided_by=approval.decided_by,
            decided_at=approval.decided_at,
            decision_note=approval.decision_note,
        )

    @app.get(
        "/v1/approvals",
        response_model=ApprovalList,
        dependencies=[Depends(read_access)],
        tags=["approvals"],
    )
    def list_approvals(
        state: StateDep,
        db: DbDep,
        status: Literal["pending", "approved", "rejected", "expired", "all"] = "pending",
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> ApprovalList:
        expired_now = _gateway(state).expire_stale()
        query = select(Approval).order_by(Approval.requested_at, Approval.id).limit(limit)
        if status != "all":
            query = query.where(Approval.status == ApprovalStatus(status))
        rows = db.scalars(query).all()
        return ApprovalList(
            approvals=[_approval_out(db, row) for row in rows], expired_now=expired_now
        )

    @app.get(
        "/v1/approvals/{approval_id}",
        response_model=ApprovalOut,
        dependencies=[Depends(read_access)],
        tags=["approvals"],
    )
    def get_approval(approval_id: uuid.UUID, db: DbDep) -> ApprovalOut:
        approval = db.get(Approval, approval_id)
        if approval is None:
            raise HTTPException(status_code=404, detail="approval not found")
        return _approval_out(db, approval)

    def _decide(
        state: ApiState,
        approval_id: uuid.UUID,
        body: ApprovalDecide,
        response: Response,
        *,
        approve: bool,
        reviewer: str,
    ) -> DecisionOut:
        gateway = _gateway(state)
        # Short sessions of their own: none is open while the gateway decides, so the decision
        # never waits on (or is blocked by) a transaction this request started.
        with state.session_factory() as session:
            if session.get(Approval, approval_id) is None:
                raise HTTPException(status_code=404, detail="approval not found")
        decision = gateway.decide(
            approval_id, approve=approve, actor=f"human:{reviewer}", note=body.note
        )
        with state.session_factory() as session:
            approval = session.get(Approval, approval_id)
            ticket = session.get(Ticket, approval.ticket_id) if approval else None
            ticket_status = ticket.status.value if ticket else None
        if not decision.ok:
            response.status_code = 409
        return DecisionOut(
            approval_id=approval_id,
            ok=decision.ok,
            result=decision.status,  # type: ignore[arg-type]
            message=decision.message,
            data=decision.data,
            ticket_status=ticket_status,
        )

    @app.post(
        "/v1/approvals/{approval_id}/approve",
        response_model=DecisionOut,
        dependencies=[Depends(write_access)],
        tags=["approvals"],
    )
    def approve(
        approval_id: uuid.UUID,
        body: ApprovalDecide,
        state: StateDep,
        response: Response,
        principal: WriteDep,
    ) -> DecisionOut:
        who = _actor_name(principal, body.reviewer)
        return _decide(state, approval_id, body, response, approve=True, reviewer=who)

    @app.post(
        "/v1/approvals/{approval_id}/reject",
        response_model=DecisionOut,
        dependencies=[Depends(write_access)],
        tags=["approvals"],
    )
    def reject(
        approval_id: uuid.UUID,
        body: ApprovalDecide,
        state: StateDep,
        response: Response,
        principal: WriteDep,
    ) -> DecisionOut:
        who = _actor_name(principal, body.reviewer)
        return _decide(state, approval_id, body, response, approve=False, reviewer=who)

    def _control_out(state: ControlState) -> ControlOut:
        return ControlOut(
            name=state.name,
            value=state.value,
            allowed=list(state.allowed),
            reason=state.reason,
            updated_by=state.updated_by,
            updated_at=state.updated_at,
        )

    def _controls(state: ApiState) -> Controls:
        if state.controls is None:
            raise HTTPException(status_code=503, detail="controls are not configured")
        return state.controls

    @app.get(
        "/v1/controls",
        response_model=ControlList,
        dependencies=[Depends(read_access)],
        tags=["controls"],
    )
    def list_controls(state: StateDep) -> ControlList:
        return ControlList(controls=[_control_out(c) for c in _controls(state).all()])

    @app.put(
        "/v1/controls/{name}",
        response_model=ControlChanged,
        dependencies=[Depends(write_access)],
        tags=["controls"],
    )
    def set_control(
        name: str, body: ControlSet, state: StateDep, principal: WriteDep
    ) -> ControlChanged:
        who = _actor_name(principal, body.reviewer)
        if (
            principal.role == "reviewer"
            and name in _SAFE_VALUES
            and body.value not in _SAFE_VALUES[name]
        ):
            raise HTTPException(
                status_code=403, detail="a reviewer can stop things; an admin turns them back on"
            )
        try:
            new, changed = _controls(state).set(name, body.value, who, body.reason)
        except ControlError as exc:
            known = "unknown control" in str(exc)
            raise HTTPException(status_code=404 if known else 422, detail=str(exc)) from exc
        return ControlChanged(**_control_out(new).model_dump(), changed=changed)

    @app.get(
        "/v1/dead-letters",
        response_model=DeadLetterList,
        dependencies=[Depends(read_access)],
        tags=["queue"],
    )
    def dead_letters(state: StateDep) -> DeadLetterList:
        queue = _queue(state)
        return DeadLetterList(
            jobs=[
                DeadLetterOut(
                    job_id=j.id,
                    ticket_id=j.ticket_id,
                    attempts=j.attempts,
                    last_error=j.last_error,
                    died_at=j.finished_at,
                )
                for j in queue.dead()
            ],
            backlog=queue.depth(),
        )

    @app.post(
        "/v1/dead-letters/{job_id}/requeue",
        dependencies=[Depends(write_access)],
        tags=["queue"],
        status_code=202,
    )
    def requeue_dead(
        job_id: uuid.UUID, body: JobRequeue, state: StateDep, principal: WriteDep
    ) -> dict[str, str]:
        who = _actor_name(principal, body.reviewer)
        if not _queue(state).requeue(job_id, f"human:{who}"):
            raise HTTPException(status_code=404, detail="no dead job with that id")
        return {"status": "queued"}

    # ---------------------------------------------------------------- sign-in and accounts

    def _reviewers(state: ApiState) -> Reviewers:
        if state.reviewers is None:
            raise HTTPException(status_code=503, detail="sign-in is not configured")
        return state.reviewers

    @app.post("/v1/admin/login", response_model=MeOut, tags=["admin"])
    def login(body: LoginBody, state: StateDep, response: Response) -> MeOut:
        try:
            result = _reviewers(state).login(body.username, body.password)
        except AuthError:
            raise HTTPException(status_code=401, detail="invalid username or password") from None
        response.set_cookie(
            SESSION_COOKIE,
            result.token,
            max_age=int(state.session_ttl_s),
            httponly=True,
            samesite="strict",
            secure=state.cookie_secure,
            path="/",
        )
        response.headers["Cache-Control"] = "no-store"
        return MeOut(username=result.reviewer.username, role=result.reviewer.role, csrf=result.csrf)

    @app.post("/v1/admin/logout", status_code=204, tags=["admin"])
    def logout(
        state: StateDep,
        response: Response,
        session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
    ) -> None:
        if state.reviewers is not None:
            state.reviewers.logout(session)
        response.delete_cookie(SESSION_COOKIE, path="/")

    @app.get("/v1/admin/me", response_model=MeOut, tags=["admin"])
    def me(principal: Annotated[Principal, Depends(read_access)], response: Response) -> MeOut:
        if principal.via != "session" or principal.csrf is None:
            raise HTTPException(status_code=403, detail="not a browser session")
        response.headers["Cache-Control"] = "no-store"
        return MeOut(username=principal.name, role=principal.role, csrf=principal.csrf)

    def _reviewer_out(r: ReviewerView) -> ReviewerOut:
        return ReviewerOut(
            username=r.username,
            role=r.role,
            active=r.active,
            locked=r.locked,
            created_at=r.created_at,
            last_login_at=r.last_login_at,
        )

    def _account_error(exc: ReviewerError) -> HTTPException:
        code = 404 if isinstance(exc, ReviewerNotFoundError) else 422
        return HTTPException(status_code=code, detail=str(exc))

    @app.get(
        "/v1/admin/reviewers",
        response_model=ReviewerList,
        dependencies=[Depends(admin_read_access)],
        tags=["admin"],
    )
    def list_reviewers(state: StateDep) -> ReviewerList:
        return ReviewerList(reviewers=[_reviewer_out(r) for r in _reviewers(state).all()])

    @app.post(
        "/v1/admin/reviewers",
        response_model=ReviewerOut,
        status_code=201,
        tags=["admin"],
    )
    def create_reviewer(
        body: ReviewerCreate,
        state: StateDep,
        admin: Annotated[Principal, Depends(admin_access)],
    ) -> ReviewerOut:
        try:
            created = _reviewers(state).create(
                body.username, body.password, body.role, actor=admin.name
            )
        except ReviewerError as exc:
            raise _account_error(exc) from exc
        return _reviewer_out(created)

    @app.put("/v1/admin/reviewers/{username}/password", status_code=204, tags=["admin"])
    def set_reviewer_password(
        username: str,
        body: PasswordSet,
        state: StateDep,
        admin: Annotated[Principal, Depends(admin_access)],
    ) -> None:
        try:
            _reviewers(state).set_password(username, body.password, actor=admin.name)
        except ReviewerError as exc:
            raise _account_error(exc) from exc

    @app.put("/v1/admin/reviewers/{username}/active", status_code=204, tags=["admin"])
    def set_reviewer_active(
        username: str,
        body: ActiveSet,
        state: StateDep,
        admin: Annotated[Principal, Depends(admin_access)],
    ) -> None:
        try:
            _reviewers(state).set_active(username, body.active, actor=admin.name)
        except ReviewerError as exc:
            raise _account_error(exc) from exc

    # ---------------------------------------------------------------- listing tickets

    @app.get(
        "/v1/tickets",
        response_model=TicketList,
        dependencies=[Depends(read_access)],
        tags=["tickets"],
    )
    def list_tickets(
        db: DbDep,
        status: str | None = None,
        customer_id: Annotated[str | None, Query(max_length=40)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        before: Annotated[str | None, Query(max_length=80)] = None,
    ) -> TicketList:
        query = select(Ticket)
        if status is not None:
            try:
                query = query.where(Ticket.status == TicketStatus(status))
            except ValueError:
                raise HTTPException(status_code=422, detail="unknown status") from None
        if customer_id is not None:
            query = query.where(Ticket.customer_id == customer_id)
        if before is not None:
            moment, _, ident = before.partition("_")
            try:
                at, tid = datetime.fromisoformat(moment), uuid.UUID(ident)
            except ValueError:
                raise HTTPException(status_code=422, detail="bad cursor") from None
            query = query.where(
                or_(Ticket.created_at < at, and_(Ticket.created_at == at, Ticket.id < tid))
            )
        rows = db.scalars(
            query.order_by(Ticket.created_at.desc(), Ticket.id.desc()).limit(limit)
        ).all()
        latest = (
            select(Run.outcome)
            .where(Run.ticket_id == Ticket.id)
            .order_by(Run.started_at.desc())
            .limit(1)
            .scalar_subquery()
        )
        outcomes: dict[uuid.UUID, str | None] = dict(
            db.execute(select(Ticket.id, latest).where(Ticket.id.in_([t.id for t in rows]))).all()
        )
        nxt = f"{rows[-1].created_at.isoformat()}_{rows[-1].id}" if len(rows) == limit else None
        return TicketList(
            tickets=[
                TicketSummary(
                    id=t.id,
                    customer_id=t.customer_id,
                    status=t.status.value,
                    created_at=t.created_at,
                    last_run_outcome=outcomes.get(t.id),
                )
                for t in rows
            ],
            next_before=nxt,
        )

    return app


def _queue(state: ApiState) -> JobQueue:
    if state.queue is None:
        raise HTTPException(status_code=404, detail="the job queue is not enabled")
    return state.queue


def _busy(runner: TicketRunner, ticket_id: uuid.UUID) -> HTTPException:
    runner.discard_unstarted_ticket(ticket_id)  # nothing was started; a retry is clean
    return HTTPException(
        status_code=503, detail="busy, try again shortly", headers={"Retry-After": "5"}
    )


def _enqueue(
    state: ApiState, ticket_id: uuid.UUID, response: Response, wait: bool
) -> TicketCreated:
    queue = state.queue
    assert queue is not None
    if queue.depth() >= state.max_backlog:
        raise _busy(state.runner, ticket_id)
    job_id = queue.enqueue(ticket_id)
    if not wait:
        return TicketCreated(ticket_id=ticket_id, status="received")
    view = queue.wait(job_id, state.wait_timeout_s)
    if view is None or view.status == JobStatus.QUEUED or view.status == JobStatus.RUNNING:
        return TicketCreated(ticket_id=ticket_id, status="resolving")
    response.status_code = 200
    if view.status == JobStatus.DEAD:
        return TicketCreated(ticket_id=ticket_id, status="escalated", outcome="escalated")
    result = view.result or {}
    return TicketCreated(
        ticket_id=ticket_id,
        status=str(result.get("outcome", "drafted")),
        run_id=uuid.UUID(result["run_id"]) if result.get("run_id") else None,
        outcome=result.get("outcome"),
        stop_reason=result.get("stop_reason"),
        answer=result.get("answer"),
    )


def _duplicate(response: Response, state: ApiState, ticket_id: uuid.UUID) -> TicketCreated:
    response.status_code = 200
    with state.session_factory() as session:
        ticket = session.get(Ticket, ticket_id)
        status = ticket.status.value if ticket else "unknown"
    return TicketCreated(ticket_id=ticket_id, status=status, duplicate=True)
