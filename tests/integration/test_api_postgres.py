"""The API on a real Postgres: concurrent runs, the unique ``external_id``, hostile input."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, sessionmaker

from kartly.api import ApiState, BoundedExecutor, create_app
from kartly.gateway import ToolGateway
from kartly.llm.types import ChatResponse
from kartly.orchestrator import TicketRunner
from kartly.store.models import AuditLog, Message, Run, Ticket
from tests.llm_support import make_response, tool_call

pytestmark = pytest.mark.postgres

KEY = "pg-api-key-123456"
AUTH = {"X-API-Key": KEY}


class LookupThenAnswer:
    """Safe to share between threads: it decides from the conversation, not from a queue."""

    name = "lookup-then-answer"

    def __init__(self, order_id: str = "KRT-100002") -> None:
        self._order_id = order_id
        self._lock = threading.Lock()
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
        with self._lock:
            self.calls.append([dict(m) for m in messages])
        if messages[-1]["role"] == "tool":
            return make_response("Your order is in transit.")
        return make_response(None, [tool_call("get_tracking", {"order_id": self._order_id})])


def build(
    factory: sessionmaker[Session],
    gateway: ToolGateway,
    client: LookupThenAnswer,
    *,
    workers: int = 3,
    pending: int = 20,
) -> tuple[Any, BoundedExecutor]:
    executor = BoundedExecutor(workers, pending)
    state = ApiState(
        session_factory=factory,
        runner=TicketRunner(factory, client, gateway, model_tier="scripted"),
        executor=executor,
        api_key=KEY,
    )
    return create_app(state), executor


def count(factory: sessionmaker[Session], model: type[Any]) -> int:
    with factory() as session:
        return session.scalar(select(func.count()).select_from(model)) or 0


def test_concurrent_tickets_all_finish_with_their_own_trace(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    client = LookupThenAnswer()
    app, executor = build(pg_factory, pg_gateway, client)
    customers = ["C-0001", "C-0002", "C-0003"] * 3

    def submit(n: int) -> dict[str, Any]:
        response = TestClient(app).post(
            "/v1/tickets?wait=true",
            json={"customer_id": customers[n], "message": f"Where is it? ({n})"},
            headers=AUTH,
        )
        assert response.status_code == 200, response.text
        return dict(response.json())

    try:
        with ThreadPoolExecutor(max_workers=len(customers)) as pool:
            results = list(pool.map(submit, range(len(customers))))
    finally:
        executor.shutdown()

    assert {r["outcome"] for r in results} == {"drafted"}
    assert len({r["run_id"] for r in results}) == len(customers)
    assert count(pg_factory, Ticket) == count(pg_factory, Run) == len(customers)
    with pg_factory() as session:
        # each run kept its own steps, in order, with no gaps or duplicates
        rows = session.execute(
            text(
                "SELECT run_id, count(*), count(DISTINCT idx), max(idx) "
                "FROM run_steps GROUP BY run_id"
            )
        ).all()
    assert len(rows) == len(customers)
    assert all(n == distinct == top + 1 for _, n, distinct, top in rows)


def test_the_same_external_id_sent_at_once_makes_one_ticket_and_one_run(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    client = LookupThenAnswer()
    app, executor = build(pg_factory, pg_gateway, client)
    start = threading.Barrier(6)

    def submit(_: int) -> tuple[int, dict[str, Any]]:
        start.wait()
        response = TestClient(app).post(
            "/v1/tickets?wait=true",
            json={"customer_id": "C-0001", "message": "hello", "external_id": "hook-1"},
            headers=AUTH,
        )
        return response.status_code, dict(response.json())

    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(submit, range(6)))
    finally:
        executor.shutdown()

    assert {status for status, _ in results} <= {200, 202}, results
    assert len({body["ticket_id"] for _, body in results}) == 1
    assert count(pg_factory, Ticket) == 1 and count(pg_factory, Run) == 1


def test_hostile_text_is_cleaned_or_refused_never_a_500(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    """NUL bytes arrive as valid JSON but Postgres text columns reject them: they are stripped.
    A lone surrogate is not valid text: it is a clean 422."""
    client = LookupThenAnswer()
    app, executor = build(pg_factory, pg_gateway, client)
    headers = {**AUTH, "Content-Type": "application/json"}
    try:
        with TestClient(app) as http:
            nul = http.post(
                "/v1/tickets?wait=true",
                content=b'{"customer_id": "C-0001", "message": "hi \\u0000 there"}',
                headers=headers,
            )
            assert nul.status_code == 200, nul.text
            ticket = http.get(f"/v1/tickets/{nul.json()['ticket_id']}", headers=AUTH).json()

            surrogate = http.post(
                "/v1/tickets",
                content=b'{"customer_id": "C-0001", "message": "hi \\ud800 there"}',
                headers=headers,
            )
            assert surrogate.status_code == 422
    finally:
        executor.shutdown()

    assert ticket["messages"][0]["content"] == "hi  there"
    with pg_factory() as session:
        stored = session.scalars(select(Message.content)).all()
        assert count(pg_factory, Ticket) == 1  # the refused one left nothing behind
    assert all("\x00" not in text_ for text_ in stored)
    assert "\x00" not in client.calls[0][-1]["content"]


def test_another_customers_order_stays_hidden_on_postgres(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    client = LookupThenAnswer(order_id="KRT-100001")  # belongs to C-0001
    app, executor = build(pg_factory, pg_gateway, client)
    try:
        with TestClient(app) as http:
            response = http.post(
                "/v1/tickets?wait=true",
                json={"customer_id": "C-0002", "message": "Show me KRT-100001"},
                headers=AUTH,
            )
            trace = http.get(f"/v1/runs/{response.json()['run_id']}/trace", headers=AUTH).json()
    finally:
        executor.shutdown()

    [tool_step] = [s for s in trace["steps"] if s["kind"] == "tool_call"]
    assert tool_step["output"]["error_code"] == "order_not_found"
    with pg_factory() as session:
        succeeded = session.scalars(
            select(AuditLog).where(AuditLog.action == "tool.succeeded")
        ).all()
    assert succeeded == []


def test_a_full_queue_leaves_no_ticket_on_postgres(
    pg_factory: sessionmaker[Session], pg_gateway: ToolGateway
) -> None:
    client = LookupThenAnswer()
    release = threading.Event()
    app, executor = build(pg_factory, pg_gateway, client, workers=1, pending=1)
    executor.submit(release.wait)  # occupies the only slot
    try:
        response = TestClient(app).post(
            "/v1/tickets", json={"customer_id": "C-0001", "message": "hi"}, headers=AUTH
        )
        assert response.status_code == 503 and response.headers["retry-after"] == "5"
        assert count(pg_factory, Ticket) == 0
    finally:
        release.set()
        executor.shutdown()
