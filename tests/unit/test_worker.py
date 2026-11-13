"""Workers: one job at a time, retry on failure, never retry after an action ran."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from kartly.api import ApiState, create_app
from kartly.orchestrator import StopReason, TicketRunner
from kartly.orchestrator.service import RunSummary
from kartly.queue import JobQueue
from kartly.store.models import Action, ActionStatus, Job, JobStatus, Run, Ticket, TicketStatus
from kartly.worker import Worker, WorkerPool
from tests.llm_support import ScriptedClient, make_response
from tests.support import make_sqlite_engine, seeded_session_factory
from tests.unit.test_api_approvals import AUTH, KEY, Inline, make_gateway
from tests.unit.test_queue import Clock, make_queue, make_ticket


@pytest.fixture
def file_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    """Threads need real connections, as on Postgres: an in-memory SQLite shares one."""
    engine = make_sqlite_engine(str(tmp_path / "queue.db"))
    yield seeded_session_factory(engine)
    engine.dispose()


def summary(ticket: uuid.UUID, stop: StopReason = StopReason.ANSWERED) -> RunSummary:
    return RunSummary(
        run_id=uuid.uuid4(),
        ticket_id=ticket,
        outcome="drafted",
        stop_reason=stop,
        detail="because",
        answer="Your order is on its way.",
        llm_calls=1,
        tool_calls=0,
        tokens_in=10,
        tokens_out=5,
        cost_usd=Decimal("0"),
    )


class Script:
    """A fake ``runner.run``: each call plays the next step (a summary or an exception)."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.calls: list[uuid.UUID] = []

    def __call__(self, ticket_id: uuid.UUID) -> RunSummary:
        self.calls.append(ticket_id)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, StopReason):
            return summary(ticket_id, step)
        return summary(ticket_id)


def add_action(factory: sessionmaker[Session], ticket: uuid.UUID, status: ActionStatus) -> None:
    with factory() as session:
        run = Run(ticket_id=ticket, mode="shadow", model_tier="t", prompt_version="v")
        session.add(run)
        session.flush()
        session.add(
            Action(
                run_id=run.id,
                ticket_id=ticket,
                tool_name="issue_refund",
                args={},
                idempotency_key=uuid.uuid4().hex,
                status=status,
            )
        )
        session.commit()


def status_of(factory: sessionmaker[Session], job_id: uuid.UUID) -> Job:
    with factory() as session:
        job = session.get(Job, job_id)
        assert job is not None
        return job


def setup(
    factory: sessionmaker[Session], script: Script, clock: Clock | None = None, **kw: Any
) -> tuple[JobQueue, Worker, Clock]:
    clock = clock or Clock()
    queue = make_queue(factory, clock, **kw)
    return queue, Worker(queue, script, factory, worker_id="w1"), clock


def test_a_worker_runs_a_job_and_stores_what_the_run_produced(
    seeded_factory: sessionmaker[Session],
) -> None:
    script = Script(StopReason.ANSWERED)
    queue, worker, _ = setup(seeded_factory, script)
    t = make_ticket(seeded_factory)
    job_id = queue.enqueue(t)

    assert worker.run_once() is True
    job = status_of(seeded_factory, job_id)
    assert job.status == JobStatus.SUCCEEDED and script.calls == [t]
    assert job.result is not None
    assert (
        job.result["outcome"] == "drafted" and job.result["answer"] == "Your order is on its way."
    )
    assert job.result["stop_reason"] == "answered" and uuid.UUID(job.result["run_id"])
    assert worker.run_once() is False  # nothing left


def test_a_crash_is_retried_after_the_backoff(seeded_factory: sessionmaker[Session]) -> None:
    script = Script(RuntimeError("database hiccup"), StopReason.ANSWERED)
    queue, worker, clock = setup(seeded_factory, script)
    job_id = queue.enqueue(make_ticket(seeded_factory))

    assert worker.run_once() is True
    job = status_of(seeded_factory, job_id)
    assert job.status == JobStatus.QUEUED and "RuntimeError: database hiccup" in (
        job.last_error or ""
    )
    assert worker.run_once() is False  # backing off (30 s)
    clock.advance(31)
    assert worker.run_once() is True
    assert status_of(seeded_factory, job_id).status == JobStatus.SUCCEEDED


def test_a_provider_failure_that_ended_the_run_is_retried(
    seeded_factory: sessionmaker[Session],
) -> None:
    script = Script(StopReason.LLM_ERROR, StopReason.ANSWERED)
    queue, worker, clock = setup(seeded_factory, script)
    job_id = queue.enqueue(make_ticket(seeded_factory))
    worker.run_once()
    job = status_of(seeded_factory, job_id)
    assert job.status == JobStatus.QUEUED and (job.last_error or "").startswith("llm_error")
    clock.advance(31)
    worker.run_once()
    assert status_of(seeded_factory, job_id).status == JobStatus.SUCCEEDED


@pytest.mark.parametrize("stop", [StopReason.ANSWERED, StopReason.STEP_LIMIT])
def test_other_stop_reasons_are_final(
    seeded_factory: sessionmaker[Session], stop: StopReason
) -> None:
    queue, worker, _ = setup(seeded_factory, Script(stop))
    job_id = queue.enqueue(make_ticket(seeded_factory))
    worker.run_once()
    assert status_of(seeded_factory, job_id).status == JobStatus.SUCCEEDED


def test_three_failures_end_in_the_dead_letter_state(
    seeded_factory: sessionmaker[Session],
) -> None:
    script = Script(*[RuntimeError("x")] * 3)
    queue, worker, clock = setup(seeded_factory, script, max_attempts=3)
    t = make_ticket(seeded_factory, TicketStatus.RESOLVING)
    job_id = queue.enqueue(t)
    for _ in range(3):
        assert worker.run_once() is True
        clock.advance(400)
    assert worker.run_once() is False
    assert status_of(seeded_factory, job_id).status == JobStatus.DEAD
    assert len(script.calls) == 3
    with seeded_factory() as session:
        ticket = session.get(Ticket, t)
        assert ticket is not None and ticket.status == TicketStatus.ESCALATED


@pytest.mark.parametrize(
    "earlier",
    [ActionStatus.SUCCEEDED, ActionStatus.AWAITING_APPROVAL, ActionStatus.PENDING],
)
def test_a_retry_is_refused_when_an_earlier_attempt_already_acted(
    seeded_factory: sessionmaker[Session], earlier: ActionStatus
) -> None:
    script = Script(RuntimeError("crashed after the refund"), StopReason.ANSWERED)
    queue, worker, clock = setup(seeded_factory, script)
    t = make_ticket(seeded_factory)
    job_id = queue.enqueue(t)
    worker.run_once()  # attempt 1 crashes...
    add_action(seeded_factory, t, earlier)  # ...after having acted
    clock.advance(31)

    assert worker.run_once() is True
    job = status_of(seeded_factory, job_id)
    assert job.status == JobStatus.DEAD and "unsafe_to_retry" in (job.last_error or "")
    assert len(script.calls) == 1  # the agent was not run again


@pytest.mark.parametrize("earlier", [ActionStatus.FAILED, ActionStatus.REJECTED])
def test_a_failed_or_rejected_action_does_not_block_a_retry(
    seeded_factory: sessionmaker[Session], earlier: ActionStatus
) -> None:
    script = Script(RuntimeError("x"), StopReason.ANSWERED)
    queue, worker, clock = setup(seeded_factory, script)
    t = make_ticket(seeded_factory)
    job_id = queue.enqueue(t)
    worker.run_once()
    add_action(seeded_factory, t, earlier)
    clock.advance(31)
    worker.run_once()
    assert status_of(seeded_factory, job_id).status == JobStatus.SUCCEEDED


def test_the_first_attempt_is_never_blocked_by_old_actions(
    seeded_factory: sessionmaker[Session],
) -> None:
    """A requeued job starts at attempt 1 again, so a human who requeues takes the call."""
    script = Script(StopReason.ANSWERED)
    queue, worker, _ = setup(seeded_factory, script)
    t = make_ticket(seeded_factory)
    add_action(seeded_factory, t, ActionStatus.SUCCEEDED)
    job_id = queue.enqueue(t)
    worker.run_once()
    assert status_of(seeded_factory, job_id).status == JobStatus.SUCCEEDED


def test_a_missing_ticket_is_buried_without_retrying(
    seeded_factory: sessionmaker[Session],
) -> None:
    script = Script(LookupError("no ticket"))
    queue, worker, _ = setup(seeded_factory, script)
    job_id = queue.enqueue(make_ticket(seeded_factory))
    worker.run_once()
    job = status_of(seeded_factory, job_id)
    assert job.status == JobStatus.DEAD and (job.last_error or "").startswith("ticket missing")


def test_run_forever_works_until_stopped(file_factory: sessionmaker[Session]) -> None:
    script = Script(*[StopReason.ANSWERED] * 2)
    queue, worker, _ = setup(file_factory, script)
    ids = [queue.enqueue(make_ticket(file_factory)) for _ in range(2)]
    pool = WorkerPool([worker], poll_s=0.01)
    pool.start()
    done = threading.Event()
    for _ in range(200):
        if all(status_of(file_factory, i).status == JobStatus.SUCCEEDED for i in ids):
            done.set()
            break
        threading.Event().wait(0.02)
    pool.stop(5)
    assert done.is_set()


def test_a_broken_queue_does_not_kill_the_worker_thread(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    queue, worker, _ = setup(seeded_factory, Script())
    calls = {"n": 0}

    def flaky() -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("database restarted")
        return False

    monkeypatch.setattr(worker, "run_once", flaky)
    stop = threading.Event()
    t = threading.Thread(target=worker.run_forever, args=(stop, 0.01))
    t.start()
    for _ in range(200):
        if calls["n"] >= 3:
            break
        threading.Event().wait(0.01)
    stop.set()
    t.join(5)
    assert calls["n"] >= 3


# ----------------------------------------------------------------------------- through the API


def queue_app(
    factory: sessionmaker[Session], script: list[Any], *, backlog: int = 20
) -> tuple[TestClient, JobQueue, Worker]:
    runner = TicketRunner(
        factory, ScriptedClient(script), make_gateway(factory), model_tier="scripted"
    )
    queue = JobQueue(factory)
    worker = Worker(queue, runner.run, factory, worker_id="api-test")
    state = ApiState(
        session_factory=factory,
        runner=runner,
        executor=Inline(),
        api_key=KEY,
        queue=queue,
        max_backlog=backlog,
        wait_timeout_s=5.0,
    )
    return TestClient(create_app(state)), queue, worker


def api_state(http: TestClient) -> ApiState:
    state = cast(Any, http.app).state.kartly
    assert isinstance(state, ApiState)
    return state


HELLO = {"customer_id": "C-0004", "message": "Where is my order KRT-100007?"}


def test_queue_mode_returns_at_once_and_a_worker_does_the_run(
    seeded_factory: sessionmaker[Session],
) -> None:
    http, queue, worker = queue_app(seeded_factory, [make_response("It was delivered.")])
    created = http.post("/v1/tickets", json=HELLO, headers=AUTH)
    assert created.status_code == 202 and created.json()["status"] == "received"
    ticket_id = created.json()["ticket_id"]
    assert queue.depth() == 1

    assert worker.run_once() is True
    shown = http.get(f"/v1/tickets/{ticket_id}", headers=AUTH).json()
    assert shown["status"] == "drafted" and len(shown["runs"]) == 1


def test_wait_true_polls_the_job_result(file_factory: sessionmaker[Session]) -> None:
    http, _queue, worker = queue_app(file_factory, [make_response("It was delivered.")])
    stop = threading.Event()
    t = threading.Thread(target=worker.run_forever, args=(stop, 0.01))
    t.start()
    try:
        done = http.post("/v1/tickets?wait=true", json=HELLO, headers=AUTH)
    finally:
        stop.set()
        t.join(5)
    body = done.json()
    assert done.status_code == 200 and body["outcome"] == "drafted"
    assert body["answer"] and body["run_id"] and body["stop_reason"] == "answered"


def test_wait_true_that_times_out_reports_the_ticket_as_resolving(
    seeded_factory: sessionmaker[Session],
) -> None:
    http, _q, _w = queue_app(seeded_factory, [])  # no worker running
    api_state(http).wait_timeout_s = 0.3
    created = http.post("/v1/tickets?wait=true", json=HELLO, headers=AUTH)
    assert created.status_code == 202 and created.json()["status"] == "resolving"


def test_a_full_backlog_answers_503_and_leaves_no_ticket(
    seeded_factory: sessionmaker[Session],
) -> None:
    http, queue, _w = queue_app(seeded_factory, [], backlog=1)
    assert http.post("/v1/tickets", json=HELLO, headers=AUTH).status_code == 202
    busy = http.post("/v1/tickets", json=HELLO, headers=AUTH)
    assert busy.status_code == 503 and busy.headers["retry-after"] == "5"
    assert queue.depth() == 1
    with seeded_factory() as session:
        assert len(session.query(Ticket).all()) == 1  # the refused one was discarded


def test_a_dead_job_is_listed_and_can_be_requeued_by_a_named_reviewer(
    seeded_factory: sessionmaker[Session],
) -> None:
    http, queue, worker = queue_app(seeded_factory, [])
    worker._run = Script(*[RuntimeError("provider down")] * 3)
    queue = JobQueue(seeded_factory, max_attempts=1)
    worker._queue = queue
    api_state(http).queue = queue
    created = http.post("/v1/tickets", json=HELLO, headers=AUTH)
    worker.run_once()

    listed = http.get("/v1/dead-letters", headers=AUTH).json()
    [only] = listed["jobs"]
    assert only["ticket_id"] == created.json()["ticket_id"] and only["attempts"] == 1
    assert "provider down" in only["last_error"]
    shown = http.get(f"/v1/tickets/{only['ticket_id']}", headers=AUTH).json()
    assert shown["status"] == "escalated"

    bad = http.post(f"/v1/dead-letters/{only['job_id']}/requeue", json={}, headers=AUTH)
    assert bad.status_code == 422
    ok = http.post(
        f"/v1/dead-letters/{only['job_id']}/requeue", json={"reviewer": "maya"}, headers=AUTH
    )
    assert ok.status_code == 202
    assert http.get("/v1/dead-letters", headers=AUTH).json() == {"jobs": [], "backlog": 1}
    again = http.post(
        f"/v1/dead-letters/{only['job_id']}/requeue", json={"reviewer": "maya"}, headers=AUTH
    )
    assert again.status_code == 404


def test_dead_letter_routes_need_the_key_and_the_queue(
    seeded_factory: sessionmaker[Session],
) -> None:
    http, _q, _w = queue_app(seeded_factory, [])
    assert http.get("/v1/dead-letters").status_code == 401
    inprocess = TestClient(
        create_app(
            ApiState(
                session_factory=seeded_factory,
                runner=TicketRunner(
                    seeded_factory, ScriptedClient([]), make_gateway(seeded_factory), model_tier="s"
                ),
                executor=Inline(),
                api_key=KEY,
            )
        )
    )
    assert inprocess.get("/v1/dead-letters", headers=AUTH).status_code == 404


def test_embedded_workers_start_and_stop_with_the_app(
    file_factory: sessionmaker[Session],
) -> None:
    http, queue, worker = queue_app(file_factory, [make_response("Delivered.")])
    pool = WorkerPool([worker], poll_s=0.01)
    api_state(http).workers = pool
    with http:  # runs the lifespan: workers start
        done = http.post("/v1/tickets?wait=true", json=HELLO, headers=AUTH)
        assert done.status_code == 200 and done.json()["outcome"] == "drafted"
    assert pool._threads == []  # stopped on shutdown
    assert queue.depth() == 0
