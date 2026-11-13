"""The durable job queue: claiming, leases, retries with backoff, dead letters, requeue."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.queue import JobQueue, backoff_s
from kartly.store.models import (
    ActorType,
    AuditLog,
    Job,
    JobStatus,
    Ticket,
    TicketStatus,
)

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def make_ticket(
    factory: sessionmaker[Session], status: TicketStatus = TicketStatus.RECEIVED
) -> uuid.UUID:
    with factory() as session:
        ticket = Ticket(customer_id="C-0001", status=status)
        session.add(ticket)
        session.commit()
        return ticket.id


def make_queue(
    factory: sessionmaker[Session], clock: Clock, *, max_attempts: int = 3, lease_s: float = 60
) -> JobQueue:
    return JobQueue(factory, max_attempts=max_attempts, lease_s=lease_s, clock=clock)


def job_row(factory: sessionmaker[Session], job_id: uuid.UUID) -> Job:
    with factory() as session:
        row = session.get(Job, job_id)
        assert row is not None
        return row


def audit(factory: sessionmaker[Session], action: str) -> list[AuditLog]:
    with factory() as session:
        return list(session.scalars(select(AuditLog).where(AuditLog.action == action)))


def test_backoff_doubles_from_five_seconds_and_is_capped() -> None:
    assert [backoff_s(n) for n in (1, 2, 3, 4)] == [30, 60, 120, 240]
    assert backoff_s(5) == 300 and backoff_s(50) == 300
    assert backoff_s(0) == 30


def test_the_first_backoff_outlasts_the_model_breakers_cooldown() -> None:
    """Else a short outage uses up every attempt while the breaker still refuses to try."""
    from kartly.llm.resilient import RetryPolicy

    assert backoff_s(1) >= RetryPolicy().cooldown_s


def test_bad_settings_are_refused(seeded_factory: sessionmaker[Session]) -> None:
    with pytest.raises(ValueError):
        JobQueue(seeded_factory, max_attempts=0)
    with pytest.raises(ValueError):
        JobQueue(seeded_factory, lease_s=0)


def test_a_job_is_claimed_once_and_the_claim_counts_an_attempt(
    seeded_factory: sessionmaker[Session],
) -> None:
    q = make_queue(seeded_factory, Clock())
    t = make_ticket(seeded_factory)
    job_id = q.enqueue(t)

    claimed = q.claim("w1")
    assert claimed is not None
    assert (claimed.id, claimed.ticket_id, claimed.attempts, claimed.max_attempts) == (
        job_id,
        t,
        1,
        3,
    )
    assert q.claim("w2") is None  # running, lease not over
    row = job_row(seeded_factory, job_id)
    assert row.status == JobStatus.RUNNING and row.locked_by == "w1"


def test_nothing_to_claim_on_an_empty_queue(seeded_factory: sessionmaker[Session]) -> None:
    assert make_queue(seeded_factory, Clock()).claim("w1") is None


def test_a_job_that_is_not_due_yet_waits(seeded_factory: sessionmaker[Session]) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock)
    q.enqueue(make_ticket(seeded_factory))
    job = q.claim("w1")
    assert job is not None
    assert q.fail(job, "boom") == "retry"
    assert q.claim("w1") is None  # 30 s of backoff
    clock.advance(29.9)
    assert q.claim("w1") is None
    clock.advance(0.2)
    again = q.claim("w1")
    assert again is not None and again.attempts == 2


def test_priority_then_age_decides_the_order(seeded_factory: sessionmaker[Session]) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock)
    old = q.enqueue(make_ticket(seeded_factory))
    clock.advance(1)
    new = q.enqueue(make_ticket(seeded_factory))
    urgent = q.enqueue(make_ticket(seeded_factory), priority=5)
    clock.advance(1)
    order = []
    for _ in range(3):
        job = q.claim("w")
        assert job is not None
        order.append(job.id)
    assert order == [urgent, old, new]


def test_complete_stores_the_result_and_frees_the_lock(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock)
    job_id = q.enqueue(make_ticket(seeded_factory))
    job = q.claim("w1")
    assert job is not None
    assert q.complete(job, {"outcome": "drafted"}) is True
    row = job_row(seeded_factory, job_id)
    assert row.status == JobStatus.SUCCEEDED and row.result == {"outcome": "drafted"}
    assert row.locked_by is None and row.finished_at is not None
    assert q.claim("w2") is None and q.depth() == 0


def test_a_worker_that_lost_its_lease_cannot_finish_the_job(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock, lease_s=60)
    job_id = q.enqueue(make_ticket(seeded_factory))
    slow = q.claim("slow")
    assert slow is not None
    clock.advance(61)  # the lease ran out; another worker takes over
    taker = q.claim("taker")
    assert taker is not None and taker.attempts == 2

    assert q.complete(slow, {"outcome": "x"}) is False
    assert q.fail(slow, "late") == "lost"
    assert job_row(seeded_factory, job_id).status == JobStatus.RUNNING
    assert q.complete(taker, {"outcome": "y"}) is True


def test_a_running_job_inside_its_lease_is_not_reclaimed(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock, lease_s=60)
    q.enqueue(make_ticket(seeded_factory))
    assert q.claim("a") is not None
    clock.advance(60)  # exactly at the end is still held
    assert q.claim("b") is None
    clock.advance(0.1)
    assert q.claim("b") is not None


def test_retries_run_out_and_the_job_goes_dead_with_the_ticket_escalated(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock, max_attempts=3)
    t = make_ticket(seeded_factory, TicketStatus.RESOLVING)
    job_id = q.enqueue(t)
    outcomes = []
    for _ in range(3):
        job = q.claim("w")
        assert job is not None
        outcomes.append(q.fail(job, "provider down"))
        clock.advance(400)
    assert outcomes == ["retry", "retry", "dead"]

    row = job_row(seeded_factory, job_id)
    assert row.status == JobStatus.DEAD and row.last_error == "provider down"
    assert row.finished_at is not None
    with seeded_factory() as session:
        ticket = session.get(Ticket, t)
        assert ticket is not None and ticket.status == TicketStatus.ESCALATED
    [event] = audit(seeded_factory, "job.dead")
    assert event.actor_type == ActorType.SYSTEM and event.ticket_id == t
    assert event.details == {"attempts": 3, "error": "provider down"}
    assert q.claim("w") is None  # dead jobs are not claimed


def test_a_job_that_keeps_killing_its_worker_is_buried(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock, max_attempts=2, lease_s=10)
    job_id = q.enqueue(make_ticket(seeded_factory))
    for _ in range(2):  # two workers claim it and die without a word
        assert q.claim("doomed") is not None
        clock.advance(11)
    assert q.claim("third") is None  # the third claim sees attempts > max and buries it
    assert job_row(seeded_factory, job_id).status == JobStatus.DEAD
    assert "kept dying" in (job_row(seeded_factory, job_id).last_error or "")
    assert len(audit(seeded_factory, "job.dead")) == 1


def test_a_closed_ticket_is_not_reopened_when_its_job_dies(
    seeded_factory: sessionmaker[Session],
) -> None:
    q = make_queue(seeded_factory, Clock())
    t = make_ticket(seeded_factory, TicketStatus.CLOSED)
    q.enqueue(t)
    job = q.claim("w")
    assert job is not None
    assert q.bury(job, "no") is True
    with seeded_factory() as session:
        ticket = session.get(Ticket, t)
        assert ticket is not None and ticket.status == TicketStatus.CLOSED


def test_bury_skips_the_remaining_attempts(seeded_factory: sessionmaker[Session]) -> None:
    q = make_queue(seeded_factory, Clock(), max_attempts=5)
    job_id = q.enqueue(make_ticket(seeded_factory))
    job = q.claim("w")
    assert job is not None
    assert q.bury(job, "unsafe") is True
    assert job_row(seeded_factory, job_id).status == JobStatus.DEAD


def test_requeue_gives_a_dead_job_a_fresh_set_of_attempts(
    seeded_factory: sessionmaker[Session],
) -> None:
    clock = Clock()
    q = make_queue(seeded_factory, clock, max_attempts=1)
    job_id = q.enqueue(make_ticket(seeded_factory))
    job = q.claim("w")
    assert job is not None
    assert q.fail(job, "x") == "dead"
    assert [d.id for d in q.dead()] == [job_id]

    assert q.requeue(job_id, "human:maya") is True
    assert q.dead() == []
    row = job_row(seeded_factory, job_id)
    assert row.status == JobStatus.QUEUED and row.attempts == 0 and row.finished_at is None
    again = q.claim("w")
    assert again is not None and again.attempts == 1
    [event] = audit(seeded_factory, "job.requeued")
    assert event.actor_type == ActorType.HUMAN and event.actor_id == "human:maya"


def test_only_dead_jobs_can_be_requeued(seeded_factory: sessionmaker[Session]) -> None:
    q = make_queue(seeded_factory, Clock())
    job_id = q.enqueue(make_ticket(seeded_factory))
    assert q.requeue(job_id, "human:maya") is False  # queued
    assert q.requeue(uuid.uuid4(), "human:maya") is False  # unknown
    assert audit(seeded_factory, "job.requeued") == []


def test_depth_counts_waiting_and_running_only(seeded_factory: sessionmaker[Session]) -> None:
    q = make_queue(seeded_factory, Clock())
    q.enqueue(make_ticket(seeded_factory))
    q.enqueue(make_ticket(seeded_factory))
    assert q.depth() == 2
    job = q.claim("w")
    assert job is not None and q.depth() == 2
    q.complete(job, {})
    assert q.depth() == 1


def test_enqueue_can_join_the_callers_transaction(seeded_factory: sessionmaker[Session]) -> None:
    q = make_queue(seeded_factory, Clock())
    t = make_ticket(seeded_factory)
    with seeded_factory() as session:
        q.enqueue(t, session=session)
        session.rollback()
    assert q.depth() == 0
    with seeded_factory() as session:
        q.enqueue(t, session=session)
        session.commit()
    assert q.depth() == 1


def test_wait_returns_when_the_job_is_finished_or_the_time_is_up(
    seeded_factory: sessionmaker[Session],
) -> None:
    q = make_queue(seeded_factory, Clock())
    job_id = q.enqueue(make_ticket(seeded_factory))
    ticks = iter(range(100))
    slept: list[float] = []
    assert q.wait(job_id, 1.0, sleep=slept.append, monotonic=lambda: next(ticks) * 0.4) is None
    assert slept  # it polled

    job = q.claim("w")
    assert job is not None
    q.complete(job, {"outcome": "drafted"})
    view = q.wait(job_id, 1.0, sleep=lambda _s: None)
    assert view is not None and view.status == JobStatus.SUCCEEDED
    assert view.result == {"outcome": "drafted"}
    assert q.wait(uuid.uuid4(), 1.0) is None
