"""The queue on a real Postgres: many workers claiming at once must never share a job."""

from __future__ import annotations

import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.queue import JobQueue
from kartly.store.models import Job, JobStatus, Ticket

pytestmark = pytest.mark.postgres


def test_concurrent_workers_never_claim_the_same_job(pg_factory: sessionmaker[Session]) -> None:
    queue = JobQueue(pg_factory)
    with pg_factory() as session:
        tickets = [Ticket(customer_id="C-0001") for _ in range(30)]
        session.add_all(tickets)
        session.commit()
        ids = [t.id for t in tickets]
    for t in ids:
        queue.enqueue(t)

    def drain(worker: int) -> list[uuid.UUID]:
        mine: list[uuid.UUID] = []
        while (job := queue.claim(f"w{worker}")) is not None:
            mine.append(job.id)
            assert queue.complete(job, {"by": worker})
        return mine

    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = [j for jobs in pool.map(drain, range(8)) for j in jobs]

    assert len(claimed) == 30
    assert max(Counter(claimed).values()) == 1  # nobody got a job twice
    with pg_factory() as session:
        done = session.scalars(select(Job).where(Job.status == JobStatus.SUCCEEDED)).all()
        assert len(done) == 30 and all(j.attempts == 1 for j in done)


def test_a_lease_that_ran_out_is_reclaimed_by_exactly_one_worker(
    pg_factory: sessionmaker[Session],
) -> None:
    from datetime import UTC, datetime, timedelta

    with pg_factory() as session:
        ticket = Ticket(customer_id="C-0001")
        session.add(ticket)
        session.commit()
        ticket_id = ticket.id
    queue = JobQueue(pg_factory, lease_s=30)
    queue.enqueue(ticket_id)
    assert queue.claim("dead-worker") is not None
    later = JobQueue(
        pg_factory, lease_s=30, clock=lambda: datetime.now(UTC) + timedelta(seconds=60)
    )

    with ThreadPoolExecutor(max_workers=6) as pool:
        got = list(pool.map(lambda n: later.claim(f"w{n}"), range(6)))
    winners = [g for g in got if g is not None]
    assert len(winners) == 1 and winners[0].attempts == 2


def test_long_error_texts_fit_their_columns_on_postgres(pg_factory: sessionmaker[Session]) -> None:
    """A cut text ends in an ellipsis; it must still fit String(500) and String(300)."""
    from kartly.controls import Controls
    from kartly.store.models import Job

    with pg_factory() as session:
        ticket = Ticket(customer_id="C-0001")
        session.add(ticket)
        session.commit()
        ticket_id = ticket.id
    queue = JobQueue(pg_factory, max_attempts=1)
    job_id = queue.enqueue(ticket_id)
    job = queue.claim("w")
    assert job is not None
    assert queue.fail(job, "e" * 5000) == "dead"
    with pg_factory() as session:
        row = session.get(Job, job_id)
        assert row is not None and row.last_error is not None and len(row.last_error) == 500
    state, _ = Controls(pg_factory).set("writes", "frozen", "a" * 200, "r" * 2000)
    assert state.reason is not None and len(state.reason) == 300
    assert state.updated_by is not None and len(state.updated_by) == 64
