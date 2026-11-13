"""A durable job queue in the database (ADR-052).

The API turns a ticket into a job; workers claim jobs and run the agent. Everything a queue must
guarantee is done with one table and short transactions:

- **Not lost:** the job is written to the same database as the ticket, so a crash after the ticket
  exists can leave a ticket without a job only if the caller commits them separately, and
  ``enqueue`` takes the caller's session for the case that they must be one transaction.
- **Not run twice at once:** a worker claims a job with a conditional UPDATE (and, on Postgres,
  ``FOR UPDATE SKIP LOCKED`` so workers never wait on each other). The claim sets a lease
  (``locked_until``). Only the worker named in ``locked_by`` can finish the job.
- **Survives a dead worker:** a running job whose lease has passed is claimed again.
- **Retries with backoff, then a dead-letter state:** a failed job waits 30 s, 60 s, 120 s ... and
  after ``max_attempts`` claims it is ``dead``: the ticket goes to a human (escalated) and the job
  stays listed until someone requeues it. A job that keeps killing its worker is buried the same
  way, so one poison ticket cannot crash-loop the pool.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import ColumnElement, CursorResult, and_, func, or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from kartly.sanitize import clean_str
from kartly.store.models import ActorType, AuditLog, Job, JobStatus, Ticket, TicketStatus
from kartly.tools.orders import utc

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_LEASE_S = 300.0  # longer than the run's own 120 s limit, so a live run never loses it
BACKOFF_BASE_S = 30.0  # not shorter than the model breaker's cooldown (see test_queue)
BACKOFF_MAX_S = 300.0
MAX_ERROR_CHARS = 500


def backoff_s(attempt: int) -> float:
    """Seconds to wait before attempt ``attempt + 1``: 30, 60, 120 ... at most 300."""
    return float(min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** max(0, attempt - 1)))


@dataclass(frozen=True)
class ClaimedJob:
    id: uuid.UUID
    ticket_id: uuid.UUID
    attempts: int  # including this one
    max_attempts: int
    worker_id: str


@dataclass(frozen=True)
class JobView:
    id: uuid.UUID
    ticket_id: uuid.UUID
    status: JobStatus
    attempts: int
    max_attempts: int
    last_error: str | None
    result: dict[str, Any] | None
    created_at: datetime
    finished_at: datetime | None


def _view(job: Job) -> JobView:
    return JobView(
        job.id,
        job.ticket_id,
        job.status,
        job.attempts,
        job.max_attempts,
        job.last_error,
        job.result,
        utc(job.created_at),
        utc(job.finished_at) if job.finished_at else None,
    )


class JobQueue:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        lease_s: float = DEFAULT_LEASE_S,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_attempts < 1 or lease_s <= 0:
            raise ValueError("max_attempts must be at least 1 and lease_s positive")
        self._factory = session_factory
        self._max_attempts = max_attempts
        self._lease = timedelta(seconds=lease_s)
        self._clock = clock or (lambda: datetime.now(UTC))

    # ------------------------------------------------------------------ producing

    def enqueue(
        self, ticket_id: uuid.UUID, *, priority: int = 0, session: Session | None = None
    ) -> uuid.UUID:
        """Add a job. With ``session`` the job joins the caller's transaction (and the caller
        commits); without one it is committed on its own."""
        job = Job(
            ticket_id=ticket_id,
            priority=priority,
            max_attempts=self._max_attempts,
            available_at=self._clock(),
        )
        if session is not None:
            session.add(job)
            session.flush()
            return job.id
        with self._factory() as own:
            own.add(job)
            own.commit()
            return job.id

    # ------------------------------------------------------------------ consuming

    @staticmethod
    def _claimable(now: datetime) -> ColumnElement[bool]:
        return or_(
            and_(Job.status == JobStatus.QUEUED, Job.available_at <= now),
            and_(Job.status == JobStatus.RUNNING, Job.locked_until < now),  # the worker died
        )

    def claim(self, worker_id: str) -> ClaimedJob | None:
        """The next job to run, or None. Highest priority first, then oldest."""
        for _ in range(10):  # a few lost races are normal; do not loop forever
            now = self._clock()
            with self._factory() as session:
                candidate = (
                    select(Job.id)
                    .where(self._claimable(now))
                    .order_by(Job.priority.desc(), Job.available_at, Job.created_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
                job_id = session.scalar(candidate)
                if job_id is None:
                    return None
                claimed = cast(
                    CursorResult[Any],
                    session.execute(
                        update(Job)
                        .where(Job.id == job_id, self._claimable(now))
                        .values(
                            status=JobStatus.RUNNING,
                            locked_by=worker_id,
                            locked_until=now + self._lease,
                            attempts=Job.attempts + 1,
                        )
                    ),
                )
                if claimed.rowcount != 1:  # another worker got there first
                    session.rollback()
                    continue
                job = session.get(Job, job_id, populate_existing=True)
                assert job is not None
                if job.attempts > job.max_attempts:  # claimed again and again, never finished
                    self._bury(session, job, "the worker kept dying on this job", now)
                    session.commit()
                    continue
                session.commit()
                return ClaimedJob(job.id, job.ticket_id, job.attempts, job.max_attempts, worker_id)
        return None

    def complete(self, job: ClaimedJob, result: dict[str, Any]) -> bool:
        """Mark the job done. False if the lease was lost (another worker has it now)."""
        with self._factory() as session:
            done = cast(
                CursorResult[Any],
                session.execute(
                    update(Job)
                    .where(
                        Job.id == job.id,
                        Job.status == JobStatus.RUNNING,
                        Job.locked_by == job.worker_id,
                    )
                    .values(
                        status=JobStatus.SUCCEEDED,
                        result=result,
                        finished_at=self._clock(),
                        locked_by=None,
                        locked_until=None,
                        last_error=None,
                    )
                ),
            )
            session.commit()
            return done.rowcount == 1

    def fail(self, job: ClaimedJob, error: str, *, retry: bool = True) -> str:
        """Record a failed attempt: ``"retry"`` (queued again after a wait), ``"dead"`` (out of
        attempts or ``retry=False``) or ``"lost"`` (this worker no longer holds the job)."""
        now = self._clock()
        with self._factory() as session:
            row = session.get(Job, job.id, with_for_update=True, populate_existing=True)
            if row is None or row.status != JobStatus.RUNNING or row.locked_by != job.worker_id:
                return "lost"
            if retry and row.attempts < row.max_attempts:
                row.status = JobStatus.QUEUED
                row.available_at = now + timedelta(seconds=backoff_s(row.attempts))
                row.locked_by = row.locked_until = None
                row.last_error = clean_str(error, MAX_ERROR_CHARS)
                session.commit()
                return "retry"
            self._bury(session, row, error, now)
            session.commit()
            return "dead"

    def _bury(self, session: Session, job: Job, error: str, now: datetime) -> None:
        """Dead-letter a job: a human takes the ticket."""
        job.status = JobStatus.DEAD
        job.last_error = clean_str(error, MAX_ERROR_CHARS)
        job.finished_at = now
        job.locked_by = job.locked_until = None
        ticket = session.get(Ticket, job.ticket_id)
        if ticket is not None and ticket.status not in (TicketStatus.CLOSED, TicketStatus.SENT):
            ticket.status = TicketStatus.ESCALATED
        session.add(
            AuditLog(
                actor_type=ActorType.SYSTEM,
                actor_id="queue",
                action="job.dead",
                ticket_id=job.ticket_id,
                target_type="job",
                target_id=str(job.id),
                details={"attempts": job.attempts, "error": job.last_error},
            )
        )

    def bury(self, job: ClaimedJob, reason: str) -> bool:
        """Dead-letter a job at once, without trying again (a retry would be unsafe)."""
        return self.fail(job, reason, retry=False) == "dead"

    # ------------------------------------------------------------------ operating

    def get(self, job_id: uuid.UUID) -> JobView | None:
        with self._factory() as session:
            job = session.get(Job, job_id, populate_existing=True)
            return _view(job) if job is not None else None

    def depth(self) -> int:
        """Jobs waiting or running: the backlog."""
        with self._factory() as session:
            count = session.scalar(
                select(func.count())
                .select_from(Job)
                .where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            )
            return int(count or 0)

    def dead(self, limit: int = 50) -> list[JobView]:
        with self._factory() as session:
            rows = session.scalars(
                select(Job)
                .where(Job.status == JobStatus.DEAD)
                .order_by(Job.finished_at, Job.id)
                .limit(limit)
            )
            return [_view(j) for j in rows]

    def requeue(self, job_id: uuid.UUID, actor: str) -> bool:
        """Give a dead job a fresh set of attempts. False if it is not dead."""
        now = self._clock()
        with self._factory() as session:
            row = session.get(Job, job_id, with_for_update=True, populate_existing=True)
            if row is None or row.status != JobStatus.DEAD:
                return False
            row.status = JobStatus.QUEUED
            row.attempts = 0
            row.available_at = now
            row.finished_at = None
            session.add(
                AuditLog(
                    actor_type=ActorType.HUMAN,
                    actor_id=clean_str(actor, 64),
                    action="job.requeued",
                    ticket_id=row.ticket_id,
                    target_type="job",
                    target_id=str(row.id),
                    details={"last_error": row.last_error},
                )
            )
            session.commit()
            return True

    def wait(
        self,
        job_id: uuid.UUID,
        timeout_s: float,
        *,
        poll_s: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> JobView | None:
        """Poll until the job has succeeded or died; None if it is still going at the timeout."""
        deadline = monotonic() + timeout_s
        while True:
            view = self.get(job_id)
            if view is None or view.status in (JobStatus.SUCCEEDED, JobStatus.DEAD):
                return view
            if monotonic() >= deadline:
                return None
            sleep(poll_s)
