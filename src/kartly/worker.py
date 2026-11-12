"""Workers that take jobs off the queue and run the agent (ADR-052).

    python -m kartly.worker            # a separate process; needs the same .env as the API

``Worker.run_once`` does one job and is what the tests drive. ``WorkerPool`` runs N workers in
threads (the API starts one when ``KARTLY_EMBEDDED_WORKERS`` is above 0).

A retry must never repeat a side effect. Before running a job for the second time, the worker
looks at the ticket's actions: if an earlier attempt already ran, queued for approval, or left an
action half done, the job is buried for a human instead (``unsafe_to_retry``).
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import socket
import threading
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.orchestrator.loop import StopReason
from kartly.orchestrator.service import RunSummary
from kartly.queue import ClaimedJob, JobQueue
from kartly.sanitize import clean_str
from kartly.store.models import Action, ActionStatus

logger = logging.getLogger("kartly.worker")

_UNSAFE = (ActionStatus.SUCCEEDED, ActionStatus.AWAITING_APPROVAL, ActionStatus.PENDING)


def make_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


def result_of(summary: RunSummary) -> dict[str, Any]:
    return {
        "run_id": str(summary.run_id),
        "outcome": summary.outcome,
        "stop_reason": str(summary.stop_reason),
        "answer": summary.answer,
    }


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        run: Callable[[uuid.UUID], RunSummary],
        session_factory: sessionmaker[Session],
        *,
        worker_id: str | None = None,
    ) -> None:
        self._queue = queue
        self._run = run
        self._factory = session_factory
        self.worker_id = worker_id or make_worker_id()

    def _earlier_attempt_acted(self, ticket_id: uuid.UUID) -> bool:
        with self._factory() as session:
            found = session.scalar(
                select(Action.id)
                .where(Action.ticket_id == ticket_id, Action.status.in_(_UNSAFE))
                .limit(1)
            )
            return found is not None

    def run_once(self) -> bool:
        """Do one job if there is one. Returns whether a job was claimed."""
        job = self._queue.claim(self.worker_id)
        if job is None:
            return False
        self._process(job)
        return True

    def _process(self, job: ClaimedJob) -> None:
        if job.attempts > 1 and self._earlier_attempt_acted(job.ticket_id):
            self._queue.bury(job, "unsafe_to_retry: an earlier attempt already took an action")
            return
        try:
            summary = self._run(job.ticket_id)
        except LookupError as exc:  # the ticket is gone; trying again cannot help
            self._queue.bury(job, f"ticket missing: {exc}")
            return
        except Exception as exc:  # the run closed itself as failed; try again later
            logger.warning(
                "job %s attempt %d crashed: %s", job.id, job.attempts, type(exc).__name__
            )
            self._queue.fail(job, f"{type(exc).__name__}: {clean_str(str(exc), 300)}")
            return
        if summary.stop_reason == StopReason.LLM_ERROR:
            self._queue.fail(job, f"llm_error: {clean_str(summary.detail, 300)}")
            return
        self._queue.complete(job, result_of(summary))

    def run_forever(self, stop: threading.Event, poll_s: float = 1.0) -> None:
        while not stop.is_set():
            try:
                busy = self.run_once()
            except Exception:  # the queue itself (the database) failed; keep the worker alive
                logger.exception("worker %s: queue error", self.worker_id)
                busy = False
            if not busy:
                stop.wait(poll_s)


class WorkerPool:
    def __init__(self, workers: Sequence[Worker], poll_s: float = 1.0) -> None:
        self._workers = list(workers)
        self._poll = poll_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for w in self._workers:
            t = threading.Thread(
                target=w.run_forever, args=(self._stop, self._poll), name=w.worker_id, daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self, timeout_s: float = 150.0) -> None:
        """Finish the jobs in hand, then stop. A job still running at the timeout is left to its
        lease: another worker reclaims it when the lease passes."""
        self._stop.set()
        for t in self._threads:
            t.join(timeout_s)
        self._threads.clear()


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.api.main import build_runtime  # imported here: it pulls in the whole stack

    parser = argparse.ArgumentParser(description="Run Kartly queue workers.")
    parser.add_argument("--workers", type=int, default=None, help="threads (default: max_workers)")
    args = parser.parse_args(argv)
    try:
        runtime = build_runtime()
    except Exception as exc:
        print(f"error: {exc}")
        return 2
    n = args.workers or runtime.settings.max_workers
    pool = WorkerPool(
        [Worker(runtime.queue, runtime.runner.run, runtime.factory) for _ in range(n)],
        runtime.settings.worker_poll_s,
    )
    pool.start()
    print(f"{n} worker(s) running; Ctrl+C to stop")
    with contextlib.suppress(KeyboardInterrupt):
        threading.Event().wait()
    pool.stop()
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
