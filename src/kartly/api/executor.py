"""Run tickets in the background, with a hard limit on how much work can pile up.

An agent run takes seconds and costs provider quota, so the API must not start an unbounded
number of them (denial of wallet, DESIGN.md section 10). ``BoundedExecutor`` runs a few at a time
and refuses new work once too much is running or waiting; the API turns that into HTTP 503.
This in-process pool stands in for the Redis queue and workers planned for Week 5; callers only
see ``submit`` and ``shutdown``, so the swap does not touch the endpoints.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol, TypeVar

T = TypeVar("T")
logger = logging.getLogger("kartly.api")


class Busy(Exception):
    """Too many runs are in progress or waiting."""


class RunExecutor(Protocol):
    def submit(self, fn: Callable[[], T]) -> Future[T]:
        """Start ``fn`` in the background. Raises Busy when the backlog is full."""
        ...

    def shutdown(self) -> None: ...


class BoundedExecutor:
    def __init__(self, max_workers: int, max_pending: int) -> None:
        if max_workers < 1 or max_pending < max_workers:
            raise ValueError("need max_workers >= 1 and max_pending >= max_workers")
        self._pool = ThreadPoolExecutor(max_workers, thread_name_prefix="kartly-run")
        self._max_workers = max_workers
        self._max_pending = max_pending
        self._pending = 0
        self._lock = threading.Lock()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def max_pending(self) -> int:
        return self._max_pending

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def submit(self, fn: Callable[[], T]) -> Future[T]:
        with self._lock:
            if self._pending >= self._max_pending:
                raise Busy(f"{self._pending} runs already in progress or waiting")
            self._pending += 1
        try:
            future = self._pool.submit(fn)
        except BaseException:
            self._release()
            raise
        future.add_done_callback(self._finished)
        return future

    def _release(self) -> None:
        with self._lock:
            self._pending -= 1

    def _finished(self, future: Future[Any]) -> None:
        self._release()
        error = future.exception() if not future.cancelled() else None
        if error is not None:
            # The runner has already closed the run as failed; this is for the operator's log.
            logger.error("background run crashed", exc_info=error)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)
