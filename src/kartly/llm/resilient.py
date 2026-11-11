"""A model client that survives a flaky provider: retries, a fallback chain and a circuit breaker.

``ResilientClient`` implements ``ChatClient`` over an ordered chain of clients (a primary and its
fallbacks). A model call has no side effects, so repeating it is always safe; tools are never
re-run by this layer.

What it does with each failure (the classes are in ``errors.py``):

- ``RateLimitError``: wait as long as the provider asks (``Retry-After``) when that is short, then
  try again. A long wait is not worth holding a customer for, so it moves on to the next model.
- ``ProviderUnavailableError``, ``LLMTimeoutError``, ``MalformedResponseError``: wait with
  exponential backoff and jitter, try again, then move on to the next model.
- ``ToolUseFailedError``: not handled here. It propagates so the agent loop can re-prompt the model
  (a fallback model would see the same history and often fail the same way).
- ``AuthenticationError``, ``ModelNotFoundError``, ``BadRequestError``: a configuration bug, so it
  fails at once instead of quietly falling back and hiding the bug.

A model that fails ``failure_threshold`` calls in a row (each after its own retries) is skipped for
``cooldown_s`` seconds, so a dead provider costs one slow call, not one per ticket. After the
cooldown one call is let through (the probe); success closes the breaker, failure opens it again.

The whole call stops at ``deadline_s``. When nothing worked it raises ``AllModelsFailedError``,
which lists what happened to every model. The response says which model answered and how many
failures came before it (``ChatResponse.failed_over_from`` and ``attempts``).
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from kartly.llm.errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
    ToolUseFailedError,
)
from kartly.llm.types import ChatClient, ChatResponse

FAIL_FAST = (AuthenticationError, ModelNotFoundError, BadRequestError, ToolUseFailedError)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3  # calls to one model before moving on
    base_delay_s: float = 0.5  # first backoff; doubles each time
    max_delay_s: float = 8.0
    max_retry_after_s: float = 15.0  # a longer Retry-After than this means "use another model"
    jitter: float = 0.25  # +/- this fraction of the delay
    deadline_s: float = 60.0  # for the whole call, all models
    failure_threshold: int = 3
    cooldown_s: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.failure_threshold < 1:
            raise ValueError("max_attempts and failure_threshold must be at least 1")
        if min(self.base_delay_s, self.max_delay_s, self.max_retry_after_s, self.cooldown_s) < 0:
            raise ValueError("delays and the cooldown cannot be negative")
        if not 0 <= self.jitter < 1 or self.deadline_s <= 0:
            raise ValueError("jitter must be in [0, 1) and the deadline positive")


class AllModelsFailedError(LLMError):
    """Every model in the chain failed or was skipped. ``failures`` has one line per model."""

    retryable = True

    def __init__(self, failures: Sequence[str]) -> None:
        super().__init__("no model could answer: " + "; ".join(failures))
        self.failures = tuple(failures)


@dataclass
class _Breaker:
    failures: int = 0
    opened_at: float | None = None
    probing: bool = False


class ResilientClient:
    def __init__(
        self,
        chain: Sequence[ChatClient],
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        if not chain:
            raise ValueError("a resilient client needs at least one model")
        self._chain = tuple(chain)
        self.name = self._chain[0].name  # the primary; ChatResponse.model says who answered
        self._policy = policy or RetryPolicy()
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._breakers = {c.name: _Breaker() for c in self._chain}

    @property
    def chain(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._chain)

    def close(self) -> None:
        for client in self._chain:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> ResilientClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ circuit breaker

    def is_open(self, model: str) -> bool:
        """True while the model is being skipped (it failed repeatedly and is cooling down)."""
        with self._lock:
            return self._skip(model)

    def _skip(self, model: str) -> bool:
        breaker = self._breakers[model]
        if breaker.opened_at is None:
            return False
        if self._clock() - breaker.opened_at < self._policy.cooldown_s:
            return True
        if breaker.probing:  # one probe at a time; others keep skipping
            return True
        breaker.probing = True
        return False

    def _succeeded(self, model: str) -> None:
        with self._lock:
            self._breakers[model] = _Breaker()

    def _failed(self, model: str) -> None:
        with self._lock:
            breaker = self._breakers[model]
            breaker.failures += 1
            breaker.probing = False
            if breaker.failures >= self._policy.failure_threshold:
                breaker.opened_at = self._clock()

    # ------------------------------------------------------------------ the call

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        started = self._clock()
        kwargs: dict[str, Any] = {
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "response_format": response_format,
        }
        failures: list[str] = []
        failed_models: list[str] = []
        calls = 0
        for client in self._chain:
            with self._lock:
                skipped = self._skip(client.name)
            if skipped:
                failures.append(f"{client.name}: skipped (failing, cooling down)")
                failed_models.append(client.name)
                continue
            response, made = self._try_model(client, started, failures, messages, kwargs)
            calls += made
            if response is not None:
                self._succeeded(client.name)
                return replace(response, attempts=calls, failed_over_from=tuple(failed_models))
            failed_models.append(client.name)
            self._failed(client.name)
            if self._clock() - started >= self._policy.deadline_s:
                failures.append("deadline reached")
                break
        raise AllModelsFailedError(failures)

    def _try_model(
        self,
        client: ChatClient,
        started: float,
        failures: list[str],
        messages: Sequence[Mapping[str, Any]],
        kwargs: dict[str, Any],
    ) -> tuple[ChatResponse | None, int]:
        """The response (None if this model failed) and how many calls were made to it."""
        policy = self._policy
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return client.chat(messages, **kwargs), attempt
            except FAIL_FAST:
                with self._lock:  # a bug, not an outage: leave the breaker as it was
                    self._breakers[client.name].probing = False
                raise
            except LLMError as exc:
                failures.append(f"{client.name}: {type(exc).__name__}")
                delay = self._delay(exc, attempt)
                remaining = policy.deadline_s - (self._clock() - started)
                if attempt == policy.max_attempts or delay is None or delay >= remaining:
                    return None, attempt  # not worth waiting: use the next model
                if delay > 0:
                    self._sleep(delay)
        raise AssertionError("unreachable")  # the loop always returns or raises

    def _delay(self, exc: LLMError, attempt: int) -> float | None:
        """Seconds to wait before the next call to the same model; None means don't."""
        policy = self._policy
        if isinstance(exc, RateLimitError) and exc.retry_after is not None:
            if exc.retry_after > policy.max_retry_after_s:
                return None
            return exc.retry_after
        base = min(policy.max_delay_s, policy.base_delay_s * 2 ** (attempt - 1))
        spread = base * policy.jitter
        jittered: float = base + self._rng.uniform(-spread, spread)
        return max(0.0, jittered)
