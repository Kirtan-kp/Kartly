"""Retries, fallback and the circuit breaker, with a fake clock so nothing really sleeps."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from kartly.llm import (
    AllModelsFailedError,
    AuthenticationError,
    BadRequestError,
    ChatResponse,
    LLMConfigError,
    LLMError,
    LLMTimeoutError,
    ModelNotFoundError,
    ModelRegistry,
    ProviderUnavailableError,
    RateLimitError,
    ResilientClient,
    RetryPolicy,
    ToolUseFailedError,
)
from tests.llm_support import FakeOpenAIServer, make_response, write_models_config


class Clock:
    """Time only moves when a test or a fake sleep moves it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class Fake:
    """Answers from a script: an Exception is raised, a ChatResponse returned."""

    def __init__(self, name: str, *script: ChatResponse | Exception, clock: Clock | None = None):
        self.name = name
        self.script = deque(script)
        self.calls = 0
        self.kwargs: list[dict[str, Any]] = []
        self.took = 0.0  # seconds each failing call takes
        self.clock = clock
        self.closed = False

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.clock is not None:
            self.clock.now += self.took
        item = self.script.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


def down() -> ProviderUnavailableError:
    return ProviderUnavailableError("503")


def reply(model: str = "m") -> ChatResponse:
    return make_response("hello", model=model)


def build(*chain: Fake, clock: Clock | None = None, **policy: Any) -> tuple[ResilientClient, Clock]:
    clock = clock or Clock()
    settings = {"jitter": 0.0, **policy}
    client = ResilientClient(
        chain, RetryPolicy(**settings), sleep=clock.sleep, clock=clock, rng=random.Random(1)
    )
    return client, clock


def ask(client: ResilientClient, **kwargs: Any) -> ChatResponse:
    return client.chat([{"role": "user", "content": "hi"}], **kwargs)


# -------------------------------------------------------------------------- the happy path


def test_a_healthy_model_is_called_once_and_nothing_is_added() -> None:
    primary = Fake("a", reply("a"))
    client, clock = build(primary)
    response = ask(client)
    assert response.model == "a" and response.attempts == 1 and response.failed_over_from == ()
    assert primary.calls == 1 and clock.slept == []


def test_the_request_is_passed_through_unchanged() -> None:
    primary = Fake("a", reply())
    client, _ = build(primary)
    ask(client, tools=[{"x": 1}], tool_choice="auto", temperature=0.3, max_tokens=77, seed=5)
    assert primary.kwargs == [
        {
            "tools": [{"x": 1}],
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_tokens": 77,
            "seed": 5,
            "response_format": None,
        }
    ]


def test_the_name_is_the_primary_and_the_chain_is_listed() -> None:
    client, _ = build(Fake("a"), Fake("b"))
    assert client.name == "a" and client.chain == ("a", "b")


# ---------------------------------------------------------------------------------- retries


@pytest.mark.parametrize("error", [down(), LLMTimeoutError("slow"), RateLimitError("429")])
def test_a_transient_error_is_retried_on_the_same_model(error: LLMError) -> None:
    primary = Fake("a", error, reply("a"))
    client, clock = build(primary)
    response = ask(client)
    assert response.model == "a" and response.attempts == 2 and response.failed_over_from == ()
    assert primary.calls == 2 and clock.slept == [0.5]


def test_the_backoff_doubles_and_is_capped() -> None:
    primary = Fake("a", *[down() for _ in range(5)], reply("a"))
    client, clock = build(primary, max_attempts=6, base_delay_s=1.0, max_delay_s=10.0)
    ask(client)
    assert clock.slept == [1.0, 2.0, 4.0, 8.0, 10.0]


def test_a_wait_exactly_as_long_as_the_time_left_is_not_taken() -> None:
    primary = Fake("a", down())
    client, clock = build(primary, Fake("b", reply("b")), base_delay_s=2.0, deadline_s=2.0)
    assert ask(client).model == "b" and clock.slept == []


def test_jitter_keeps_each_wait_within_the_stated_spread() -> None:
    primary = Fake("a", *[down() for _ in range(20)], reply("a"))
    client, clock = build(
        primary, max_attempts=21, base_delay_s=1.0, max_delay_s=1.0, jitter=0.25, deadline_s=1e6
    )
    ask(client)
    assert all(0.75 <= s <= 1.25 for s in clock.slept) and len(set(clock.slept)) > 1


def test_a_retry_after_hint_is_obeyed_exactly() -> None:
    primary = Fake("a", RateLimitError("429", retry_after=2.5), reply("a"))
    client, clock = build(primary)
    ask(client)
    assert clock.slept == [2.5]


def test_a_retry_after_of_zero_retries_at_once() -> None:
    primary = Fake("a", RateLimitError("429", retry_after=0.0), reply("a"))
    client, clock = build(primary)
    ask(client)
    assert clock.slept == [] and primary.calls == 2


def test_a_long_retry_after_goes_to_the_next_model_without_waiting() -> None:
    primary = Fake("a", RateLimitError("429", retry_after=60.0))
    fallback = Fake("b", reply("b"))
    client, clock = build(primary, fallback)
    response = ask(client)
    assert response.model == "b" and response.failed_over_from == ("a",)
    assert primary.calls == 1 and clock.slept == []


def test_a_retry_after_exactly_at_the_limit_is_still_waited_for() -> None:
    primary = Fake("a", RateLimitError("429", retry_after=15.0), reply("a"))
    client, clock = build(primary, max_retry_after_s=15.0)
    ask(client)
    assert clock.slept == [15.0]


# ---------------------------------------------------------------------------------- fallback


def test_a_model_that_keeps_failing_hands_over_to_the_next() -> None:
    primary = Fake("a", down(), down(), down())
    fallback = Fake("b", reply("b"))
    client, clock = build(primary, fallback)
    response = ask(client)
    assert response.model == "b" and response.failed_over_from == ("a",)
    assert response.attempts == 4  # three failed calls and the one that worked
    assert primary.calls == 3 and fallback.calls == 1
    assert clock.slept == [0.5, 1.0]  # no wait after the last failure on a model


def test_the_chain_is_tried_in_order() -> None:
    a, b, c = Fake("a", down()), Fake("b", down()), Fake("c", reply("c"))
    client, _ = build(a, b, c, max_attempts=1)
    response = ask(client)
    assert response.model == "c" and response.failed_over_from == ("a", "b")
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)


def test_when_every_model_fails_the_error_says_what_happened_to_each() -> None:
    client, _ = build(
        Fake("a", down(), down()),
        Fake("b", LLMTimeoutError("t"), LLMTimeoutError("t")),
        max_attempts=2,
    )
    with pytest.raises(AllModelsFailedError) as caught:
        ask(client)
    error = caught.value
    assert isinstance(error, LLMError) and error.retryable
    assert error.failures == (
        "a: ProviderUnavailableError",
        "a: ProviderUnavailableError",
        "b: LLMTimeoutError",
        "b: LLMTimeoutError",
    )


# ------------------------------------------------------------------------------- fail fast


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("401"),
        ModelNotFoundError("404"),
        BadRequestError("400"),
        ToolUseFailedError("bad call"),
    ],
)
def test_configuration_errors_and_bad_tool_calls_are_raised_untouched(error: LLMError) -> None:
    primary, fallback = Fake("a", error), Fake("b", reply("b"))
    client, clock = build(primary, fallback)
    with pytest.raises(type(error)):
        ask(client)
    assert primary.calls == 1 and fallback.calls == 0 and clock.slept == []


def test_a_bug_does_not_count_against_the_circuit_breaker() -> None:
    primary = Fake("a", *[BadRequestError("400")] * 5)
    client, _ = build(primary, Fake("b"), failure_threshold=2)
    for _ in range(5):
        with pytest.raises(BadRequestError):
            ask(client)
    assert primary.calls == 5 and not client.is_open("a")


# ------------------------------------------------------------------------ circuit breaker


def test_a_model_that_fails_repeatedly_is_skipped_until_it_cools_down() -> None:
    primary = Fake("a", *[down() for _ in range(3)], reply("a"))
    fallback = Fake("b", *[reply("b") for _ in range(5)])
    client, clock = build(primary, fallback, max_attempts=1, failure_threshold=3, cooldown_s=30)
    for _ in range(3):
        assert ask(client).model == "b"
    assert client.is_open("a") and primary.calls == 3

    skipped = ask(client)  # the breaker is open: the primary is not even called
    assert skipped.model == "b" and skipped.failed_over_from == ("a",) and primary.calls == 3

    clock.now += 29.9
    assert client.is_open("a")
    clock.now += 0.1  # exactly 30 seconds: cooled down, so one probe goes through, and it works
    assert ask(client).model == "a" and primary.calls == 4
    assert not client.is_open("a")


def test_the_breaker_opens_only_after_the_threshold_of_consecutive_failures() -> None:
    primary = Fake("a", down(), down(), reply("a"), down(), down(), reply("a"))
    fallback = Fake("b", *[reply("b") for _ in range(4)])
    client, _ = build(primary, fallback, max_attempts=1, failure_threshold=3)
    models = [ask(client).model for _ in range(6)]
    assert models == ["b", "b", "a", "b", "b", "a"]  # a success resets the count
    assert not client.is_open("a")


def test_a_failed_probe_opens_the_breaker_again() -> None:
    primary = Fake("a", down(), down(), down())
    fallback = Fake("b", *[reply("b") for _ in range(6)])
    client, clock = build(primary, fallback, max_attempts=1, failure_threshold=2, cooldown_s=10)
    ask(client)
    ask(client)
    assert client.is_open("a")
    clock.now += 11
    assert ask(client).model == "b" and primary.calls == 3  # the probe failed
    assert client.is_open("a")
    ask(client)
    assert primary.calls == 3  # skipped again, for a fresh cooldown


def test_only_one_call_probes_a_recovering_model_at_a_time() -> None:
    clock = Clock()
    seen: list[bool] = []

    class Probe(Fake):
        def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
            seen.append(client.is_open("a"))  # asked by "another request" during the probe
            return super().chat(messages, **kwargs)

    primary = Probe("a", down(), down(), reply("a"))
    fallback = Fake("b", *[reply("b") for _ in range(3)])
    client, _ = build(
        primary, fallback, clock=clock, max_attempts=1, failure_threshold=2, cooldown_s=10
    )
    ask(client)
    ask(client)
    clock.now += 11
    assert ask(client).model == "a"
    assert seen[-1] is True  # while the probe was running, everyone else was still skipping


def test_a_bug_found_by_the_probe_lets_the_next_call_probe_again() -> None:
    primary = Fake("a", down(), down(), BadRequestError("400"), reply("a"))
    client, clock = build(
        primary,
        Fake("b", reply("b"), reply("b")),
        max_attempts=1,
        failure_threshold=2,
        cooldown_s=5,
    )
    ask(client)
    ask(client)
    clock.now += 6
    with pytest.raises(BadRequestError):
        ask(client)
    assert ask(client).model == "a"  # not stuck "probing" forever


def test_skipped_models_are_listed_when_everything_is_down() -> None:
    primary = Fake("a", down())
    client, _ = build(primary, max_attempts=1, failure_threshold=1)
    with pytest.raises(AllModelsFailedError):
        ask(client)
    with pytest.raises(AllModelsFailedError) as caught:
        ask(client)
    assert "a: skipped" in str(caught.value) and primary.calls == 1


# ---------------------------------------------------------------------------------- deadline


def test_a_wait_that_would_pass_the_deadline_moves_on_instead() -> None:
    primary = Fake("a", down())
    fallback = Fake("b", reply("b"))
    client, clock = build(primary, fallback, base_delay_s=5.0, deadline_s=4.0)
    assert ask(client).model == "b"
    assert clock.slept == [] and primary.calls == 1


def test_no_more_models_are_tried_once_the_deadline_has_passed() -> None:
    clock = Clock()
    primary = Fake("a", down(), clock=clock)
    primary.took = 10.0
    fallback = Fake("b", reply("b"))
    client, _ = build(primary, fallback, clock=clock, max_attempts=1, deadline_s=5.0)
    with pytest.raises(AllModelsFailedError) as caught:
        ask(client)
    assert fallback.calls == 0 and "deadline reached" in str(caught.value)


# -------------------------------------------------------------------------- setup and close


@pytest.mark.parametrize(
    "bad",
    [
        {"max_attempts": 0},
        {"failure_threshold": 0},
        {"base_delay_s": -1},
        {"max_delay_s": -1},
        {"max_retry_after_s": -1},
        {"cooldown_s": -1},
        {"jitter": 1.0},
        {"jitter": -0.1},
        {"deadline_s": 0},
    ],
)
def test_a_nonsense_policy_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**bad)


def test_an_empty_chain_is_refused() -> None:
    with pytest.raises(ValueError):
        ResilientClient([])


def test_closing_closes_every_model_client() -> None:
    a, b = Fake("a"), Fake("b")
    with ResilientClient([a, b]):
        pass
    assert a.closed and b.closed


# ------------------------------------------------------------------------------- registry


def test_the_registry_builds_a_chain_without_duplicates(
    fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    registry = ModelRegistry.from_yaml(config)
    with registry.resilient("fake_small", ["fake_small"]) as client:
        assert client.chain == ("fake_small",)
    with pytest.raises(LLMConfigError, match="unknown model 'nope'"):
        registry.resilient("fake_small", ["nope"])
