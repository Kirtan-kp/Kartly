"""The fault-injection scenarios, as tests: every one must end the way it was designed to."""

from __future__ import annotations

import json

import pytest

from kartly.chaos import (
    SCENARIOS,
    FaultyClient,
    HealthyModel,
    Outcome,
    Scenario,
    VirtualTime,
    main,
    render,
    run_scenario,
    violations,
)
from kartly.llm.errors import (
    AuthenticationError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ProviderUnavailableError,
    RateLimitError,
)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_ends_as_designed(scenario: Scenario) -> None:
    outcome = run_scenario(scenario)
    assert outcome.violations == [], render([outcome])


def test_a_refund_is_paid_once_however_often_the_run_is_retried() -> None:
    [crash] = [s for s in SCENARIOS if s.name == "crash_after_refund"]
    outcome = run_scenario(crash)
    assert outcome.refunds == 1 and outcome.attempts == 2
    assert outcome.error is not None and "unsafe_to_retry" in outcome.error


def test_a_short_outage_recovers_because_the_retry_waits_out_the_breaker() -> None:
    [short] = [s for s in SCENARIOS if s.name == "outage_then_recovery"]
    outcome = run_scenario(short)
    assert outcome.ticket == "drafted" and outcome.attempts == 2 and outcome.seconds >= 30


def test_a_long_outage_ends_with_a_human_and_not_a_stuck_ticket() -> None:
    [down] = [s for s in SCENARIOS if s.name == "total_outage"]
    outcome = run_scenario(down)
    assert (outcome.ticket, outcome.job, outcome.attempts) == ("escalated", "dead", 3)


def faulty(fault: str | None) -> tuple[FaultyClient, VirtualTime]:
    time = VirtualTime()
    return FaultyClient("m", HealthyModel("read"), lambda n: fault, time), time


@pytest.mark.parametrize(
    ("fault", "error", "waited"),
    [
        ("429:7", RateLimitError, 0),
        ("5xx", ProviderUnavailableError, 0),
        ("timeout:30", LLMTimeoutError, 30),
        ("malformed", MalformedResponseError, 0),
        ("auth", AuthenticationError, 0),
        ("what", LLMError, 0),
    ],
)
def test_each_fault_raises_the_matching_error(
    fault: str, error: type[Exception], waited: float
) -> None:
    client, time = faulty(fault)
    with pytest.raises(error) as caught:
        client.chat([])
    assert time.t == waited and client.calls == 1
    if fault == "429:7":
        assert isinstance(caught.value, RateLimitError) and caught.value.retry_after == 7


def test_a_slow_fault_answers_late_and_a_healthy_call_does_not_wait() -> None:
    client, time = faulty("slow:20")
    assert client.chat([]).tool_calls
    assert time.t == 20
    ok, ok_time = faulty(None)
    ok.chat([])
    assert ok_time.t == 0


def test_the_healthy_model_looks_the_order_up_and_then_answers() -> None:
    model = HealthyModel("read")
    first = model.reply("m", [{"role": "user", "content": "hi"}])
    assert first.tool_calls[0].name == "get_order"
    second = model.reply("m", [{"role": "tool", "content": "{}"}])
    assert second.content and "delivered" in second.content
    refund = HealthyModel("refund").reply("m", [])
    assert refund.tool_calls[0].name == "issue_refund"


def outcome(**kw: object) -> Outcome:
    base: dict[str, object] = {
        "scenario": "x",
        "story": "s",
        "ticket": "drafted",
        "job": "succeeded",
        "refunds": 0,
        "customer_sees": "reply drafted for review: 'ok'",
    }
    return Outcome(**{**base, **kw})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"ticket": "escalated"}, "ticket ended"),
        ({"job": "dead"}, "job ended"),
        ({"ticket": "resolving", "job": "succeeded"}, "no one is working on"),
        ({"refunds": 2}, "refunds paid"),
        ({"customer_sees": "nothing sent"}, "without an answer"),
    ],
)
def test_the_checker_flags_each_kind_of_bad_ending(
    change: dict[str, object], fragment: str
) -> None:
    found = violations(Scenario("x", "s"), outcome(**change))
    assert any(fragment in f for f in found), found
    assert violations(Scenario("x", "s"), outcome()) == []


def test_the_report_and_command_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--only", "healthy", "blip"]) == 0
    text = capsys.readouterr().out
    assert "2/2 scenarios behaved as designed" in text and "customer:" in text
    assert main(["--only", "healthy", "--json"]) == 0
    [row] = json.loads(capsys.readouterr().out)
    assert row["scenario"] == "healthy" and row["violations"] == []
    assert main(["--only", "nope"]) == 2
    bad = outcome(ticket="resolving", violations=["boom"], error="why")
    shown = render([bad])
    assert "FAIL" in shown and "VIOLATION: boom" in shown and "parked because: why" in shown
