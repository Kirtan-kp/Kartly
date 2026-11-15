"""Alert rules, their one-announcement-per-change behaviour, and the automatic kill switch."""

from __future__ import annotations

import copy
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.alerts import (
    ACTOR,
    CRITICAL,
    WARNING,
    Alert,
    AlertLoop,
    AlertManager,
    Thresholds,
    combined,
    evaluate,
    log_notifier,
    main,
    webhook_notifier,
)
from kartly.api import ApiState, create_app
from kartly.controls import Controls
from kartly.orchestrator import TicketRunner
from kartly.store.models import ActorType, AlertState, AuditLog
from tests.llm_support import ScriptedClient
from tests.unit.test_api_approvals import AUTH, KEY, Inline, make_gateway

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def healthy() -> dict[str, Any]:
    return {
        "window": {"seconds": 900.0},
        "runs": {"total": 50, "escalation_rate": 0.1, "failure_rate": 0.0},
        "llm": {"calls": 100, "error_rate": 0.01, "fallback_replies": 0},
        "tools": {"get_order": {"calls": 40, "errors": 1, "error_rate": 0.025}},
        "actions": {"issue_refund": {"succeeded": 3}},
        "approvals": {"pending": 0, "oldest_pending_age_s": None, "decided": {}},
        "queue": {"queued": 0, "running": 0, "dead": 0, "oldest_queued_age_s": None},
        "audit": {},
        "controls": {"agent": "on", "writes": "on"},
    }


def with_(**changes: Any) -> dict[str, Any]:
    """healthy metrics with nested changes, e.g. with_(runs={"total": 5})"""
    m = healthy()
    for key, value in changes.items():
        m[key] = {**m[key], **value} if isinstance(value, dict) else value
    return m


def rules(m: dict[str, Any]) -> dict[str, str]:
    return {f.rule: f.severity for f in evaluate(m)}


def test_a_healthy_system_fires_nothing() -> None:
    assert evaluate(healthy()) == []


# ------------------------------------------------------------------------------ the rules


@pytest.mark.parametrize(
    ("runs", "rate", "fires"),
    [(10, 0.51, True), (10, 0.5, False), (9, 0.9, False), (100, 0.9, True)],
)
def test_escalation_spike_needs_enough_runs_and_a_rate_above_half(
    runs: int, rate: float, fires: bool
) -> None:
    m = with_(runs={"total": runs, "escalation_rate": rate})
    assert ("escalation_spike" in rules(m)) is fires


@pytest.mark.parametrize(
    ("runs", "rate", "expected"),
    [
        (10, 0.11, WARNING),
        (10, 0.1, None),
        (9, 0.5, None),
        (20, 0.51, CRITICAL),
        (20, 0.5, WARNING),
        (19, 0.9, WARNING),  # not enough runs for critical, but plenty for a warning
    ],
)
def test_run_failures_warn_then_turn_critical(runs: int, rate: float, expected: str | None) -> None:
    found = rules(with_(runs={"total": runs, "failure_rate": rate}))
    assert found.get("run_failures") == expected


def test_a_critical_run_failure_rate_proposes_switching_the_agent_off() -> None:
    [f] = [
        f
        for f in evaluate(with_(runs={"total": 30, "failure_rate": 0.9}))
        if f.rule == "run_failures"
    ]
    assert f.action == ("agent", "off")
    [w] = [
        f
        for f in evaluate(with_(runs={"total": 30, "failure_rate": 0.2}))
        if f.rule == "run_failures"
    ]
    assert w.action is None


@pytest.mark.parametrize(
    ("calls", "rate", "expected"),
    [
        (10, 0.31, WARNING),
        (10, 0.3, None),
        (9, 0.9, None),
        (10, 0.81, CRITICAL),
        (10, 0.8, WARNING),
    ],
)
def test_model_errors(calls: int, rate: float, expected: str | None) -> None:
    assert rules(with_(llm={"calls": calls, "error_rate": rate})).get("model_errors") == expected


def test_a_primary_that_is_mostly_replaced_by_the_fallback_is_flagged() -> None:
    assert "fallback_heavy" in rules(with_(llm={"calls": 20, "fallback_replies": 11}))
    assert "fallback_heavy" not in rules(with_(llm={"calls": 20, "fallback_replies": 10}))
    assert "fallback_heavy" not in rules(with_(llm={"calls": 5, "fallback_replies": 5}))


def test_failing_tools_are_named() -> None:
    tools = {
        "get_order": {"calls": 40, "errors": 20, "error_rate": 0.5},
        "get_tracking": {"calls": 9, "errors": 9, "error_rate": 1.0},  # too few calls
        "check_return_eligibility": {"calls": 10, "errors": 10, "error_rate": 1.0},  # just enough
        "search_policy": {"calls": 50, "errors": 5, "error_rate": 0.1},
    }
    [f] = [f for f in evaluate(with_(tools=tools)) if f.rule == "tool_errors"]
    assert "get_order 50%" in f.message and "get_tracking" not in f.message
    assert "check_return_eligibility 100%" in f.message


@pytest.mark.parametrize(
    ("wait", "expected"),
    [(None, None), (300.0, None), (301.0, WARNING), (1800.0, WARNING), (1801.0, CRITICAL)],
)
def test_a_job_waiting_too_long_means_the_workers_are_down(
    wait: float | None, expected: str | None
) -> None:
    assert rules(with_(queue={"oldest_queued_age_s": wait})).get("queue_stuck") == expected


def test_dead_letters_and_aging_approvals() -> None:
    assert rules(with_(queue={"dead": 2})) == {"dead_letters": WARNING}
    assert "approvals_aging" not in rules(with_(approvals={"oldest_pending_age_s": 3600.0}))
    aging = with_(approvals={"pending": 2, "oldest_pending_age_s": 3601.0})
    assert rules(aging) == {"approvals_aging": WARNING}


@pytest.mark.parametrize(("attempts", "fires"), [(2, False), (3, True), (10, True)])
def test_cross_customer_access_is_critical_and_proposes_freezing_writes(
    attempts: int, fires: bool
) -> None:
    found = evaluate(with_(audit={"cross_customer_access": attempts}))
    assert bool(found) is fires
    if fires:
        assert found[0].severity == CRITICAL and found[0].action == ("writes", "frozen")


def test_failed_writes_are_critical_at_five() -> None:
    four = with_(actions={"issue_refund": {"failed": 2}, "cancel_order": {"failed": 2}})
    five = with_(actions={"issue_refund": {"failed": 3}, "cancel_order": {"failed": 2}})
    assert evaluate(four) == []
    [f] = evaluate(five)
    assert f.rule == "failed_writes" and f.action == ("writes", "frozen")


def test_thresholds_can_be_changed() -> None:
    m = with_(runs={"total": 4, "escalation_rate": 0.6})
    assert evaluate(m) == []
    assert [f.rule for f in evaluate(m, Thresholds(min_runs=4))] == ["escalation_spike"]


def test_the_messages_say_how_long_the_window_is() -> None:
    [f] = evaluate(with_(runs={"total": 20, "escalation_rate": 0.8}))
    assert "80%" in f.message and "20 runs" in f.message and "15 min" in f.message


# ------------------------------------------------------------------------------ the manager


class Feed:
    """collect() replaced by a list of metrics, one per check."""

    def __init__(self, *metrics: dict[str, Any]) -> None:
        self.metrics = list(metrics)

    def __call__(self, *_a: Any, **_k: Any) -> dict[str, Any]:
        return copy.deepcopy(self.metrics.pop(0) if len(self.metrics) > 1 else self.metrics[0])


def manager(
    factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    feed: Feed,
    *,
    auto: bool = False,
    sent: list[Alert] | None = None,
    controls: Controls | None = None,
) -> AlertManager:
    monkeypatch.setattr("kartly.alerts.collect", feed)
    out = sent if sent is not None else []
    return AlertManager(
        factory,
        controls=controls if controls is not None else Controls(factory),
        notifier=out.append,
        auto_actions=auto,
        clock=lambda: NOW,
    )


BAD = with_(queue={"dead": 1})
CRIT = with_(audit={"cross_customer_access": 5})


def audit_actions(factory: sessionmaker[Session], prefix: str) -> list[AuditLog]:
    with factory() as session:
        return list(
            session.scalars(
                select(AuditLog).where(AuditLog.action.like(f"{prefix}%")).order_by(AuditLog.id)
            )
        )


def test_a_rule_that_stays_bad_is_announced_once_and_then_cleared_once(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Alert] = []
    m = manager(seeded_factory, monkeypatch, Feed(BAD, BAD, BAD, healthy(), healthy()), sent=sent)
    first = m.check()
    assert [f.rule for f in first.firing] == ["dead_letters"]
    assert [a.rule for a in first.fired] == ["dead_letters"]
    assert m.check().fired == [] and m.check().fired == []  # still bad: no new announcement
    cleared = m.check()
    assert [a.rule for a in cleared.cleared] == ["dead_letters"] and cleared.firing == []
    assert m.check().cleared == []
    assert [(a.event, a.rule) for a in sent] == [
        ("fired", "dead_letters"),
        ("cleared", "dead_letters"),
    ]
    assert [a.action for a in audit_actions(seeded_factory, "alert.")] == [
        "alert.fired",
        "alert.cleared",
    ]


def test_the_state_table_follows_the_rules(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    m = manager(seeded_factory, monkeypatch, Feed(BAD, healthy()))
    m.check()
    [s] = m.states()
    assert (s.name, s.status, s.severity, s.since) == ("dead_letters", "firing", WARNING, NOW)
    assert "waiting for a person" in s.message and s.details == {"dead": 1}
    m.check()
    [after] = m.states()
    assert after.status == "ok"


def test_a_rule_that_comes_back_is_announced_again(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[Alert] = []
    m = manager(seeded_factory, monkeypatch, Feed(BAD, healthy(), BAD), sent=sent)
    for _ in range(3):
        m.check()
    assert [a.event for a in sent] == ["fired", "cleared", "fired"]


def test_a_rule_getting_worse_is_announced_again_but_getting_better_is_not(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    warn = with_(queue={"oldest_queued_age_s": 400.0})
    crit = with_(queue={"oldest_queued_age_s": 2000.0})
    sent: list[Alert] = []
    m = manager(seeded_factory, monkeypatch, Feed(warn, crit, warn), sent=sent)
    for _ in range(3):
        m.check()
    assert [(a.event, a.severity) for a in sent] == [("fired", WARNING), ("fired", CRITICAL)]
    [s] = m.states()
    assert s.severity == WARNING  # the stored state follows the latest reading


def test_a_critical_alert_freezes_writes_when_automatic_actions_are_on(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = Controls(seeded_factory)
    sent: list[Alert] = []
    m = manager(seeded_factory, monkeypatch, Feed(CRIT), auto=True, sent=sent, controls=controls)
    result = m.check()
    assert controls.writes_on() is False and controls.agent_on() is True
    assert result.fired[0].action_taken == "writes set to frozen"
    assert "(action: writes set to frozen)" in sent[0].text()
    [event] = audit_actions(seeded_factory, "control.set")
    assert (
        event.actor_id == ACTOR and event.actor_type == ActorType.SYSTEM or event.actor_id == ACTOR
    )
    state = controls.get("writes")
    assert (
        state.updated_by == ACTOR
        and state.reason is not None
        and "cross_customer_access" in state.reason
    )


def test_without_automatic_actions_the_alert_only_suggests_one(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = Controls(seeded_factory)
    m = manager(seeded_factory, monkeypatch, Feed(CRIT), auto=False, controls=controls)
    result = m.check()
    assert controls.writes_on() is True
    note = result.fired[0].action_taken
    assert note is not None and note.startswith("suggested: set writes to frozen")


def test_a_warning_never_acts_even_with_automatic_actions_on(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = Controls(seeded_factory)
    warn = with_(runs={"total": 30, "failure_rate": 0.2})
    m = manager(seeded_factory, monkeypatch, Feed(warn), auto=True, controls=controls)
    result = m.check()
    assert result.fired[0].action_taken is None
    assert controls.agent_on() is True and audit_actions(seeded_factory, "control.set") == []


def test_a_failing_system_switches_the_agent_off_when_allowed(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = Controls(seeded_factory)
    down = with_(runs={"total": 40, "failure_rate": 0.9})
    m = manager(seeded_factory, monkeypatch, Feed(down), auto=True, controls=controls)
    m.check()
    assert controls.agent_on() is False and controls.writes_on() is True


def test_an_action_a_person_already_took_is_not_repeated(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = Controls(seeded_factory)
    controls.set("writes", "frozen", "maya", "already on it")
    m = manager(seeded_factory, monkeypatch, Feed(CRIT), auto=True, controls=controls)
    result = m.check()
    assert result.fired[0].action_taken == "writes was already frozen"
    assert controls.get("writes").updated_by == "maya"  # still hers


def test_the_alert_is_not_lost_when_a_notifier_fails(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def broken(_a: Alert) -> None:
        raise RuntimeError("chat is down")

    notifier = combined(broken, lambda a: seen.append(a.rule))
    monkeypatch.setattr("kartly.alerts.collect", Feed(BAD))
    m = AlertManager(seeded_factory, notifier=notifier, clock=lambda: NOW)
    m.check()
    assert seen == ["dead_letters"]
    assert [s.status for s in m.states()] == ["firing"]


def test_the_real_metrics_drive_the_alerts_end_to_end(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        for _ in range(3):
            session.add(
                AuditLog(
                    actor_type=ActorType.SYSTEM,
                    action="cross_customer_access",
                    ts=NOW - timedelta(minutes=2),
                )
            )
        session.commit()
    controls = Controls(seeded_factory)
    m = AlertManager(seeded_factory, controls=controls, auto_actions=True, clock=lambda: NOW)
    assert [f.rule for f in m.check().firing] == ["cross_customer_access"]
    assert controls.writes_on() is False


def test_the_message_is_cut_to_fit_the_column(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {
        f"tool_{i:03d}_{'x' * 20}": {"calls": 50, "errors": 40, "error_rate": 0.8}
        for i in range(30)
    }
    m = manager(seeded_factory, monkeypatch, Feed(with_(tools=tools)))
    m.check()
    with seeded_factory() as session:
        row = session.get(AlertState, "tool_errors")
        assert row is not None and len(row.message) <= 300


# ------------------------------------------------------------------------------ announcing


def test_alert_text_reads_plainly() -> None:
    assert Alert("fired", "r", CRITICAL, "m").text() == "[kartly ALERT] r: m"
    assert Alert("fired", "r", WARNING, "m").text() == "[kartly warning] r: m"
    assert Alert("cleared", "r", CRITICAL, "back").text() == "[kartly cleared] r: back"


def test_the_log_notifier_uses_levels(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="kartly.alerts"):
        log_notifier(Alert("fired", "a", CRITICAL, "x"))
        log_notifier(Alert("fired", "b", WARNING, "x"))
        log_notifier(Alert("cleared", "c", CRITICAL, "x"))
    assert [r.levelname for r in caplog.records] == ["ERROR", "WARNING", "INFO"]


def test_the_webhook_needs_https_and_posts_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        webhook_notifier("http://hooks.example/x")
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr("kartly.alerts.httpx.post", post)
    webhook_notifier("https://hooks.example/x")(Alert("fired", "r", CRITICAL, "m"))
    assert calls == [
        {"url": "https://hooks.example/x", "json": {"text": "[kartly ALERT] r: m"}, "timeout": 5.0}
    ]


def test_a_failing_webhook_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("no route")

    monkeypatch.setattr("kartly.alerts.httpx.post", post)
    webhook_notifier("https://hooks.example/x")(Alert("fired", "r", WARNING, "m"))
    assert "alert webhook failed: ConnectError" in caplog.text


def test_a_webhook_error_status_is_swallowed_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def post(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(500, request=httpx.Request("POST", url))

    monkeypatch.setattr("kartly.alerts.httpx.post", post)
    webhook_notifier("https://hooks.example/x")(Alert("fired", "r", WARNING, "m"))  # no raise


# ------------------------------------------------------------------------------ the timer


class CountingManager:
    def __init__(self, fail_first: bool = False) -> None:
        self.calls = 0
        self.fail_first = fail_first
        self.enough = threading.Event()

    def check(self) -> None:
        self.calls += 1
        if self.calls >= 3:
            self.enough.set()
        if self.fail_first and self.calls == 1:
            raise RuntimeError("database restarted")


@pytest.mark.parametrize("fail_first", [False, True])
def test_the_loop_checks_repeatedly_survives_errors_and_stops(fail_first: bool) -> None:
    fake = CountingManager(fail_first)
    loop = AlertLoop(fake, 0.01)  # type: ignore[arg-type]
    loop.start()
    assert fake.enough.wait(5)
    loop.stop(5)
    done = fake.calls
    threading.Event().wait(0.1)
    assert fake.calls == done  # stopped


# ------------------------------------------------------------------------------ API and CLI


def alerts_app(factory: sessionmaker[Session], manager_: AlertManager | None) -> TestClient:
    runner = TicketRunner(factory, ScriptedClient([]), make_gateway(factory), model_tier="s")
    state = ApiState(
        session_factory=factory,
        runner=runner,
        executor=Inline(),
        api_key=KEY,
        alerts=manager_,
    )
    return TestClient(create_app(state))


def test_alert_routes_need_the_key_and_a_configured_manager(
    seeded_factory: sessionmaker[Session],
) -> None:
    http = alerts_app(seeded_factory, None)
    assert http.get("/v1/alerts").status_code == 401
    assert http.post("/v1/alerts/check").status_code == 401
    assert http.get("/v1/alerts", headers=AUTH).status_code == 503
    assert http.post("/v1/alerts/check", headers=AUTH).status_code == 503


def test_the_api_lists_and_checks_alerts(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    m = manager(seeded_factory, monkeypatch, Feed(BAD, healthy()))
    http = alerts_app(seeded_factory, m)
    assert http.get("/v1/alerts", headers=AUTH).json() == {"alerts": []}
    checked = http.post("/v1/alerts/check", headers=AUTH).json()
    assert checked == {"firing": ["dead_letters"], "fired": ["dead_letters"], "cleared": []}
    [row] = http.get("/v1/alerts", headers=AUTH).json()["alerts"]
    assert (
        row["name"] == "dead_letters" and row["status"] == "firing" and row["severity"] == "warning"
    )
    after = http.post("/v1/alerts/check", headers=AUTH).json()
    assert after == {"firing": [], "fired": [], "cleared": ["dead_letters"]}


def test_the_app_runs_the_alert_loop_with_its_lifespan(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = CountingManager()
    runner = TicketRunner(
        seeded_factory, ScriptedClient([]), make_gateway(seeded_factory), model_tier="s"
    )
    loop = AlertLoop(fake, 0.01)  # type: ignore[arg-type]
    state = ApiState(
        session_factory=seeded_factory,
        runner=runner,
        executor=Inline(),
        api_key=KEY,
        alert_loop=loop,
    )
    with TestClient(create_app(state)):
        assert fake.enough.wait(5)
    done = fake.calls
    threading.Event().wait(0.1)
    assert fake.calls == done  # stopped with the app


def test_the_command_line_prints_what_is_firing(
    seeded_factory: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from kartly.config import Settings

    monkeypatch.setattr("kartly.store.db.make_engine", lambda *a, **k: seeded_factory.kw["bind"])
    monkeypatch.setattr("sqlalchemy.engine.Engine.dispose", lambda self, *a, **k: None)
    monkeypatch.setattr("kartly.config.load_env", lambda: None)
    monkeypatch.setattr("kartly.config.get_settings", lambda: Settings(_env_file=None))  # type: ignore[call-arg]
    assert main([]) == 0
    assert "nothing is firing" in capsys.readouterr().out
    monkeypatch.setattr("kartly.alerts.collect", Feed(CRIT))
    assert main(["--window", "60"]) == 1
    assert "CRITICAL cross_customer_access" in capsys.readouterr().out


def test_only_a_critical_finding_may_act_whatever_its_rule_proposes(
    seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from kartly.alerts import Finding

    monkeypatch.setattr(
        "kartly.alerts.evaluate",
        lambda *_a, **_k: [
            Finding("odd", WARNING, "a warning that asks for a freeze", {}, ("writes", "frozen"))
        ],
    )
    controls = Controls(seeded_factory)
    m = manager(seeded_factory, monkeypatch, Feed(healthy()), auto=True, controls=controls)
    controls = Controls(seeded_factory)
    m = manager(seeded_factory, monkeypatch, Feed(healthy()), auto=True, controls=controls)
    monkeypatch.setattr(
        "kartly.alerts.evaluate",
        lambda *_a, **_k: [Finding("odd", WARNING, "x", {}, ("writes", "frozen"))],
    )
    result = m.check()
    assert result.fired[0].action_taken is None and controls.writes_on() is True


def test_the_loop_waits_between_checks() -> None:
    fake = CountingManager()
    loop = AlertLoop(fake, 0.5)  # type: ignore[arg-type]
    loop.start()
    threading.Event().wait(0.2)
    loop.stop(5)
    assert fake.calls == 1  # one check at once, the next not for half a second
