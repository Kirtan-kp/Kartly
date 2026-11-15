"""Alerts: rules over the metrics, announced once, optionally acting on their own (ADR-055).

    python -m kartly.alerts            # check now; exit 1 if anything is firing

``evaluate`` is a pure function from the metrics (see ``kartly.metrics``) to the rules that are
firing. ``AlertManager.check`` compares that with what was firing last time, stores the new state,
and for each change writes an audit row, tells the notifier, and, when automatic actions are on and
the alert is critical, flips the kill switch. A rule that stays bad is announced once, not every
minute. A cleared rule is announced once too. Automatic actions only ever make the system safer
(freeze writes, send tickets to humans); a person lifts them, and an automatic action never
overrides a control a person already set the same way (setting a value it already has changes
nothing).
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from kartly.controls import ControlError, Controls
from kartly.metrics import collect
from kartly.sanitize import clean_str
from kartly.store.models import ActorType, AlertState, AuditLog
from kartly.tools.orders import utc

logger = logging.getLogger("kartly.alerts")

WARNING = "warning"
CRITICAL = "critical"
_RANK = {WARNING: 1, CRITICAL: 2}
ACTOR = "system:alerts"


@dataclass(frozen=True)
class Thresholds:
    """When a rule fires. Rates need a minimum number of events, or one failed run out of two
    would look like a 50% outage."""

    min_runs: int = 10
    min_runs_critical: int = 20
    escalation_rate: float = 0.5
    run_failure_rate: float = 0.1
    run_failure_rate_critical: float = 0.5
    min_llm_calls: int = 10
    llm_error_rate: float = 0.3
    llm_error_rate_critical: float = 0.8
    fallback_share: float = 0.5
    min_tool_calls: int = 10
    tool_error_rate: float = 0.3
    queue_wait_s: float = 300.0
    queue_wait_critical_s: float = 1800.0
    approval_wait_s: float = 3600.0
    cross_customer_access: int = 3
    failed_writes: int = 5


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    action: tuple[str, str] | None = None  # (control, value) taken if actions are enabled


def evaluate(m: Mapping[str, Any], t: Thresholds | None = None) -> list[Finding]:
    t = t or Thresholds()
    found: list[Finding] = []
    runs, llm, queue, approvals = m["runs"], m["llm"], m["queue"], m["approvals"]
    n = runs["total"]
    window_min = round(m["window"]["seconds"] / 60)

    rate = runs["escalation_rate"]
    if n >= t.min_runs and rate is not None and rate > t.escalation_rate:
        found.append(
            Finding(
                "escalation_spike",
                WARNING,
                f"{rate:.0%} of {n} runs in the last {window_min} min ended escalated",
                {"rate": rate, "runs": n},
            )
        )

    fail = runs["failure_rate"]
    if fail is not None:
        if n >= t.min_runs_critical and fail > t.run_failure_rate_critical:
            found.append(
                Finding(
                    "run_failures",
                    CRITICAL,
                    f"{fail:.0%} of {n} runs failed in the last {window_min} min",
                    {"rate": fail, "runs": n},
                    action=("agent", "off"),
                )
            )
        elif n >= t.min_runs and fail > t.run_failure_rate:
            found.append(
                Finding(
                    "run_failures",
                    WARNING,
                    f"{fail:.0%} of {n} runs failed in the last {window_min} min",
                    {"rate": fail, "runs": n},
                )
            )

    calls, err = llm["calls"], llm["error_rate"]
    if calls >= t.min_llm_calls and err is not None:
        if err > t.llm_error_rate_critical:
            sev = CRITICAL
        elif err > t.llm_error_rate:
            sev = WARNING
        else:
            sev = ""
        if sev:
            found.append(
                Finding(
                    "model_errors",
                    sev,
                    f"{err:.0%} of {calls} model calls failed in the last {window_min} min",
                    {"rate": err, "calls": calls},
                )
            )
    if calls >= t.min_llm_calls and llm["fallback_replies"] / calls > t.fallback_share:
        share = llm["fallback_replies"] / calls
        found.append(
            Finding(
                "fallback_heavy",
                WARNING,
                f"{share:.0%} of replies came from a fallback model: the primary is struggling",
                {"share": round(share, 4), "calls": calls},
            )
        )

    bad_tools = {
        name: d["error_rate"]
        for name, d in m["tools"].items()
        if d["calls"] >= t.min_tool_calls
        and d["error_rate"] is not None
        and d["error_rate"] > t.tool_error_rate
    }
    if bad_tools:
        listing = ", ".join(f"{k} {v:.0%}" for k, v in sorted(bad_tools.items()))
        found.append(Finding("tool_errors", WARNING, f"tools failing often: {listing}", bad_tools))

    wait = queue["oldest_queued_age_s"]
    if wait is not None and wait > t.queue_wait_s:
        found.append(
            Finding(
                "queue_stuck",
                CRITICAL if wait > t.queue_wait_critical_s else WARNING,
                f"the oldest waiting job has been queued {wait / 60:.0f} min: workers may be down",
                {"oldest_queued_age_s": wait, "queued": queue["queued"]},
            )
        )
    if queue["dead"] > 0:
        found.append(
            Finding(
                "dead_letters",
                WARNING,
                f"{queue['dead']} ticket(s) the system gave up on are waiting for a person",
                {"dead": queue["dead"]},
            )
        )

    age = approvals["oldest_pending_age_s"]
    if age is not None and age > t.approval_wait_s:
        found.append(
            Finding(
                "approvals_aging",
                WARNING,
                f"{approvals['pending']} approval(s) waiting; the oldest for {age / 3600:.1f} h "
                "(they expire after 24 h)",
                {"pending": approvals["pending"], "oldest_age_s": age},
            )
        )

    crossed = m["audit"].get("cross_customer_access", 0)
    if crossed >= t.cross_customer_access:
        found.append(
            Finding(
                "cross_customer_access",
                CRITICAL,
                f"{crossed} attempt(s) to reach another customer's order in the last "
                f"{window_min} min",
                {"attempts": crossed},
                action=("writes", "frozen"),
            )
        )

    failed = sum(s.get("failed", 0) for s in m["actions"].values())
    if failed >= t.failed_writes:
        found.append(
            Finding(
                "failed_writes",
                CRITICAL,
                f"{failed} write action(s) failed in the last {window_min} min",
                {"failed": failed},
                action=("writes", "frozen"),
            )
        )
    return found


# ------------------------------------------------------------------------------ announcing

Notifier = Callable[["Alert"], None]


@dataclass(frozen=True)
class Alert:
    event: str  # "fired" | "cleared"
    rule: str
    severity: str
    message: str
    action_taken: str | None = None

    def text(self) -> str:
        icon = {"fired": "ALERT" if self.severity == CRITICAL else "warning", "cleared": "cleared"}
        line = f"[kartly {icon[self.event]}] {self.rule}: {self.message}"
        return f"{line} (action: {self.action_taken})" if self.action_taken else line


def log_notifier(alert: Alert) -> None:
    level = (
        logging.ERROR if alert.event == "fired" and alert.severity == CRITICAL else logging.WARNING
    )
    if alert.event == "cleared":
        level = logging.INFO
    logger.log(level, alert.text())


def webhook_notifier(url: str, *, timeout_s: float = 5.0) -> Notifier:
    """POST ``{"text": ...}`` (what Slack-style webhooks take). Never raises: an alert about an
    outage must not die because the chat service is also down."""
    if not url.startswith("https://"):
        raise ValueError("the alert webhook must be an https URL")

    def send(alert: Alert) -> None:
        try:
            httpx.post(url, json={"text": alert.text()}, timeout=timeout_s).raise_for_status()
        except Exception as exc:
            logger.warning("alert webhook failed: %s", type(exc).__name__)

    return send


def combined(*notifiers: Notifier) -> Notifier:
    def send(alert: Alert) -> None:
        for n in notifiers:
            try:
                n(alert)
            except Exception:
                logger.exception("a notifier failed")

    return send


# ------------------------------------------------------------------------------ the manager


@dataclass(frozen=True)
class CheckResult:
    firing: list[Finding]
    fired: list[Alert]
    cleared: list[Alert]


@dataclass(frozen=True)
class AlertView:
    name: str
    status: str
    severity: str
    message: str
    details: dict[str, Any] | None
    since: datetime
    last_checked: datetime


class AlertManager:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        controls: Controls | None = None,
        notifier: Notifier = log_notifier,
        auto_actions: bool = False,
        window_s: float = 900.0,
        thresholds: Thresholds | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._factory = session_factory
        self._controls = controls
        self._notify = notifier
        self._auto = auto_actions
        self._window = window_s
        self._thresholds = thresholds or Thresholds()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()  # one check at a time inside this process

    def check(self) -> CheckResult:
        with self._lock:
            now = self._clock()
            metrics = collect(
                self._factory, window_s=self._window, controls=self._controls, now=now
            )
            firing = evaluate(metrics, self._thresholds)
            fired, cleared = self._store(firing, now)
        taken: dict[str, str] = {}
        for alert in fired:
            finding = next(f for f in firing if f.rule == alert.rule)
            action = self._act(finding) if finding.action else None
            if action:
                taken[alert.rule] = action
        # announce after the state is stored and any action is taken, so the text can say so
        announced = [
            Alert(a.event, a.rule, a.severity, a.message, taken.get(a.rule)) for a in fired
        ] + cleared
        for alert in announced:
            self._notify(alert)
        return CheckResult(firing, announced[: len(fired)], cleared)

    def _store(self, firing: Sequence[Finding], now: datetime) -> tuple[list[Alert], list[Alert]]:
        fired: list[Alert] = []
        cleared: list[Alert] = []
        by_rule = {f.rule: f for f in firing}
        with self._factory() as session:
            states = {s.name: s for s in session.scalars(select(AlertState).with_for_update())}
            for rule, f in by_rule.items():
                state = states.get(rule)
                message = clean_str(f.message, 300)
                if state is None:
                    session.add(
                        AlertState(
                            name=rule,
                            status="firing",
                            severity=f.severity,
                            message=message,
                            details=f.details,
                            since=now,
                            last_checked=now,
                        )
                    )
                    new = True
                else:
                    new = state.status != "firing" or _RANK[f.severity] > _RANK[state.severity]
                    if new:
                        state.since = now
                    state.status, state.severity = "firing", f.severity
                    state.message, state.details, state.last_checked = message, f.details, now
                if new:
                    fired.append(Alert("fired", rule, f.severity, message))
                    self._audit(session, "alert.fired", rule, f.severity, f.details, now)
            for rule, state in states.items():
                if rule in by_rule:
                    continue
                if state.status == "firing":
                    state.status, state.since = "ok", now
                    cleared.append(Alert("cleared", rule, state.severity, "back to normal"))
                    self._audit(session, "alert.cleared", rule, state.severity, None, now)
                state.last_checked = now
            session.commit()
        return fired, cleared

    @staticmethod
    def _audit(
        session: Session,
        action: str,
        rule: str,
        severity: str,
        details: Mapping[str, Any] | None,
        now: datetime,
    ) -> None:
        session.add(
            AuditLog(
                actor_type=ActorType.SYSTEM,
                actor_id="alerts",
                action=action,
                target_type="alert",
                target_id=rule,
                details={"severity": severity, **(details or {})},
                ts=now,
            )
        )

    def _act(self, finding: Finding) -> str | None:
        """Take the rule's automatic action, if allowed. Returns a note for the announcement."""
        if finding.severity != CRITICAL or finding.action is None:
            return None
        control, value = finding.action
        if not self._auto or self._controls is None:
            return f"suggested: set {control} to {value} (automatic actions are off)"
        try:
            _state, changed = self._controls.set(
                control, value, ACTOR, f"alert {finding.rule}: {finding.message}"
            )
        except ControlError:
            logger.exception("automatic action %s=%s failed", control, value)
            return None
        return f"{control} set to {value}" if changed else f"{control} was already {value}"

    def states(self) -> list[AlertView]:
        with self._factory() as session:
            rows = session.scalars(select(AlertState).order_by(AlertState.name))
            return [
                AlertView(
                    s.name,
                    s.status,
                    s.severity,
                    s.message,
                    s.details,
                    utc(s.since),
                    utc(s.last_checked),
                )
                for s in rows
            ]


class AlertLoop:
    """Runs ``check`` every ``interval_s`` in a background thread, inside the API process."""

    def __init__(self, manager: AlertManager, interval_s: float) -> None:
        self._manager = manager
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="alerts", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._manager.check()
            except Exception:  # the database blipped; the next round will try again
                logger.exception("alert check failed")
            self._stop.wait(self._interval)

    def stop(self, timeout_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
            self._thread = None


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.config import get_settings, load_env
    from kartly.store.db import make_engine, make_session_factory

    parser = argparse.ArgumentParser(description="Check the alert rules once.")
    parser.add_argument("--window", type=float, default=None, help="seconds (default: setting)")
    args = parser.parse_args(argv)
    load_env()
    settings = get_settings()
    engine = make_engine()
    try:
        factory = make_session_factory(engine)
        manager = AlertManager(
            factory,
            controls=Controls(factory),
            auto_actions=settings.alert_auto_actions,
            window_s=args.window or settings.alert_window_s,
        )
        result = manager.check()
    finally:
        engine.dispose()
    if not result.firing:
        print("nothing is firing")
        return 0
    for f in result.firing:
        print(f"{f.severity.upper():<8} {f.rule}: {f.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
