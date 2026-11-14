"""Operational metrics: exact counts from the database, windowed, and as Prometheus text."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from kartly.api import ApiState, create_app
from kartly.controls import Controls
from kartly.metrics import collect, percentile, to_prometheus
from kartly.orchestrator import TicketRunner
from kartly.store.models import (
    Action,
    ActionStatus,
    ActorType,
    Approval,
    ApprovalStatus,
    AuditLog,
    Job,
    JobStatus,
    Run,
    RunMode,
    RunStatus,
    RunStep,
    StepKind,
    Ticket,
    TicketStatus,
)
from tests.llm_support import ScriptedClient
from tests.support import FIXED_NOW
from tests.unit.test_api_approvals import AUTH, KEY, Inline, make_gateway

NOW = FIXED_NOW
HOUR = timedelta(hours=1)


def add_run(
    session: Session,
    ticket: Ticket,
    *,
    ago: timedelta = timedelta(minutes=5),
    outcome: str = "drafted",
    status: RunStatus = RunStatus.SUCCEEDED,
    tokens: tuple[int, int] = (100, 20),
    cost: str = "0.001",
    steps: Sequence[dict[str, Any]] = (),
) -> Run:
    run = Run(
        ticket_id=ticket.id,
        mode=RunMode.SHADOW,
        status=status,
        outcome=outcome,
        tokens_in=tokens[0],
        tokens_out=tokens[1],
        shadow_cost_usd=Decimal(cost),
        started_at=NOW - ago,
    )
    session.add(run)
    session.flush()
    for i, step in enumerate(steps):
        session.add(
            RunStep(
                run_id=run.id,
                idx=i,
                kind=step.get("kind", StepKind.LLM_CALL),
                name=step.get("name", "chat"),
                model=step.get("model", "m1"),
                latency_ms=step.get("latency_ms", 100),
                tokens_in=step.get("tokens_in", 0),
                tokens_out=step.get("tokens_out", 0),
                error=step.get("error"),
                output=step.get("output"),
                created_at=NOW - ago,
            )
        )
    return run


def new_ticket(session: Session, status: TicketStatus = TicketStatus.DRAFTED) -> Ticket:
    ticket = Ticket(customer_id="C-0002", status=status)
    session.add(ticket)
    session.flush()
    return ticket


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 0.5) is None
    assert percentile([7], 0.95) == 7
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert percentile(values, 0.5) == 50  # the 5th of 10
    assert percentile(values, 0.95) == 100  # the 10th
    assert percentile(values, 0.9) == 90
    assert percentile([3, 1, 2], 0.5) == 2  # order does not matter


def test_an_empty_database_reports_zeros_and_no_rates(
    seeded_factory: sessionmaker[Session],
) -> None:
    m = collect(seeded_factory, now=NOW)
    assert m["runs"]["total"] == 0 and m["runs"]["escalation_rate"] is None
    assert m["runs"]["latency_ms"] == {"p50": None, "p95": None, "sampled_runs": 0}
    assert m["llm"]["calls"] == 0 and m["llm"]["error_rate"] is None
    assert m["approvals"] == {"pending": 0, "oldest_pending_age_s": None, "decided": {}}
    assert m["queue"]["queued"] == 0 and m["queue"]["oldest_queued_age_s"] is None
    assert m["controls"] == {} and m["tools"] == {} and m["actions"] == {}


def test_a_window_must_be_positive(seeded_factory: sessionmaker[Session]) -> None:
    with pytest.raises(ValueError):
        collect(seeded_factory, window_s=0)


def test_runs_are_counted_in_the_window_only(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        add_run(session, t, outcome="drafted")
        add_run(session, t, outcome="drafted", ago=timedelta(minutes=59))
        add_run(session, t, outcome="escalated", status=RunStatus.SUCCEEDED)
        add_run(session, t, outcome="escalated", status=RunStatus.FAILED, ago=timedelta(minutes=1))
        add_run(session, t, outcome="drafted", ago=timedelta(hours=2))  # outside the hour
        session.commit()
    m = collect(seeded_factory, window_s=3600, now=NOW)["runs"]
    assert m["total"] == 4
    assert m["by_outcome"] == {"drafted": 2, "escalated": 2}
    assert m["by_status"] == {"succeeded": 3, "failed": 1}
    assert m["escalation_rate"] == 0.5 and m["failure_rate"] == 0.25
    assert (m["tokens_in"], m["tokens_out"]) == (400, 80)
    assert m["shadow_cost_usd"] == "0.004000" and m["avg_shadow_cost_usd"] == "0.001000"
    wide = collect(seeded_factory, window_s=3 * 3600, now=NOW)["runs"]
    assert wide["total"] == 5 and wide["tokens_in"] == 500


def test_run_latency_is_the_time_spent_in_calls(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        for total in (100, 200, 300, 400, 1000):
            half = total // 2
            add_run(
                session,
                t,
                steps=[
                    {"latency_ms": half},
                    {"kind": StepKind.TOOL_CALL, "name": "get_order", "latency_ms": total - half},
                ],
            )
        session.commit()
    lat = collect(seeded_factory, now=NOW)["runs"]["latency_ms"]
    assert lat == {"p50": 300.0, "p95": 1000.0, "sampled_runs": 5}


def test_model_calls_errors_retries_and_fallbacks(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        add_run(
            session,
            t,
            steps=[
                {"model": "big", "tokens_in": 10, "tokens_out": 5, "output": {"content": "a"}},
                {"model": "big", "error": "ProviderUnavailableError: 503"},
                {"model": "small", "output": {"attempts": 3}},
                {"model": "small", "output": {"attempts": 4, "failed_over_from": ["big"]}},
                {"kind": StepKind.TOOL_CALL, "name": "get_order", "model": None},
            ],
        )
        session.commit()
    llm = collect(seeded_factory, now=NOW)["llm"]
    assert llm["calls"] == 4 and llm["errors"] == 1 and llm["error_rate"] == 0.25
    assert llm["retried_replies"] == 2 and llm["fallback_replies"] == 1
    assert llm["by_model"]["big"]["calls"] == 2 and llm["by_model"]["big"]["errors"] == 1
    assert llm["by_model"]["big"]["tokens_in"] == 10 and llm["by_model"]["small"]["calls"] == 2


def test_tool_calls_and_write_actions(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        run = add_run(
            session,
            t,
            steps=[
                {"kind": StepKind.TOOL_CALL, "name": "get_order", "model": None},
                {"kind": StepKind.TOOL_CALL, "name": "get_order", "model": None, "error": "x"},
                {"kind": StepKind.TOOL_CALL, "name": "get_tracking", "model": None},
            ],
        )
        for status in (ActionStatus.SUCCEEDED, ActionStatus.SUCCEEDED, ActionStatus.REJECTED):
            session.add(
                Action(
                    run_id=run.id,
                    ticket_id=t.id,
                    tool_name="issue_refund",
                    args={},
                    idempotency_key=uuid.uuid4().hex,
                    status=status,
                )
            )
        session.commit()
    m = collect(seeded_factory, now=NOW + timedelta(seconds=1))
    assert m["tools"]["get_order"] == {"calls": 2, "errors": 1, "error_rate": 0.5}
    assert m["tools"]["get_tracking"]["error_rate"] == 0.0
    assert m["actions"] == {"issue_refund": {"succeeded": 2, "rejected": 1}}


def test_approvals_pending_age_and_decisions(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as session:
        t = new_ticket(session, TicketStatus.AWAITING_APPROVAL)
        run = add_run(session, t)
        for status, age in [
            (ApprovalStatus.PENDING, timedelta(minutes=10)),
            (ApprovalStatus.PENDING, timedelta(minutes=3)),
            (ApprovalStatus.APPROVED, timedelta(minutes=20)),
            (ApprovalStatus.REJECTED, timedelta(minutes=30)),
            (ApprovalStatus.APPROVED, timedelta(hours=5)),  # decided before the window
        ]:
            action = Action(
                run_id=run.id,
                ticket_id=t.id,
                tool_name="issue_refund",
                args={},
                idempotency_key=uuid.uuid4().hex,
                status=ActionStatus.AWAITING_APPROVAL,
            )
            session.add(action)
            session.flush()
            session.add(
                Approval(
                    action_id=action.id,
                    ticket_id=t.id,
                    payload_hash="h",
                    reason="shadow_mode",
                    status=status,
                    requested_at=NOW - age,
                    expires_at=NOW + HOUR,
                )
            )
        session.commit()
    a = collect(seeded_factory, window_s=3600, now=NOW)["approvals"]
    assert a["pending"] == 2 and a["oldest_pending_age_s"] == 600.0
    assert a["decided"] == {"approved": 1, "rejected": 1}


def test_queue_depth_dead_letters_and_the_oldest_wait(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        for status, wait in [
            (JobStatus.QUEUED, timedelta(seconds=90)),
            (JobStatus.QUEUED, timedelta(seconds=30)),
            (JobStatus.QUEUED, timedelta(seconds=-60)),  # not due yet: age is not negative
            (JobStatus.RUNNING, timedelta(seconds=10)),
            (JobStatus.DEAD, timedelta(seconds=500)),
            (JobStatus.SUCCEEDED, timedelta(seconds=500)),
        ]:
            session.add(Job(ticket_id=t.id, status=status, available_at=NOW - wait))
        session.commit()
    q = collect(seeded_factory, now=NOW)["queue"]
    assert (q["queued"], q["running"], q["dead"], q["succeeded"]) == (3, 1, 1, 1)
    assert q["oldest_queued_age_s"] == 90.0


def test_a_queue_with_only_future_jobs_reports_no_wait(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        session.add(Job(ticket_id=t.id, status=JobStatus.QUEUED, available_at=NOW + HOUR))
        session.commit()
    assert collect(seeded_factory, now=NOW)["queue"]["oldest_queued_age_s"] == 0.0


def test_tickets_by_status_and_watched_audit_events_in_the_window(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        new_ticket(session, TicketStatus.DRAFTED)
        new_ticket(session, TicketStatus.ESCALATED)
        new_ticket(session, TicketStatus.ESCALATED)
        for action, ago in [
            ("cross_customer_access", timedelta(minutes=1)),
            ("cross_customer_access", timedelta(minutes=2)),
            ("job.dead", timedelta(minutes=3)),
            ("cross_customer_access", timedelta(hours=3)),  # too old
            ("tool.call", timedelta(minutes=1)),  # not watched
        ]:
            session.add(
                AuditLog(actor_type=ActorType.SYSTEM, action=action, ts=NOW - ago, details=None)
            )
        session.commit()
    m = collect(seeded_factory, window_s=3600, now=NOW)
    assert m["tickets"] == {"drafted": 1, "escalated": 2}
    assert m["audit"] == {"cross_customer_access": 2, "job.dead": 1}


def test_the_state_of_the_controls_is_included(seeded_factory: sessionmaker[Session]) -> None:
    controls = Controls(seeded_factory)
    controls.set("writes", "frozen", "maya", "incident")
    assert collect(seeded_factory, controls=controls, now=NOW)["controls"] == {
        "agent": "on",
        "writes": "frozen",
    }


# ------------------------------------------------------------------------------ Prometheus


def sample(seeded_factory: sessionmaker[Session]) -> dict[str, Any]:
    with seeded_factory() as session:
        t = new_ticket(session)
        add_run(session, t, steps=[{"model": 'we"ird\\model'}])
        add_run(session, t, outcome="escalated")
        session.commit()
    return collect(seeded_factory, controls=Controls(seeded_factory), now=NOW)


def test_prometheus_text_has_help_once_per_metric_and_skips_missing_values(
    seeded_factory: sessionmaker[Session],
) -> None:
    text = to_prometheus(sample(seeded_factory))
    assert text.endswith("\n")
    assert text.count("# TYPE kartly_runs_by_outcome gauge") == 1
    assert 'kartly_runs_by_outcome{outcome="drafted"} 1' in text
    assert 'kartly_runs_by_outcome{outcome="escalated"} 1' in text
    assert "kartly_runs 2" in text and "kartly_run_escalation_rate 0.5" in text
    assert 'kartly_control{name="writes",value="on"} 1' in text
    assert "kartly_jobs_oldest_queued_age_seconds" not in text  # no value: no line
    assert "kartly_approvals_oldest_pending_age_seconds" not in text


def test_prometheus_labels_cannot_break_the_format(
    seeded_factory: sessionmaker[Session],
) -> None:
    text = to_prometheus(sample(seeded_factory))
    assert 'kartly_llm_model_calls{model="we_ird_model"} 1' in text
    for line in text.splitlines():
        if not line.startswith("#"):
            assert line.count('"') % 2 == 0 and "\\" not in line


def test_old_model_and_tool_steps_are_outside_the_window(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as session:
        t = new_ticket(session)
        add_run(
            session,
            t,
            ago=timedelta(hours=3),
            steps=[
                {"model": "old"},
                {"kind": StepKind.TOOL_CALL, "name": "old_tool", "model": None},
            ],
        )
        session.commit()
    m = collect(seeded_factory, window_s=3600, now=NOW)
    assert m["llm"]["calls"] == 0 and m["llm"]["by_model"] == {} and m["tools"] == {}
    assert collect(seeded_factory, window_s=5 * 3600, now=NOW)["llm"]["calls"] == 1


# ------------------------------------------------------------------------------ the API


def app(factory: sessionmaker[Session]) -> TestClient:
    runner = TicketRunner(factory, ScriptedClient([]), make_gateway(factory), model_tier="s")
    state = ApiState(
        session_factory=factory,
        runner=runner,
        executor=Inline(),
        api_key=KEY,
        controls=Controls(factory),
    )
    return TestClient(create_app(state))


def test_metrics_endpoints_need_the_key(seeded_factory: sessionmaker[Session]) -> None:
    http = app(seeded_factory)
    assert http.get("/v1/metrics").status_code == 401
    assert http.get("/metrics").status_code == 401


def test_json_metrics_and_window_validation(seeded_factory: sessionmaker[Session]) -> None:
    http = app(seeded_factory)
    ok = http.get("/v1/metrics?window_s=600", headers=AUTH)
    assert ok.status_code == 200
    body = ok.json()
    assert body["window"]["seconds"] == 600 and body["runs"]["total"] == 0
    assert body["controls"] == {"agent": "on", "writes": "on"}
    for bad in ("0", "-5", "abc", str(91 * 86400)):
        assert http.get(f"/v1/metrics?window_s={bad}", headers=AUTH).status_code == 422


def test_prometheus_endpoint_serves_text(seeded_factory: sessionmaker[Session]) -> None:
    http = app(seeded_factory)
    r = http.get("/metrics", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert "# TYPE kartly_runs gauge" in r.text and "kartly_runs 0" in r.text


def test_every_response_has_a_request_id_and_one_log_line(
    seeded_factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    http = app(seeded_factory)
    with caplog.at_level(logging.INFO, logger="kartly.access"):
        mine = http.get("/healthz", headers={"X-Request-ID": "abc-123.x"})
        generated = http.get("/healthz")
        hostile = http.get("/healthz", headers={"X-Request-ID": "a b\tc;" * 30})
    assert mine.headers["x-request-id"] == "abc-123.x"
    assert len(generated.headers["x-request-id"]) == 32
    assert hostile.headers["x-request-id"] != "a b\tc;" * 30
    lines = [r.getMessage() for r in caplog.records if r.name == "kartly.access"]
    assert len(lines) == 3
    assert lines[0].startswith("GET /healthz -> 200 in ") and "request_id=abc-123.x" in lines[0]


def test_the_log_line_leaves_out_the_query_string(
    seeded_factory: sessionmaker[Session], caplog: pytest.LogCaptureFixture
) -> None:
    http = app(seeded_factory)
    with caplog.at_level(logging.INFO, logger="kartly.access"):
        http.get("/v1/metrics?window_s=60", headers=AUTH)
    [line] = [r.getMessage() for r in caplog.records if r.name == "kartly.access"]
    assert "window_s" not in line and "/v1/metrics -> 200" in line
