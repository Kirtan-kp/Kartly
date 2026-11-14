"""Operational numbers, computed from the database (ADR-054).

Counters kept in memory would be wrong the moment a second process (a worker, a second API
instance) existed, and would reset on every restart. Everything here is a query over what the
system already stores: runs, steps, actions, approvals, jobs, audit rows and the controls. So the
API, a worker and a script all report the same figures, and history survives a restart.

``collect`` returns plain data; ``to_prometheus`` renders the same data as Prometheus text.
Latency is the time spent in model and tool calls (sum of step latencies per run), because a run's
wall-clock start and end are stored to the second only. Percentiles use the most recent
``MAX_ROWS`` runs in the window; counts are exact.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kartly.controls import Controls
from kartly.store.models import (
    Action,
    Approval,
    ApprovalStatus,
    AuditLog,
    Job,
    JobStatus,
    Run,
    RunStep,
    StepKind,
    Ticket,
)
from kartly.tools.orders import utc

MAX_ROWS = 5000
# Audit events an operator wants to see counted. Anything else is not listed here on purpose.
WATCHED_AUDIT = (
    "cross_customer_access",
    "approval.blocked_writes_frozen",
    "kill_switch.held",
    "control.set",
    "job.dead",
    "job.requeued",
)


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for no data."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def _counts(session: Session, column: Any, *where: Any) -> dict[str, int]:
    rows = session.execute(select(column, func.count()).where(*where).group_by(column)).all()
    return {str(getattr(k, "value", k) if k is not None else "none"): int(n) for k, n in rows}


def _age_s(moment: datetime | None, now: datetime) -> float | None:
    return round((now - utc(moment)).total_seconds(), 1) if moment is not None else None


def collect(
    factory: Callable[[], Session],
    *,
    window_s: float = 86400.0,
    controls: Controls | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    now = now or datetime.now(UTC)
    since = now - timedelta(seconds=window_s)
    control_values = {c.name: c.value for c in controls.all()} if controls else {}  # own session
    with factory() as session:
        return {
            "window": {"since": since.isoformat(), "until": now.isoformat(), "seconds": window_s},
            "runs": _runs(session, since),
            "llm": _llm(session, since),
            "tools": _tools(session, since),
            "actions": _actions(session, since),
            "approvals": _approvals(session, since, now),
            "queue": _queue(session, now),
            "tickets": _counts(session, Ticket.status),
            "audit": _counts(
                session, AuditLog.action, AuditLog.ts >= since, AuditLog.action.in_(WATCHED_AUDIT)
            ),
            "controls": control_values,
        }


def _runs(session: Session, since: datetime) -> dict[str, Any]:
    in_window = Run.started_at >= since
    total = session.scalar(select(func.count()).select_from(Run).where(in_window)) or 0
    by_outcome = _counts(session, Run.outcome, in_window)
    by_status = _counts(session, Run.status, in_window)
    sums = session.execute(
        select(
            func.coalesce(func.sum(Run.tokens_in), 0),
            func.coalesce(func.sum(Run.tokens_out), 0),
            func.coalesce(func.sum(Run.shadow_cost_usd), 0),
        ).where(in_window)
    ).one()
    recent = [
        r.id
        for r in session.execute(
            select(Run.id).where(in_window).order_by(Run.started_at.desc()).limit(MAX_ROWS)
        )
    ]
    latency_by_run: dict[Any, int] = {}
    if recent:
        for run_id, ms in session.execute(
            select(RunStep.run_id, func.coalesce(func.sum(RunStep.latency_ms), 0))
            .where(RunStep.run_id.in_(recent))
            .group_by(RunStep.run_id)
        ):
            latency_by_run[run_id] = int(ms or 0)
    latencies = [float(v) for v in latency_by_run.values()]
    escalated = by_outcome.get("escalated", 0)
    return {
        "total": int(total),
        "by_outcome": by_outcome,
        "by_status": by_status,
        "escalation_rate": round(escalated / total, 4) if total else None,
        "failure_rate": round(by_status.get("failed", 0) / total, 4) if total else None,
        "tokens_in": int(sums[0]),
        "tokens_out": int(sums[1]),
        "shadow_cost_usd": f"{Decimal(sums[2] or 0):.6f}",
        "avg_shadow_cost_usd": f"{Decimal(sums[2] or 0) / total:.6f}" if total else None,
        "latency_ms": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "sampled_runs": len(latencies),
        },
    }


def _llm(session: Session, since: datetime) -> dict[str, Any]:
    window = (RunStep.kind == StepKind.LLM_CALL, RunStep.created_at >= since)
    rows = session.execute(
        select(
            RunStep.model,
            func.count(),
            func.count(RunStep.error),
            func.coalesce(func.sum(RunStep.tokens_in), 0),
            func.coalesce(func.sum(RunStep.tokens_out), 0),
            func.coalesce(func.sum(RunStep.shadow_cost_usd), 0),
        )
        .where(*window)
        .group_by(RunStep.model)
    ).all()
    by_model: dict[str, dict[str, Any]] = {
        str(model or "unknown"): {
            "calls": int(calls),
            "errors": int(errors),
            "tokens_in": int(t_in),
            "tokens_out": int(t_out),
            "shadow_cost_usd": f"{Decimal(cost or 0):.6f}",
        }
        for model, calls, errors, t_in, t_out, cost in rows
    }
    calls = sum(m["calls"] for m in by_model.values())
    errors = sum(m["errors"] for m in by_model.values())
    # Replies that needed a retry or came from a fallback model carry these in their output.
    retried = failed_over = 0
    for (output,) in session.execute(
        select(RunStep.output).where(*window, RunStep.error.is_(None)).limit(MAX_ROWS)
    ):
        if isinstance(output, dict):
            retried += 1 if output.get("attempts", 1) > 1 else 0
            failed_over += 1 if output.get("failed_over_from") else 0
    return {
        "calls": calls,
        "errors": errors,
        "error_rate": round(errors / calls, 4) if calls else None,
        "retried_replies": retried,
        "fallback_replies": failed_over,
        "by_model": by_model,
    }


def _tools(session: Session, since: datetime) -> dict[str, Any]:
    window = (RunStep.kind == StepKind.TOOL_CALL, RunStep.created_at >= since)
    rows = session.execute(
        select(RunStep.name, func.count(), func.count(RunStep.error))
        .where(*window)
        .group_by(RunStep.name)
    ).all()
    return {
        name: {"calls": int(n), "errors": int(e), "error_rate": round(e / n, 4) if n else None}
        for name, n, e in rows
    }


def _actions(session: Session, since: datetime) -> dict[str, Any]:
    rows = session.execute(
        select(Action.tool_name, Action.status, func.count())
        .where(Action.created_at >= since)
        .group_by(Action.tool_name, Action.status)
    ).all()
    out: dict[str, dict[str, int]] = {}
    for tool, status, n in rows:
        out.setdefault(tool, {})[status.value] = int(n)
    return out


def _approvals(session: Session, since: datetime, now: datetime) -> dict[str, Any]:
    pending = session.scalar(
        select(func.count()).select_from(Approval).where(Approval.status == ApprovalStatus.PENDING)
    )
    oldest = session.scalar(
        select(func.min(Approval.requested_at)).where(Approval.status == ApprovalStatus.PENDING)
    )
    return {
        "pending": int(pending or 0),
        "oldest_pending_age_s": _age_s(oldest, now),
        "decided": _counts(
            session,
            Approval.status,
            Approval.status != ApprovalStatus.PENDING,
            Approval.requested_at >= since,
        ),
    }


def _queue(session: Session, now: datetime) -> dict[str, Any]:
    by_status = _counts(session, Job.status)
    oldest = session.scalar(
        select(func.min(Job.available_at)).where(Job.status == JobStatus.QUEUED)
    )
    return {
        "queued": by_status.get("queued", 0),
        "running": by_status.get("running", 0),
        "dead": by_status.get("dead", 0),
        "succeeded": by_status.get("succeeded", 0),
        "oldest_queued_age_s": max(0.0, _age_s(oldest, now) or 0.0) if oldest is not None else None,
    }


# ------------------------------------------------------------------------------ Prometheus

_LABEL = re.compile(r"[^A-Za-z0-9_:.\-/ ]")


def _esc(value: str) -> str:
    """Anything that could end a label value early (a quote, a backslash, a newline) is replaced."""
    return _LABEL.sub("_", value)


def to_prometheus(m: Mapping[str, Any]) -> str:
    """The same numbers as Prometheus text. Gauges only: they are recomputed on every scrape."""
    lines: list[str] = []
    declared: set[str] = set()

    def gauge(name: str, value: float | int | None, help_: str, /, **labels: str) -> None:
        if value is None:
            return
        if name not in declared:
            declared.add(name)
            lines.append(f"# HELP kartly_{name} {help_}")
            lines.append(f"# TYPE kartly_{name} gauge")
        label = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        lines.append(f"kartly_{name}{{{label}}} {value}" if label else f"kartly_{name} {value}")

    w = m["window"]["seconds"]
    runs, llm = m["runs"], m["llm"]
    gauge("window_seconds", w, "Length of the window the figures cover")
    gauge("runs", runs["total"], "Agent runs in the window")
    for outcome, n in runs["by_outcome"].items():
        gauge("runs_by_outcome", n, "Runs in the window by outcome", outcome=outcome)
    gauge("run_escalation_rate", runs["escalation_rate"], "Share of runs that ended escalated")
    gauge("run_failure_rate", runs["failure_rate"], "Share of runs that failed")
    gauge("tokens_in", runs["tokens_in"], "Prompt tokens in the window")
    gauge("tokens_out", runs["tokens_out"], "Completion tokens in the window")
    gauge("shadow_cost_usd", float(runs["shadow_cost_usd"]), "Cost at reference prices")
    gauge("run_latency_ms_p50", runs["latency_ms"]["p50"], "Time in model and tool calls, median")
    gauge("run_latency_ms_p95", runs["latency_ms"]["p95"], "Time in model and tool calls, p95")
    gauge("llm_calls", llm["calls"], "Model calls in the window")
    gauge("llm_errors", llm["errors"], "Model calls that failed")
    gauge("llm_error_rate", llm["error_rate"], "Share of model calls that failed")
    gauge("llm_retried_replies", llm["retried_replies"], "Replies that needed a retry")
    gauge("llm_fallback_replies", llm["fallback_replies"], "Replies from a fallback model")
    for model, d in llm["by_model"].items():
        gauge("llm_model_calls", d["calls"], "Model calls by model", model=model)
        gauge("llm_model_errors", d["errors"], "Failed model calls by model", model=model)
    for tool, d in m["tools"].items():
        gauge("tool_calls", d["calls"], "Tool calls by tool", tool=tool)
        gauge("tool_errors", d["errors"], "Failed tool calls by tool", tool=tool)
    for tool, statuses in m["actions"].items():
        for status, n in statuses.items():
            gauge("actions", n, "Write actions by tool and status", tool=tool, status=status)
    a = m["approvals"]
    gauge("approvals_pending", a["pending"], "Approvals waiting for a person")
    gauge("approvals_oldest_pending_age_seconds", a["oldest_pending_age_s"], "Age of the oldest")
    q = m["queue"]
    for state in ("queued", "running", "dead"):
        gauge("jobs", q[state], "Jobs by state", state=state)
    gauge(
        "jobs_oldest_queued_age_seconds", q["oldest_queued_age_s"], "Age of the oldest waiting job"
    )
    for status, n in m["tickets"].items():
        gauge("tickets", n, "Tickets by status", status=status)
    for event, n in m["audit"].items():
        gauge("audit_events", n, "Watched audit events in the window", event=event)
    for name, value in m["controls"].items():
        gauge("control", 1, "Current value of a control", name=name, value=value)
    return "\n".join(lines) + "\n"
