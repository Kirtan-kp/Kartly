"""Read a stored run back as one JSON-ready document, and print it for people.

``load_trace`` is what ``GET /v1/runs/{id}/trace`` returns (next step). It joins the run, its
ordered steps and the gateway's audit rows for that run, so a reviewer sees what the model asked
for next to what the gateway actually did.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from kartly.store.models import AuditLog, Run, RunStep, Ticket


def _money(value: Decimal) -> str:
    return f"{value:.10f}"


def load_trace(session: Session, run_id: uuid.UUID) -> dict[str, Any] | None:
    run = session.get(Run, run_id)
    if run is None:
        return None
    ticket = session.get(Ticket, run.ticket_id)
    steps = session.scalars(select(RunStep).where(RunStep.run_id == run_id).order_by(RunStep.idx))
    audit = session.scalars(select(AuditLog).where(AuditLog.run_id == run_id).order_by(AuditLog.id))
    return {
        "run": {
            "id": str(run.id),
            "ticket_id": str(run.ticket_id),
            "status": run.status.value,
            "outcome": run.outcome,
            "mode": run.mode.value,
            "model_tier": run.model_tier,
            "prompt_version": run.prompt_version,
            "tokens_in": run.tokens_in,
            "tokens_out": run.tokens_out,
            "shadow_cost_usd": _money(Decimal(run.shadow_cost_usd)),
            "error": run.error,
            "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        },
        "ticket": (
            {"id": str(ticket.id), "customer_id": ticket.customer_id, "status": ticket.status.value}
            if ticket
            else None
        ),
        "steps": [
            {
                "idx": s.idx,
                "kind": s.kind.value,
                "name": s.name,
                "model": s.model,
                "input": s.input,
                "output": s.output,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "latency_ms": s.latency_ms,
                "shadow_cost_usd": _money(Decimal(s.shadow_cost_usd)),
                "error": s.error,
            }
            for s in steps
        ],
        "audit": [
            {"id": a.id, "ts": a.ts.isoformat(), "action": a.action, "details": a.details}
            for a in audit
        ],
    }


def _describe_step(step: dict[str, Any]) -> str:
    output = step["output"] or {}
    if step["kind"] == "llm_call":
        if step["error"]:
            return f"FAILED {step['error']}"
        calls = output.get("tool_calls") or []
        if calls:
            return "asks for " + ", ".join(f"{c['name']}({c['arguments']})" for c in calls)
        return f"replies: {output.get('content')!r}"
    if step["kind"] == "tool_call":
        args = (step["input"] or {}).get("arguments")
        status = "ok" if output.get("ok") else f"error {output.get('error_code') or step['error']}"
        return f"{step['name']}({args}) -> {status}"
    return str(step["name"])


def format_trace(trace: dict[str, Any]) -> str:
    run = trace["run"]
    lines = [
        f"Run {run['id']}: {run['status']}/{run['outcome']}  "
        f"tokens {run['tokens_in']} in + {run['tokens_out']} out  "
        f"shadow cost ${Decimal(run['shadow_cost_usd']):.6f}",
    ]
    if run["error"]:
        lines.append(f"  stopped because: {run['error']}")
    for step in trace["steps"]:
        timing = f"{(step['latency_ms'] or 0) / 1000:.1f}s"
        who = step["model"] or step["name"] if step["kind"] == "llm_call" else "tool"
        lines.append(
            f"  {step['idx']:>2} {step['kind']:<9} {who:<12} {timing:>6}  {_describe_step(step)}"
        )
    ticket = trace["ticket"]
    if ticket:
        lines.append(f"Ticket {ticket['id']} is now {ticket['status']}")
    return "\n".join(lines)
