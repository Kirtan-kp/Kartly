"""The metrics queries on a real Postgres (enum grouping, sums and JSON output columns)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker

from kartly.metrics import collect, to_prometheus
from kartly.store.models import (
    Job,
    JobStatus,
    Run,
    RunMode,
    RunStatus,
    RunStep,
    StepKind,
    Ticket,
)

pytestmark = pytest.mark.postgres


def test_metrics_run_on_postgres(pg_factory: sessionmaker[Session]) -> None:
    now = datetime.now(UTC)
    with pg_factory() as session:
        ticket = Ticket(customer_id="C-0001")
        session.add(ticket)
        session.flush()
        run = Run(
            ticket_id=ticket.id,
            mode=RunMode.SHADOW,
            status=RunStatus.SUCCEEDED,
            outcome="drafted",
            tokens_in=50,
            tokens_out=10,
            started_at=now - timedelta(minutes=1),
        )
        session.add(run)
        session.flush()
        session.add_all(
            [
                RunStep(
                    run_id=run.id,
                    idx=0,
                    kind=StepKind.LLM_CALL,
                    name="chat",
                    model="m",
                    latency_ms=120,
                    output={"attempts": 2, "failed_over_from": ["x"]},
                ),
                RunStep(
                    run_id=run.id, idx=1, kind=StepKind.TOOL_CALL, name="get_order", latency_ms=30
                ),
            ]
        )
        session.add(Job(ticket_id=ticket.id, status=JobStatus.QUEUED, available_at=now))
        session.commit()
    m = collect(pg_factory, window_s=3600)
    assert m["runs"]["total"] == 1 and m["runs"]["by_outcome"] == {"drafted": 1}
    assert m["runs"]["latency_ms"]["p50"] == 150.0
    assert m["llm"]["retried_replies"] == 1 and m["llm"]["fallback_replies"] == 1
    assert m["tools"]["get_order"]["calls"] == 1 and m["queue"]["queued"] == 1
    assert m["tickets"] == {"received": 1}
    assert "kartly_runs 1" in to_prometheus(m)
