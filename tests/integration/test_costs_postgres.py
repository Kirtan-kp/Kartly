"""The cost report's grouping, JSON reads and decimal sums on a real Postgres."""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy.orm import Session, sessionmaker

from kartly.costs import build_report
from kartly.store.models import Run, RunMode, RunStatus, RunStep, StepKind, Ticket
from tests.support import FIXED_NOW

pytestmark = pytest.mark.postgres


def test_the_report_groups_by_model_and_reads_the_tier_from_json(
    pg_factory: sessionmaker[Session],
) -> None:
    with pg_factory() as session:
        for tier, outcome in ((1, "drafted"), (1, "escalated"), (2, "drafted")):
            ticket = Ticket(customer_id="C-0001")
            session.add(ticket)
            session.flush()
            run = Run(
                ticket_id=ticket.id,
                mode=RunMode.SHADOW,
                status=RunStatus.SUCCEEDED,
                outcome=outcome,
                shadow_cost_usd=Decimal("0.0012345678"),
                started_at=FIXED_NOW,
            )
            session.add(run)
            session.flush()
            session.add(
                RunStep(
                    run_id=run.id, idx=0, kind=StepKind.STAGE, name="classify", model="small",
                    tokens_in=300, tokens_out=40, shadow_cost_usd=Decimal("0.0000500000"),
                    output={"tier": tier, "intent": "refund"}, created_at=FIXED_NOW,
                )
            )  # fmt: skip
            session.add(
                RunStep(
                    run_id=run.id, idx=1, kind=StepKind.LLM_CALL, name="chat", model="large",
                    tokens_in=2000, tokens_out=300, shadow_cost_usd=Decimal("0.0011845678"),
                    created_at=FIXED_NOW,
                )
            )  # fmt: skip
        session.commit()
    report = build_report(pg_factory, None, window_s=3600, now=FIXED_NOW)
    assert [m["model"] for m in report["by_model"]] == ["large", "small"]
    assert report["by_model"][0]["cost_usd"] == "0.003554"
    assert report["total"]["tokens_in"] == 6900 and report["total"]["runs"] == 3
    tiers = {g["tier"]: g for g in report["by_tier"]}
    assert (tiers["tier 1"]["runs"], tiers["tier 1"]["escalated"]) == (2, 1)
    assert tiers["tier 2"]["runs"] == 1 and tiers["tier 1"]["answered"] == 1
