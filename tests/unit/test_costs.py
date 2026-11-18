"""The shadow cost report: where the money went and what other choices would have cost."""

from __future__ import annotations

import json
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from kartly.costs import UNCLASSIFIED, build_report, main, price_tokens, render_text
from kartly.llm.config import ModelsFile
from kartly.store.models import Run, RunMode, RunStatus, RunStep, StepKind, Ticket
from tests.support import FIXED_NOW
from tests.unit.test_api_admin import build  # reuses the API wiring helper

NOW = FIXED_NOW
HOUR = timedelta(hours=1)
DAY = 86400.0

MODELS = ModelsFile.model_validate(
    {
        "providers": {"p": {"base_url": "http://x/v1", "api_key_env": None}},
        "models": {
            "small": {
                "provider": "p",
                "model": "s",
                "price": {"input_per_mtok": 0.1, "output_per_mtok": 0.4},
            },
            "large": {
                "provider": "p",
                "model": "l",
                "price": {"input_per_mtok": 1, "output_per_mtok": 4},
            },
            "huge": {
                "provider": "p",
                "model": "h",
                "price": {"input_per_mtok": 10, "output_per_mtok": 40},
            },
        },
        "roles": {"small": ["small"], "large": ["large"]},
    }
)


def step(
    model: str | None = "small",
    tin: int = 1000,
    tout: int = 200,
    cost: str = "0.0001",
    kind: StepKind = StepKind.LLM_CALL,
    name: str = "chat",
    output: dict[str, Any] | None = None,
    ago: timedelta = timedelta(minutes=5),
) -> dict[str, Any]:
    return {
        "model": model, "tin": tin, "tout": tout, "cost": cost, "kind": kind, "name": name,
        "output": output, "ago": ago,
    }  # fmt: skip


def classify(tier: Any, model: str = "small", **kw: Any) -> dict[str, Any]:
    return step(
        model, kind=StepKind.STAGE, name="classify", output={"tier": tier, "intent": "refund"}, **kw
    )


def add_run(
    session: Session,
    *steps: dict[str, Any],
    outcome: str | None = "drafted",
    cost: str = "0.001",
    ago: timedelta = timedelta(minutes=5),
) -> Run:
    ticket = Ticket(customer_id="C-0002")
    session.add(ticket)
    session.flush()
    run = Run(
        ticket_id=ticket.id,
        mode=RunMode.SHADOW,
        status=RunStatus.SUCCEEDED,
        outcome=outcome,
        shadow_cost_usd=Decimal(cost),
        started_at=NOW - ago,
    )
    session.add(run)
    session.flush()
    for i, s in enumerate(steps):
        session.add(
            RunStep(
                run_id=run.id, idx=i, kind=s["kind"], name=s["name"], model=s["model"],
                tokens_in=s["tin"], tokens_out=s["tout"], shadow_cost_usd=Decimal(s["cost"]),
                output=s["output"], created_at=NOW - s["ago"],
            )
        )  # fmt: skip
    session.flush()
    return run


def report(factory: sessionmaker[Session], **kw: Any) -> dict[str, Any]:
    kw.setdefault("window_s", DAY)
    return build_report(factory, kw.pop("models", MODELS), now=NOW, **kw)


def test_price_tokens_is_exact() -> None:
    assert price_tokens(1_000_000, 0, Decimal("0.15"), Decimal("0.6")) == Decimal("0.15")
    assert price_tokens(500_000, 250_000, Decimal("2"), Decimal("8")) == Decimal("3")
    assert price_tokens(0, 0, Decimal("5"), Decimal("5")) == 0


def test_an_empty_window_reports_zeros_and_no_ratios(seeded_factory: sessionmaker[Session]) -> None:
    r = report(seeded_factory)
    assert r["total"] == {
        "runs": 0, "calls": 0, "tokens_in": 0, "tokens_out": 0,
        "cost_usd": "0.000000", "per_1000_runs_usd": None,
    }  # fmt: skip
    assert r["per_run"]["avg_usd"] is None and r["per_run"]["sampled_runs"] == 0
    assert r["by_model"] == [] and r["by_tier"] == []
    assert r["what_if"]["routing_saving_vs_baseline"] is None
    assert all(s["vs_actual"] is None for s in r["what_if"]["all_traffic_on"])
    assert "no model calls" in render_text(r)


def test_model_totals_count_llm_calls_and_the_classifier_but_nothing_else(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(
            s,
            classify(1, "small", tin=300, tout=50, cost="0.00005"),
            step("small", 1000, 200, "0.0002"),
            step("large", 4000, 800, "0.0070"),
            step("small", 500, 100, "0.0001"),
            step(None, 999, 999, "9"),  # no model
            step("small", 999, 999, "9", kind=StepKind.TOOL_CALL, name="lookup_order"),
            step("small", 999, 999, "9", kind=StepKind.STAGE, name="screen"),
            step("small", 999, 999, "9", ago=timedelta(days=2)),  # outside the window
        )
        s.commit()
    r = report(seeded_factory)
    assert r["by_model"] == [
        {
            "model": "large",
            "calls": 1,
            "tokens_in": 4000,
            "tokens_out": 800,
            "cost_usd": "0.007000",
        },
        {
            "model": "small",
            "calls": 3,
            "tokens_in": 1800,
            "tokens_out": 350,
            "cost_usd": "0.000350",
        },
    ]
    assert r["total"]["tokens_in"] == 5800 and r["total"]["tokens_out"] == 1150
    assert r["total"]["calls"] == 4 and r["total"]["cost_usd"] == "0.007350"


def test_the_window_includes_its_start_and_excludes_anything_older(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(s, step(tin=10, tout=1, ago=timedelta(hours=24)), ago=timedelta(hours=24))
        add_run(
            s,
            step(tin=77, tout=7, ago=timedelta(hours=24, seconds=1)),
            ago=timedelta(hours=24, seconds=1),
        )
        s.commit()
    r = report(seeded_factory)
    assert r["total"]["runs"] == 1 and r["total"]["tokens_in"] == 10


def test_per_run_figures_and_the_projection(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as s:
        for cost in ("0.001", "0.002", "0.003", "0.004"):
            add_run(s, step(cost=cost), cost=cost)
        s.commit()
    r = report(seeded_factory)
    assert r["per_run"] == {
        "avg_usd": "0.002500", "p50_usd": "0.002000", "p95_usd": "0.004000",
        "max_usd": "0.004000", "sampled_runs": 4,
    }  # fmt: skip
    assert r["total"]["runs"] == 4 and r["total"]["cost_usd"] == "0.010000"
    assert r["total"]["per_1000_runs_usd"] == "2.500000"  # 0.01 / 4 runs * 1000


def test_tiers_count_escalations_and_charge_them_to_the_answered_tickets(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(s, classify(1), outcome="drafted", cost="0.001")
        add_run(s, classify(1), outcome="drafted", cost="0.003")
        add_run(s, classify(2), outcome="escalated", cost="0.006")
        add_run(s, classify(2), outcome="awaiting_approval", cost="0.002")
        add_run(s, classify(2), outcome="failed", cost="0.001")
        add_run(s, step(), outcome="escalated", cost="0.0005")  # closed by rules, never classified
        add_run(s, classify("high"), outcome="drafted", cost="0.0004")  # a tier we cannot read
        add_run(s, classify(1), outcome=None, cost="0.0001")  # still running
        s.commit()
    tiers = {g["tier"]: g for g in report(seeded_factory)["by_tier"]}
    assert set(tiers) == {"tier 1", "tier 2", UNCLASSIFIED}
    one, two, none = tiers["tier 1"], tiers["tier 2"], tiers[UNCLASSIFIED]
    assert (one["runs"], one["answered"], one["escalated"]) == (
        3,
        2,
        0,
    )  # the running one is not answered
    assert one["cost_usd"] == "0.004100" and one["cost_per_answered_usd"] == "0.002050"
    assert (two["runs"], two["answered"], two["escalated"], two["failed"]) == (3, 1, 1, 1)
    assert two["cost_per_answered_usd"] == "0.009000"  # all 0.009 of spend carried by one answer
    assert (none["runs"], none["answered"]) == (2, 1)
    assert none["avg_cost_usd"] == "0.000450"


def test_a_tier_with_no_answers_has_no_cost_per_answer(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(s, classify(3), outcome="escalated", cost="0.001")
        s.commit()
    [g] = report(seeded_factory)["by_tier"]
    assert (
        g["answered"] == 0
        and g["cost_per_answered_usd"] is None
        and g["avg_cost_usd"] == "0.001000"
    )


def test_what_if_prices_the_same_tokens_on_every_model_cheapest_first(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(s, step("small", 1_000_000, 500_000, "0.30"))  # actual: 0.30
        s.commit()
    r = report(seeded_factory)
    wi = r["what_if"]
    assert [x["model"] for x in wi["all_traffic_on"]] == ["small", "large", "huge"]
    assert [x["cost_usd"] for x in wi["all_traffic_on"]] == ["0.300000", "3.000000", "30.000000"]
    assert [x["vs_actual"] for x in wi["all_traffic_on"]] == [1.0, 10.0, 100.0]
    assert wi["baseline_model"] == "large"
    assert wi["routing_saving_vs_baseline"] == 0.9  # 1 - 0.30 / 3.00


def test_the_saving_can_be_negative_and_is_absent_without_a_large_role(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(s, step("huge", 1_000_000, 0, "10"))
        s.commit()
    assert report(seeded_factory)["what_if"]["routing_saving_vs_baseline"] == -9.0
    no_large = MODELS.model_copy(update={"roles": {"small": ["small"]}})
    wi = report(seeded_factory, models=no_large)["what_if"]
    assert wi["baseline_model"] is None and wi["routing_saving_vs_baseline"] is None


def test_without_a_price_list_there_is_no_what_if(seeded_factory: sessionmaker[Session]) -> None:
    r = report(seeded_factory, models=None)
    assert r["what_if"] is None and "What the same tokens" not in render_text(r)


def test_the_text_report_lists_every_section_and_the_caveats(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        add_run(s, classify(1), step("large", 2000, 400, "0.0004"))
        s.commit()
    text = render_text(report(seeded_factory))
    expected_lines = (
        "last 1 day(s)", "By model", "By routing tier", "tier 1", "large", "huge",
        "routing saves", "note: Costs use reference prices", "note: What-if",
    )  # fmt: skip
    for expected in expected_lines:
        assert expected in text, expected


# ------------------------------------------------------------------------------ API and CLI


@pytest.fixture
def http(seeded_factory: sessionmaker[Session]) -> TestClient:
    web = build(seeded_factory)
    web.app.state.kartly.models = MODELS  # type: ignore[attr-defined]
    return web


def test_the_costs_route_needs_credentials(http: TestClient) -> None:
    assert http.get("/v1/costs").status_code == 401
    assert http.get("/v1/costs", headers={"X-API-Key": "test-api-key-123456"}).status_code == 200


def test_the_costs_route_returns_the_report_and_validates_the_window(
    http: TestClient, seeded_factory: sessionmaker[Session]
) -> None:
    with seeded_factory() as s:
        add_run(s, step("small", 1000, 100, "0.0001"), ago=timedelta(minutes=1))
        s.commit()
    headers = {"X-API-Key": "test-api-key-123456"}
    got = http.get("/v1/costs?window_s=86400", headers=headers).json()
    assert set(got) == {"window", "total", "per_run", "by_model", "by_tier", "what_if", "notes"}
    assert got["window"]["seconds"] == 86400
    for bad in ("0", "-5", str(366 * 86400), "abc"):
        assert http.get(f"/v1/costs?window_s={bad}", headers=headers).status_code == 422, bad


def test_the_costs_route_works_for_a_signed_in_reviewer_and_without_prices(
    seeded_factory: sessionmaker[Session],
) -> None:
    web = build(seeded_factory)  # no price list configured
    web.app.state.kartly.reviewers.create("omar", "correct horse battery")  # type: ignore[attr-defined]
    web.post("/v1/admin/login", json={"username": "omar", "password": "correct horse battery"})
    got = web.get("/v1/costs")
    assert got.status_code == 200 and got.json()["what_if"] is None


@pytest.fixture
def cli(seeded_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kartly.store.db.make_engine", lambda *a, **k: seeded_factory.kw["bind"])
    monkeypatch.setattr("sqlalchemy.engine.Engine.dispose", lambda self, *a, **k: None)
    monkeypatch.setattr("kartly.config.load_env", lambda: None)


def test_the_command_line_prints_text_or_json(
    cli: None, capsys: pytest.CaptureFixture[str], seeded_factory: sessionmaker[Session]
) -> None:
    assert main(["--days", "30"]) == 0
    out = capsys.readouterr().out
    assert "Shadow cost report, last 30 day(s)" in out and "What the same tokens" in out
    assert main(["--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["window"]["seconds"] == 7 * 86400


@pytest.mark.parametrize("days", ["0", "-1", "366"])
def test_the_command_line_rejects_a_bad_window(
    cli: None, capsys: pytest.CaptureFixture[str], days: str
) -> None:
    assert main(["--days", days]) == 2
    assert "--days" in capsys.readouterr().out


def test_a_missing_price_file_only_drops_the_what_if_section(
    cli: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from kartly.config import Settings

    monkeypatch.setattr(
        "kartly.config.get_settings", lambda: Settings(models_config_path="/nonexistent.yaml")
    )
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "warning: no what-if section" in out and "What the same tokens" not in out


def test_percentiles_use_the_nearest_rank_over_many_runs(
    seeded_factory: sessionmaker[Session],
) -> None:
    with seeded_factory() as s:
        for n in range(1, 21):
            add_run(s, step(cost="0.000001"), cost=f"0.{n:03d}")
        s.commit()
    per_run = report(seeded_factory)["per_run"]
    assert per_run["p50_usd"] == "0.010000"  # the 10th of 20
    assert per_run["p95_usd"] == "0.019000"  # the 19th of 20
    assert per_run["max_usd"] == "0.020000" and per_run["sampled_runs"] == 20


def test_ratios_are_rounded_to_four_places(seeded_factory: sessionmaker[Session]) -> None:
    with seeded_factory() as s:
        add_run(s, step("small", 1_000_000, 0, "0.30"))  # on large the same tokens cost 1.00
        s.commit()
    by_model = {x["model"]: x for x in report(seeded_factory)["what_if"]["all_traffic_on"]}
    assert by_model["large"]["vs_actual"] == 3.3333
