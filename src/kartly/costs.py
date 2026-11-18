"""Shadow pricing report: what the agent cost, where, and what other choices would have cost.

Real spend is $0 on a free tier or a local model, so every call is priced at the reference prices
in ``config/models.yaml`` when it is made (ADR-013). This module reads those stored figures back
and answers three questions from the database alone (ADR-058):

- **Where did the money go?** Per model (calls, tokens, cost) and per routing tier (runs, how many
  were escalated, cost per ticket that was actually answered).
- **What does one ticket cost?** Average, median, 95th percentile and the dearest run.
- **What if?** The same recorded tokens priced at every other configured model, and the saving the
  routing gives against sending everything to the large model.

The "what if" figures hold token counts fixed. A different model would write different amounts,
would sometimes need more tool calls, and would answer better or worse; none of that is measured
here (the eval suite measures quality). Cached-prompt discounts are not applied because cached
token counts are not stored. So the what-if numbers are a guide to order of magnitude, not a quote.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from kartly.llm.config import ModelsFile
from kartly.llm.pricing import ONE_MILLION
from kartly.metrics import MAX_ROWS, percentile
from kartly.store.models import Run, RunStep, StepKind

NOTES = (
    "Costs use reference prices from config/models.yaml; real spend on a free tier or a local "
    "model is $0.",
    "What-if figures keep the recorded token counts. Another model would use different amounts, "
    "and its answers may be better or worse; quality is measured by the evals, not here.",
    "Cached-prompt discounts are not applied (cached token counts are not stored).",
    "Per-run figures use the latest 5,000 runs in the window; totals are exact.",
)
_NOT_ANSWERED = ("escalated", "failed")
UNCLASSIFIED = "unclassified"  # closed by the rules before the classifier ran, or no classifier


def _money(value: Decimal | float | None) -> str:
    return f"{Decimal(value or 0):.6f}"


def _ratio(num: Decimal, den: Decimal) -> float | None:
    return round(float(num / den), 4) if den else None


def price_tokens(
    tokens_in: int, tokens_out: int, in_per_mtok: Decimal, out_per_mtok: Decimal
) -> Decimal:
    """Cost of a token count at a per-million-token price."""
    return (tokens_in * in_per_mtok + tokens_out * out_per_mtok) / ONE_MILLION


def _by_model(session: Session, since: datetime) -> list[dict[str, Any]]:
    is_model_step = (RunStep.kind == StepKind.LLM_CALL) | (
        (RunStep.kind == StepKind.STAGE) & (RunStep.name == "classify")
    )
    rows = session.execute(
        select(
            RunStep.model,
            func.count(),
            func.coalesce(func.sum(RunStep.tokens_in), 0),
            func.coalesce(func.sum(RunStep.tokens_out), 0),
            func.coalesce(func.sum(RunStep.shadow_cost_usd), 0),
        )
        .where(RunStep.created_at >= since, RunStep.model.is_not(None), is_model_step)
        .group_by(RunStep.model)
        .order_by(func.sum(RunStep.shadow_cost_usd).desc(), RunStep.model)
    ).all()
    return [
        {
            "model": str(model),
            "calls": int(calls),
            "tokens_in": int(tin),
            "tokens_out": int(tout),
            "cost_usd": _money(cost),
        }
        for model, calls, tin, tout, cost in rows
    ]


def _per_run(session: Session, since: datetime) -> tuple[list[Run], int]:
    total = session.scalar(select(func.count()).select_from(Run).where(Run.started_at >= since))
    runs = list(
        session.scalars(
            select(Run)
            .where(Run.started_at >= since)
            .order_by(Run.started_at.desc())
            .limit(MAX_ROWS)
        )
    )
    return runs, int(total or 0)


def _tier_of(session: Session, run_ids: list[Any]) -> dict[Any, str]:
    if not run_ids:
        return {}
    rows = session.execute(
        select(RunStep.run_id, RunStep.output).where(
            RunStep.run_id.in_(run_ids), RunStep.kind == StepKind.STAGE, RunStep.name == "classify"
        )
    ).all()
    found: dict[Any, str] = {}
    for run_id, output in rows:
        tier = output.get("tier") if isinstance(output, dict) else None
        found[run_id] = f"tier {tier}" if isinstance(tier, int) else UNCLASSIFIED
    return found


def _by_tier(session: Session, runs: list[Run]) -> list[dict[str, Any]]:
    tiers = _tier_of(session, [r.id for r in runs])
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(tiers.get(run.id, UNCLASSIFIED), []).append(run)
    out = []
    for name in sorted(groups):
        members = groups[name]
        cost = sum((Decimal(r.shadow_cost_usd) for r in members), Decimal(0))
        answered = [r for r in members if r.outcome is not None and r.outcome not in _NOT_ANSWERED]
        out.append(
            {
                "tier": name,
                "runs": len(members),
                "escalated": sum(1 for r in members if r.outcome == "escalated"),
                "failed": sum(1 for r in members if r.outcome == "failed"),
                "answered": len(answered),
                "cost_usd": _money(cost),
                "avg_cost_usd": _money(cost / len(members)),
                # Spend on tickets nobody answered still has to be paid for by the ones that were.
                "cost_per_answered_usd": _money(cost / len(answered)) if answered else None,
            }
        )
    return out


def _what_if(
    models_file: ModelsFile, tokens_in: int, tokens_out: int, actual: Decimal
) -> dict[str, Any]:
    priced = sorted(
        (
            (
                price_tokens(
                    tokens_in, tokens_out, m.price.input_per_mtok, m.price.output_per_mtok
                ),
                name,
            )
            for name, m in models_file.models.items()
        ),
    )
    large = models_file.roles.get("large")
    baseline = next(((c, n) for c, n in priced if large and n == large[0]), None)
    saving = round(float(1 - actual / baseline[0]), 4) if baseline and baseline[0] > 0 else None
    return {
        "all_traffic_on": [
            {"model": n, "cost_usd": _money(c), "vs_actual": _ratio(c, actual)} for c, n in priced
        ],
        "baseline_model": baseline[1] if baseline else None,
        "routing_saving_vs_baseline": saving,
    }


def build_report(
    factory: Callable[[], Session],
    models_file: ModelsFile | None = None,
    *,
    window_s: float = 7 * 86400,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    since = now - timedelta(seconds=window_s)
    with factory() as session:
        by_model = _by_model(session, since)
        runs, run_total = _per_run(session, since)
        by_tier = _by_tier(session, runs)
    tokens_in = sum(m["tokens_in"] for m in by_model)
    tokens_out = sum(m["tokens_out"] for m in by_model)
    actual = sum((Decimal(m["cost_usd"]) for m in by_model), Decimal(0))
    costs = [float(r.shadow_cost_usd) for r in runs]
    report: dict[str, Any] = {
        "window": {"since": since.isoformat(), "until": now.isoformat(), "seconds": window_s},
        "total": {
            "runs": run_total,
            "calls": sum(m["calls"] for m in by_model),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": _money(actual),
            "per_1000_runs_usd": _money(actual / run_total * 1000) if run_total else None,
        },
        "per_run": {
            "avg_usd": _money(sum(costs) / len(costs)) if costs else None,
            "p50_usd": _money(percentile(costs, 0.5)) if costs else None,
            "p95_usd": _money(percentile(costs, 0.95)) if costs else None,
            "max_usd": _money(max(costs)) if costs else None,
            "sampled_runs": len(costs),
        },
        "by_model": by_model,
        "by_tier": by_tier,
        "what_if": _what_if(models_file, tokens_in, tokens_out, actual) if models_file else None,
        "notes": list(NOTES),
    }
    return report


def render_text(report: dict[str, Any]) -> str:
    t, w = report["total"], report["window"]
    days = w["seconds"] / 86400
    lines = [
        f"Shadow cost report, last {days:g} day(s)",
        f"  {t['runs']} runs, {t['calls']} model calls, {t['tokens_in']:,} tokens in, "
        f"{t['tokens_out']:,} out",
        f"  cost ${t['cost_usd']}"
        + (f"  (${t['per_1000_runs_usd']} per 1,000 runs)" if t["per_1000_runs_usd"] else ""),
    ]
    p = report["per_run"]
    if p["avg_usd"] is not None:
        lines.append(
            f"  per run: avg ${p['avg_usd']}, median ${p['p50_usd']}, p95 ${p['p95_usd']}, "
            f"max ${p['max_usd']}"
        )
    lines += ["", "By model"]
    lines += [
        f"  {m['model']:<16} {m['calls']:>6} calls {m['tokens_in']:>10,} in "
        f"{m['tokens_out']:>9,} out  ${m['cost_usd']}"
        for m in report["by_model"]
    ] or ["  no model calls in this window"]
    lines += ["", "By routing tier"]
    lines += [
        f"  {g['tier']:<13} {g['runs']:>5} runs, {g['escalated']} escalated, {g['failed']} failed,"
        f" ${g['cost_usd']} total, per answered ticket "
        + (f"${g['cost_per_answered_usd']}" if g["cost_per_answered_usd"] else "n/a")
        for g in report["by_tier"]
    ] or ["  no runs in this window"]
    wi = report["what_if"]
    if wi:
        lines += ["", "What the same tokens would cost on one model for everything"]
        lines += [
            f"  {s['model']:<16} ${s['cost_usd']}"
            + (f"  ({s['vs_actual']:.2f}x actual)" if s["vs_actual"] is not None else "")
            for s in wi["all_traffic_on"]
        ]
        if wi["routing_saving_vs_baseline"] is not None:
            lines.append(
                f"  routing saves {wi['routing_saving_vs_baseline']:.0%} against everything on "
                f"{wi['baseline_model']}"
            )
    lines += ["", *[f"note: {n}" for n in report["notes"]]]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.config import get_settings, load_env
    from kartly.llm.config import load_models_file
    from kartly.llm.errors import LLMConfigError
    from kartly.store.db import make_engine, make_session_factory

    parser = argparse.ArgumentParser(description="Shadow cost report from stored runs.")
    parser.add_argument("--days", type=float, default=7.0, help="window length (default 7)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    if not 0 < args.days <= 365:
        print("error: --days must be between 0 and 365")
        return 2
    load_env()
    settings = get_settings()
    try:
        models_file: ModelsFile | None = load_models_file(settings.models_config_path)
    except LLMConfigError as exc:
        print(f"warning: no what-if section ({exc})")
        models_file = None
    engine = make_engine()
    try:
        report = build_report(make_session_factory(engine), models_file, window_s=args.days * 86400)
    finally:
        engine.dispose()
    print(json.dumps(report, indent=2) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
