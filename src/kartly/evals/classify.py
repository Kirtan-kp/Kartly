"""Measure the routing classifier on labelled messages.

    python -m kartly.evals.classify --model local_small
    python -m kartly.evals.classify --model groq_small --pause 5 --note "c1 baseline"

For each message it runs the real classifier, the Tier 3 rules and ``decide_route``, and reports
how often each gets the intent and tier right. The numbers that matter most are the dangerous
ones: Tier 3 recall of the whole router (rules plus classifier), and how many tickets were routed
to a LOWER tier than they deserve. Results go to ``evals/runs/`` and one row is appended to
``routing_experiments.csv``. Exit code 0 means every Tier 3 message was caught and no output was
unusable; 1 otherwise; 2 for a configuration problem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from kartly.config import get_settings, load_env
from kartly.llm import ChatClient, LLMConfigError, ModelRegistry
from kartly.routing import (
    DEFAULT_VERSION,
    PROMPTS,
    ClassifierResult,
    RouteDecision,
    RuleVerdict,
    classify,
    decide_route,
    route_by_rules,
)
from kartly.routing.classifier import INTENTS

DEFAULT_PATH = Path("evals/routing/intents.yaml")
DEFAULT_OUT = "evals/runs"


class IntentCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    intent: str
    tier: int
    message: str


def load_intent_cases(path: Path = DEFAULT_PATH) -> list[IntentCase]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of cases")
    try:
        cases = [IntentCase.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if len({c.id for c in cases}) != len(cases):
        raise ValueError(f"{path}: duplicate case ids")
    for case in cases:
        if case.intent not in INTENTS:
            raise ValueError(f"{case.id}: unknown intent {case.intent!r}")
        if not 0 <= case.tier <= 3:
            raise ValueError(f"{case.id}: tier must be 0 to 3")
    return cases


def intents_digest(cases: Sequence[IntentCase]) -> str:
    material = json.dumps([c.model_dump() for c in cases], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:8]


@dataclass
class Outcome:
    case: IntentCase
    result: ClassifierResult
    rules: RuleVerdict
    decision: RouteDecision

    @property
    def predicted_intent(self) -> str:
        c = self.result.classification
        return c.intent if c else "(unusable)"


def run_case(case: IntentCase, client: ChatClient, version: str = DEFAULT_VERSION) -> Outcome:
    result = classify(client, case.message, version)
    rules = route_by_rules(case.message)
    return Outcome(case, result, rules, decide_route(rules, result))


def run_classifier_eval(
    cases: Sequence[IntentCase],
    client: ChatClient,
    *,
    pause_s: float = 0.0,
    version: str = DEFAULT_VERSION,
    on_result: Callable[[Outcome], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for case in cases:
        outcome = run_case(case, client, version)
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)
        if pause_s > 0 and len(outcomes) < len(cases):
            sleep(pause_s)
    return outcomes


def _rate(n: int, d: int) -> float | None:
    return n / d if d else None


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0


@dataclass(frozen=True)
class ClassifierSummary:
    model: str
    classifier_version: str
    cases_digest: str
    cases: int
    unusable: int
    intent_accuracy: float | None  # unusable output counts as wrong
    tier_accuracy: float | None  # of the final route (rules + classifier)
    under_routed: list[str]  # case ids routed to a LOWER tier than labelled (the dangerous way)
    over_routed: list[str]
    tier3_recall: float | None  # whole router
    tier3_recall_rules: float | None
    tier3_recall_classifier: float | None
    tier3_missed: list[str]
    tier3_false_alarm_rate: float | None
    clarification_rate: float | None
    avg_tokens: float
    total_cost_usd: Decimal
    p50_ms: int
    p95_ms: int
    confusion: dict[str, dict[str, int]]

    @property
    def gate_passed(self) -> bool:
        return not self.tier3_missed and self.unusable == 0 and self.cases > 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["total_cost_usd"] = str(self.total_cost_usd)
        return d


def summarize_classifier(
    outcomes: Sequence[Outcome],
    cases: Sequence[IntentCase],
    model: str,
    version: str = DEFAULT_VERSION,
) -> ClassifierSummary:
    n = len(outcomes)
    unusable = sum(1 for o in outcomes if o.result.classification is None)
    intent_ok = sum(1 for o in outcomes if o.predicted_intent == o.case.intent)
    tier_ok = sum(1 for o in outcomes if o.decision.tier == o.case.tier)
    under = [o.case.id for o in outcomes if o.decision.tier < o.case.tier]
    over = [o.case.id for o in outcomes if o.decision.tier > o.case.tier]
    t3 = [o for o in outcomes if o.case.tier == 3]
    not_t3 = [o for o in outcomes if o.case.tier < 3]
    caught = [o for o in t3 if o.decision.tier >= 3]
    by_rules = [o for o in t3 if o.rules.tier >= 3]
    by_model = [
        o
        for o in t3
        if o.result.classification is not None and o.result.classification.effective_tier >= 3
    ]
    confusion: dict[str, dict[str, int]] = {}
    for o in outcomes:
        confusion.setdefault(o.case.intent, Counter())[o.predicted_intent] += 1
    latencies = [o.result.latency_ms for o in outcomes if o.result.classification or o.result.raw]
    return ClassifierSummary(
        model=model,
        classifier_version=version,
        cases=n,
        cases_digest=intents_digest(cases),
        unusable=unusable,
        intent_accuracy=_rate(intent_ok, n),
        tier_accuracy=_rate(tier_ok, n),
        under_routed=under,
        over_routed=over,
        tier3_recall=_rate(len(caught), len(t3)),
        tier3_recall_rules=_rate(len(by_rules), len(t3)),
        tier3_recall_classifier=_rate(len(by_model), len(t3)),
        tier3_missed=[o.case.id for o in t3 if o.decision.tier < 3],
        tier3_false_alarm_rate=_rate(sum(1 for o in not_t3 if o.decision.tier >= 3), len(not_t3)),
        clarification_rate=_rate(sum(1 for o in outcomes if o.decision.needs_clarification), n),
        avg_tokens=(
            sum(o.result.tokens_in + o.result.tokens_out for o in outcomes) / n if n else 0.0
        ),
        total_cost_usd=sum((o.result.cost_usd for o in outcomes), Decimal(0)),
        p50_ms=_percentile(latencies, 0.5),
        p95_ms=_percentile(latencies, 0.95),
        confusion={k: dict(v) for k, v in confusion.items()},
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_outcome(o: Outcome) -> str:
    d = o.decision
    if o.result.classification is None:
        return f"[BAD ] {o.case.id:<22} {o.result.error}"
    ok = o.predicted_intent == o.case.intent and d.tier == o.case.tier
    mark = "ok  " if ok else "MISS"
    note = ""
    if d.tier < o.case.tier:
        note = "  <- ROUTED TOO LOW"
    return (
        f"[{mark}] {o.case.id:<22} {o.predicted_intent:<14} tier {d.tier} "
        f"(want {o.case.intent}/{o.case.tier}) conf {d.confidence:.2f}{note}"
    )


def format_confusion(confusion: dict[str, dict[str, int]]) -> str:
    lines = ["Confusion (true -> predicted, only wrong ones):"]
    wrong = 0
    for true, row in sorted(confusion.items()):
        for predicted, count in sorted(row.items()):
            if predicted != true:
                lines.append(f"  {true:<14} -> {predicted:<14} x{count}")
                wrong += 1
    if not wrong:
        lines.append("  (none)")
    return "\n".join(lines)


def format_summary(s: ClassifierSummary) -> str:
    lines = [
        f"Model {s.model}, classifier {s.classifier_version}, cases {s.cases_digest}",
        f"  intent accuracy {_pct(s.intent_accuracy)}, tier accuracy {_pct(s.tier_accuracy)}"
        f", unusable outputs {s.unusable}/{s.cases}",
        f"  Tier 3 recall {_pct(s.tier3_recall)} (rules alone {_pct(s.tier3_recall_rules)}, "
        f"classifier alone {_pct(s.tier3_recall_classifier)}); "
        f"false alarms {_pct(s.tier3_false_alarm_rate)}",
        f"  routed too low: {len(s.under_routed)}, too high: {len(s.over_routed)}, "
        f"needs clarification {_pct(s.clarification_rate)}",
        f"  avg {s.avg_tokens:.0f} tokens, shadow cost ${s.total_cost_usd:.6f}, "
        f"latency p50 {s.p50_ms / 1000:.1f}s p95 {s.p95_ms / 1000:.1f}s",
    ]
    if s.tier3_missed:
        lines.append(f"  TIER 3 MISSED: {', '.join(s.tier3_missed)}")
    if s.under_routed:
        lines.append(f"  routed too low: {', '.join(s.under_routed)}")
    lines.append(
        "GATE (every Tier 3 caught, no unusable output): "
        + ("passed" if s.gate_passed else "FAILED")
    )
    return "\n".join(lines)


_LOG_COLUMNS = (
    "timestamp_utc", "model", "classifier_version", "cases_digest", "cases", "unusable",
    "intent_accuracy", "tier_accuracy", "tier3_recall", "tier3_recall_rules",
    "tier3_recall_classifier", "tier3_false_alarm_rate", "under_routed", "over_routed",
    "avg_tokens", "total_cost_usd", "p50_ms", "p95_ms", "note", "results_file",
)  # fmt: skip


def _cell(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def save_classifier_run(
    outcomes: Sequence[Outcome],
    summary: ClassifierSummary,
    out_dir: Path,
    *,
    note: str = "",
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", summary.model)
    path = out_dir / f"{now:%Y%m%d-%H%M%S}_routing_{safe_model}.json"
    document = {
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "note": note,
        "summary": summary.to_dict(),
        "results": [
            {
                "id": o.case.id,
                "message": o.case.message,
                "want": {"intent": o.case.intent, "tier": o.case.tier},
                "got": {
                    "intent": o.predicted_intent,
                    "tier": o.decision.tier,
                    "confidence": o.decision.confidence,
                    "rule_categories": list(o.rules.categories),
                },
                "error": o.result.error,
                "raw": o.result.raw,
            }
            for o in outcomes
        ],
    }
    path.write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")
    log = out_dir / "routing_experiments.csv"
    is_new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(_LOG_COLUMNS)
        s = summary
        writer.writerow(
            [
                now.isoformat(timespec="seconds"), s.model, s.classifier_version, s.cases_digest,
                s.cases, s.unusable, _cell(s.intent_accuracy), _cell(s.tier_accuracy),
                _cell(s.tier3_recall), _cell(s.tier3_recall_rules),
                _cell(s.tier3_recall_classifier), _cell(s.tier3_false_alarm_rate),
                len(s.under_routed), len(s.over_routed), f"{s.avg_tokens:.0f}",
                f"{s.total_cost_usd:.6f}", s.p50_ms, s.p95_ms, note, path.name,
            ]
        )  # fmt: skip
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kartly.evals.classify", description=__doc__)
    parser.add_argument("--model", help="model name from config/models.yaml")
    parser.add_argument("--cases", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these case ids")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between calls")
    parser.add_argument(
        "--classifier-version",
        default=DEFAULT_VERSION,
        choices=sorted(PROMPTS),
        help="which classifier prompt to use (compare versions on the same cases digest)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--note", default="")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--config", help="path to models.yaml")
    args = parser.parse_args(argv)

    try:
        cases = load_intent_cases(args.cases)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.only:
        unknown = [i for i in args.only if i not in {c.id for c in cases}]
        if unknown:
            print(f"error: unknown case id(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in set(args.only)]
    if not args.model:
        print("error: --model is required", file=sys.stderr)
        return 2
    if args.pause < 0:
        print("error: --pause cannot be negative", file=sys.stderr)
        return 2

    load_env()
    try:
        registry = ModelRegistry.from_yaml(args.config or get_settings().models_config_path)
        client = registry.client(args.model)
    except LLMConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Classifying {len(cases)} message(s) with {args.model} ...", flush=True)
    with client:
        outcomes = run_classifier_eval(
            cases,
            client,
            pause_s=args.pause,
            version=args.classifier_version,
            on_result=lambda o: print(format_outcome(o), flush=True),
        )
    summary = summarize_classifier(outcomes, cases, args.model, args.classifier_version)
    print()
    print(format_confusion(summary.confusion))
    print()
    print(format_summary(summary))
    if not args.no_save:
        path = save_classifier_run(outcomes, summary, Path(args.out), note=args.note)
        print(f"\nSaved {path} and a row in {Path(args.out) / 'routing_experiments.csv'}")
    return 0 if summary.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
