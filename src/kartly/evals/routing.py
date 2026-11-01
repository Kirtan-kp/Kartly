"""Measure the Tier 3 routing rules on labelled messages (no model, no database).

``python -m kartly.evals.routing`` prints recall (of messages a human must handle, how many the
rules caught), the false-alarm rate (ordinary messages they flagged), and the phrasings that are
known to be wrong either way. The first two are gates: the Tier 3 recall target is 99% (DESIGN.md
section 2). Messages marked ``known`` are reported on their own line and never count, so the gate
stays honest about what it covers and the list shows where the classifier layer has to help.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from kartly.routing import RuleVerdict, route_by_rules

DEFAULT_PATH = Path("evals/routing/tier3.yaml")


class RoutingCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    category: str | None  # the Tier 3 category a human must handle; None = ordinary message
    message: str
    known: Literal["miss", "false_alarm"] | None = None


def load_routing_cases(path: Path = DEFAULT_PATH) -> list[RoutingCase]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of cases")
    try:
        cases = [RoutingCase.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    for case in cases:
        if case.known == "miss" and case.category is None:
            raise ValueError(f"{case.id}: a known miss needs a category")
        if case.known == "false_alarm" and case.category is not None:
            raise ValueError(f"{case.id}: a known false alarm must be an ordinary message")
    return cases


@dataclass
class RoutingReport:
    positives: int = 0
    caught: int = 0
    ordinary: int = 0
    false_alarms: int = 0
    missed: list[str] = field(default_factory=list)  # ids of Tier 3 messages the rules missed
    flagged: list[str] = field(default_factory=list)  # ids of ordinary messages they flagged
    wrong_category: list[str] = field(default_factory=list)
    known_misses_still_missed: int = 0
    known_misses_total: int = 0
    known_false_alarms_still_raised: int = 0
    known_false_alarms_total: int = 0
    by_category: dict[str, tuple[int, int]] = field(default_factory=dict)  # caught, total

    @property
    def recall(self) -> float | None:
        return self.caught / self.positives if self.positives else None

    @property
    def false_alarm_rate(self) -> float | None:
        return self.false_alarms / self.ordinary if self.ordinary else None

    @property
    def gate_passed(self) -> bool:
        return not self.missed and not self.flagged and self.positives > 0 and self.ordinary > 0


def evaluate_rules(
    cases: Sequence[RoutingCase], route: Callable[[str], RuleVerdict] = route_by_rules
) -> RoutingReport:
    report = RoutingReport()
    for case in cases:
        verdict = route(case.message)
        fired = verdict.tier >= 3
        if case.known == "miss":
            report.known_misses_total += 1
            report.known_misses_still_missed += 0 if fired else 1
            continue
        if case.known == "false_alarm":
            report.known_false_alarms_total += 1
            report.known_false_alarms_still_raised += 1 if fired else 0
            continue
        if case.category is None:
            report.ordinary += 1
            if fired:
                report.false_alarms += 1
                report.flagged.append(case.id)
            continue
        report.positives += 1
        caught, total = report.by_category.get(case.category, (0, 0))
        report.by_category[case.category] = (caught + (1 if fired else 0), total + 1)
        if fired:
            report.caught += 1
            if case.category not in verdict.categories:
                report.wrong_category.append(case.id)
        else:
            report.missed.append(case.id)
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_report(report: RoutingReport) -> str:
    lines = [
        f"Tier 3 recall: {report.caught}/{report.positives} ({_pct(report.recall)})",
        f"False alarms:  {report.false_alarms}/{report.ordinary} ({_pct(report.false_alarm_rate)})",
    ]
    for category, (caught, total) in sorted(report.by_category.items()):
        lines.append(f"  {category:<17} {caught}/{total}")
    if report.missed:
        lines.append(f"MISSED (gate): {', '.join(report.missed)}")
    if report.flagged:
        lines.append(f"FALSE ALARMS (gate): {', '.join(report.flagged)}")
    if report.wrong_category:
        lines.append(f"Caught under another category: {', '.join(report.wrong_category)}")
    lines.append(
        f"Known gaps: {report.known_misses_still_missed}/{report.known_misses_total} "
        f"phrasings still missed, {report.known_false_alarms_still_raised}/"
        f"{report.known_false_alarms_total} known false alarms still raised (not gated)"
    )
    lines.append("GATE: " + ("passed" if report.gate_passed else "FAILED"))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kartly.evals.routing", description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_PATH, help="routing cases (YAML)")
    parser.add_argument("--show", action="store_true", help="print every message and its verdict")
    args = parser.parse_args(argv)
    try:
        cases = load_routing_cases(args.cases)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.show:
        for case in cases:
            verdict = route_by_rules(case.message)
            mark = "T3" if verdict.tier >= 3 else "--"
            print(f"{mark} {case.id:<22} {','.join(verdict.categories) or '-':<18} {case.message}")
    report = evaluate_rules(cases)
    print(format_report(report))
    return 0 if report.gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
