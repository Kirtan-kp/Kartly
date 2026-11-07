"""Measure the claim judge on labelled replies (needs a model; no database).

    python -m kartly.evals.claims --model groq_small          # evals/claims/cases.yaml on kb/
    python -m kartly.evals.claims --model groq_small --show   # also print what it flagged

Each case is a reply plus the policy chunks it was written from, labelled ``supported`` (every
statement is in the evidence) or ``unsupported`` (at least one is not). Two rates matter, and they
pull against each other:

- caught: of the unsupported replies, how many the judge held (a miss sends an invented claim
  to a customer);
- falsely held: of the supported replies, how many the judge held (each one costs a human a look).

A judge that errors or returns unusable output counts as a hold in production (fail closed), so it
is counted as a hold here too, and also reported on its own. Every run appends a row to
``evals/runs/claims_experiments.csv`` with the judge version and fingerprints of the cases and the
knowledge base, so rows compare only when those match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from kartly.config import get_settings, load_env
from kartly.guards.claims import JUDGE_VERSION, ClaimJudge, ClaimVerdict
from kartly.llm.errors import LLMConfigError
from kartly.llm.registry import ModelRegistry
from kartly.retrieval import Chunk, KnowledgeBaseError, kb_digest, load_kb

DEFAULT_CASES = Path("evals/claims/cases.yaml")
DEFAULT_KB = "kb"
DEFAULT_OUT = "evals/runs"


class ClaimCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    chunks: list[str]
    reply: str
    label: Literal["supported", "unsupported"]
    extra_evidence: str = ""
    tags: list[str] = []


def load_cases(path: Path | str) -> list[ClaimCase]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must be a non-empty list of cases")
    try:
        cases = [ClaimCase.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate case ids")
    return cases


def cases_digest(cases: Sequence[ClaimCase]) -> str:
    blob = "\n".join(c.model_dump_json() for c in cases)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def evidence_for(case: ClaimCase, chunks: dict[str, Chunk]) -> str:
    """The text the judge sees for a case, laid out as ``evidence_from_messages`` lays it out."""
    missing = [c for c in case.chunks if c not in chunks]
    if missing:
        raise ValueError(f"case {case.id} names chunks not in the knowledge base: {missing}")
    parts = [f"[{cid}] {chunks[cid].text}" for cid in case.chunks]
    if case.extra_evidence:
        parts.append(case.extra_evidence)
    return "\n\n".join(parts)


@dataclass(frozen=True)
class ClaimResult:
    case: ClaimCase
    verdict: ClaimVerdict
    attempt: int = 1

    @property
    def held(self) -> bool:
        """What production would do: hold on a flagged claim, and also when the judge failed."""
        return not self.verdict.ok

    @property
    def correct(self) -> bool:
        return self.held == (self.case.label == "unsupported")


def run_cases(
    cases: Sequence[ClaimCase],
    chunks: dict[str, Chunk],
    judge: ClaimJudge,
    *,
    repeat: int = 1,
    pause_s: float = 0.0,
    on_result: Callable[[ClaimResult], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[ClaimResult]:
    results: list[ClaimResult] = []
    total = len(cases) * repeat
    for attempt in range(1, repeat + 1):
        for case in cases:
            verdict = judge.judge(case.question, evidence_for(case, chunks), case.reply)
            result = ClaimResult(case, verdict, attempt)
            results.append(result)
            if on_result is not None:
                on_result(result)
            if pause_s > 0 and len(results) < total:
                sleep(pause_s)
    return results


@dataclass(frozen=True)
class ClaimSummary:
    model: str
    judge_version: str
    cases_digest: str
    kb_digest: str
    runs: int
    unsupported_runs: int
    caught: int
    supported_runs: int
    falsely_held: int
    errors: int
    flaky_cases: tuple[str, ...]  # cases whose verdict changed between repeats
    missed: tuple[str, ...]  # unsupported case ids that were not held in at least one run
    avg_tokens: float
    p50_latency_s: float
    total_cost_usd: Decimal

    @property
    def caught_rate(self) -> float | None:
        return self.caught / self.unsupported_runs if self.unsupported_runs else None

    @property
    def false_hold_rate(self) -> float | None:
        return self.falsely_held / self.supported_runs if self.supported_runs else None


def summarize(
    results: Sequence[ClaimResult], model: str, cases: Sequence[ClaimCase], kb: str
) -> ClaimSummary:
    unsupported = [r for r in results if r.case.label == "unsupported"]
    supported = [r for r in results if r.case.label == "supported"]
    held_by_case: dict[str, set[bool]] = defaultdict(set)
    for r in results:
        held_by_case[r.case.id].add(r.held)
    times = [r.verdict.latency_ms / 1000 for r in results]
    return ClaimSummary(
        model=model,
        judge_version=JUDGE_VERSION,
        cases_digest=cases_digest(cases),
        kb_digest=kb,
        runs=len(results),
        unsupported_runs=len(unsupported),
        caught=sum(r.held for r in unsupported),
        supported_runs=len(supported),
        falsely_held=sum(r.held for r in supported),
        errors=sum(r.verdict.error is not None for r in results),
        flaky_cases=tuple(sorted(c for c, v in held_by_case.items() if len(v) > 1)),
        missed=tuple(sorted({r.case.id for r in unsupported if not r.held})),
        avg_tokens=(
            statistics.fmean(r.verdict.tokens_in + r.verdict.tokens_out for r in results)
            if results
            else 0.0
        ),
        p50_latency_s=statistics.median(times) if times else 0.0,
        total_cost_usd=sum((r.verdict.cost_usd for r in results), Decimal(0)),
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_result(result: ClaimResult, show: bool = False) -> str:
    v = result.verdict
    expected = "unsupported" if result.case.label == "unsupported" else "supported"
    status = "ok  " if result.correct else "MISS" if result.case.label == "unsupported" else "HELD"
    suffix = f" #{result.attempt}" if result.attempt > 1 else ""
    line = f"[{status}] {result.case.id}{suffix}  (expected {expected})  {v.latency_ms / 1000:.1f}s"
    if v.error:
        line += f"  error: {v.error}"
    if show or not result.correct:
        for claim in v.unsupported:
            line += f"\n        flagged: {claim}"
    return line


def format_summary(s: ClaimSummary) -> str:
    lines = [
        f"Claim judge {s.model} ({s.judge_version}), cases {s.cases_digest}, kb {s.kb_digest}",
        f"  caught {s.caught}/{s.unsupported_runs} unsupported replies ({_pct(s.caught_rate)})",
        f"  falsely held {s.falsely_held}/{s.supported_runs} supported replies "
        f"({_pct(s.false_hold_rate)})",
        f"  judge errors/unusable output: {s.errors} (counted as holds)",
        f"  avg {s.avg_tokens:,.0f} tokens/check, shadow cost ${s.total_cost_usd:.6f}, "
        f"latency p50 {s.p50_latency_s:.1f}s",
    ]
    if s.missed:
        lines.append("  missed: " + ", ".join(s.missed))
    if s.flaky_cases:
        lines.append("  changed between repeats: " + ", ".join(s.flaky_cases))
    return "\n".join(lines)


def save_run(summary: ClaimSummary, out_dir: Path, note: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "claims_experiments.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "judge_version",
                    "cases_digest",
                    "kb_digest",
                    "runs",
                    "caught",
                    "unsupported_runs",
                    "falsely_held",
                    "supported_runs",
                    "errors",
                    "avg_tokens",
                    "p50_latency_s",
                    "cost_usd",
                    "note",
                ]
            )
        writer.writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                summary.model,
                summary.judge_version,
                summary.cases_digest,
                summary.kb_digest,
                summary.runs,
                summary.caught,
                summary.unsupported_runs,
                summary.falsely_held,
                summary.supported_runs,
                summary.errors,
                f"{summary.avg_tokens:.0f}",
                f"{summary.p50_latency_s:.2f}",
                f"{summary.total_cost_usd:.6f}",
                note,
            ]
        )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the claim judge on labelled replies.")
    parser.add_argument("--model", required=True, help="model name from config/models.yaml")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--kb", default=DEFAULT_KB)
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these case ids")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case, to see variance")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between calls")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--show", action="store_true", help="print flagged claims for every case")
    parser.add_argument("--config", help="path to models.yaml")
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.pause < 0:
        print("error: --repeat must be at least 1 and --pause cannot be negative", file=sys.stderr)
        return 2
    load_env()
    try:
        cases = load_cases(args.cases)
        chunk_list = load_kb(args.kb)
        chunks = {c.id: c for c in chunk_list}
        for case in cases:
            evidence_for(case, chunks)  # fail early on a bad chunk id
        if args.only:
            unknown = sorted(set(args.only) - {c.id for c in cases})
            if unknown:
                print(f"error: unknown case ids: {', '.join(unknown)}", file=sys.stderr)
                return 2
            cases = [c for c in cases if c.id in set(args.only)]
        registry = ModelRegistry.from_yaml(args.config or get_settings().models_config_path)
        client = registry.client(args.model)
    except (ValueError, KnowledgeBaseError, LLMConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Running {len(cases)} case(s) x {args.repeat} with judge {args.model} ...", flush=True)
    with client:
        results = run_cases(
            cases,
            chunks,
            ClaimJudge(client),
            repeat=args.repeat,
            pause_s=args.pause,
            on_result=lambda r: print(format_result(r, args.show), flush=True),
        )
    summary = summarize(results, args.model, cases, kb_digest(chunk_list))
    print()
    print(format_summary(summary))
    if not args.no_save:
        path = save_run(summary, Path(args.out))
        print(f"\nSaved a row in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
