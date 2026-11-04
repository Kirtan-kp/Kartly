"""Measure retrieval on labelled questions (no model, no database).

    python -m kartly.evals.retrieval                    # BM25 on kb/, evals/retrieval/queries.yaml
    python -m kartly.evals.retrieval --show             # every question with the chunks it got

hit@k is "a relevant chunk is among the top k", MRR@10 is the mean of 1/rank of the first relevant
chunk, recall@5 is the share of the relevant chunks found in the top 5. Questions the knowledge
base cannot answer (``relevant: []``) are not scored; their top scores are reported next to the
answerable ones so a "no answer" cut-off can be chosen from data later. Every run appends a row to
``evals/runs/retrieval_experiments.csv`` with fingerprints of the knowledge base and the questions,
so rows are only compared when both match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from kartly.retrieval import (
    BM25Retriever,
    Chunk,
    DenseRetriever,
    Embedder,
    EmbeddingError,
    HybridRetriever,
    KnowledgeBaseError,
    OllamaEmbedder,
    Retriever,
    kb_digest,
    load_kb,
)
from kartly.retrieval.dense import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL

DEFAULT_QUERIES = Path("evals/retrieval/queries.yaml")
DEFAULT_KB = "kb"
DEFAULT_OUT = "evals/runs"
DEPTH = 10  # how many results are fetched; hit@k and MRR are computed from them

# Each builder gets the chunks and a function that creates the embedder on first use, so "bm25"
# never needs Ollama.
RETRIEVERS: dict[str, Callable[[Sequence[Chunk], Callable[[], Embedder]], Retriever]] = {
    "bm25": lambda chunks, _embedder: BM25Retriever(chunks),
    "dense": lambda chunks, embedder: DenseRetriever(chunks, embedder()),
    "hybrid": lambda chunks, embedder: HybridRetriever(
        [BM25Retriever(chunks), DenseRetriever(chunks, embedder())]
    ),
}


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    query: str
    relevant: list[str]
    tags: list[str] = []


def load_queries(path: Path, chunks: Sequence[Chunk]) -> list[RetrievalCase]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of questions")
    try:
        cases = [RetrievalCase.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if len({c.id for c in cases}) != len(cases):
        raise ValueError(f"{path}: duplicate question ids")
    known = {c.id for c in chunks}
    for case in cases:
        missing = [r for r in case.relevant if r not in known]
        if missing:
            raise ValueError(f"{case.id}: unknown chunk id(s): {', '.join(missing)}")
        if len(set(case.relevant)) != len(case.relevant):
            raise ValueError(f"{case.id}: a chunk is listed twice")
    return cases


def queries_digest(cases: Sequence[RetrievalCase]) -> str:
    material = json.dumps([c.model_dump() for c in cases], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()[:8]


@dataclass
class Outcome:
    case: RetrievalCase
    ranked: list[str]  # chunk ids, best first
    scores: list[float]

    @property
    def first_relevant_rank(self) -> int | None:
        for rank, chunk_id in enumerate(self.ranked, start=1):
            if chunk_id in self.case.relevant:
                return rank
        return None

    def hit(self, k: int) -> bool:
        rank = self.first_relevant_rank
        return rank is not None and rank <= k

    def recall(self, k: int) -> float:
        found = sum(1 for r in self.case.relevant if r in self.ranked[:k])
        return found / len(self.case.relevant)

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank
        return 1 / rank if rank else 0.0

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0


def run_retrieval_eval(cases: Sequence[RetrievalCase], retriever: Retriever) -> list[Outcome]:
    outcomes = []
    for case in cases:
        hits = retriever.search(case.query, DEPTH)
        outcomes.append(Outcome(case, [h.chunk.id for h in hits], [h.score for h in hits]))
    return outcomes


@dataclass(frozen=True)
class Metrics:
    questions: int
    hit1: float | None
    hit3: float | None
    hit5: float | None
    recall5: float | None
    mrr: float | None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(outcomes: Sequence[Outcome]) -> Metrics:
    scored = [o for o in outcomes if o.case.relevant]
    return Metrics(
        questions=len(scored),
        hit1=_mean([float(o.hit(1)) for o in scored]),
        hit3=_mean([float(o.hit(3)) for o in scored]),
        hit5=_mean([float(o.hit(5)) for o in scored]),
        recall5=_mean([o.recall(5) for o in scored]),
        mrr=_mean([o.reciprocal_rank for o in scored]),
    )


@dataclass(frozen=True)
class Cutoff:
    """ "Answer only if the best score is at least ``score``": the choice that best separates
    answerable from unanswerable questions on this run (ties go to the lower score)."""

    score: float
    answerable_kept: int
    answerable_total: int
    unanswerable_rejected: int
    unanswerable_total: int


def best_cutoff(outcomes: Sequence[Outcome]) -> Cutoff | None:
    answerable = [o.top_score for o in outcomes if o.case.relevant]
    unanswerable = [o.top_score for o in outcomes if not o.case.relevant]
    if not answerable or not unanswerable:
        return None
    best: tuple[float, float] | None = None
    chosen = 0.0
    for candidate in sorted(set(answerable + unanswerable)):
        kept = sum(s >= candidate for s in answerable)
        rejected = sum(s < candidate for s in unanswerable)
        value = kept / len(answerable) + rejected / len(unanswerable)
        if best is None or value > best[0]:
            best = (value, candidate)
            chosen = candidate
    return Cutoff(
        score=chosen,
        answerable_kept=sum(s >= chosen for s in answerable),
        answerable_total=len(answerable),
        unanswerable_rejected=sum(s < chosen for s in unanswerable),
        unanswerable_total=len(unanswerable),
    )


@dataclass(frozen=True)
class Report:
    retriever: str
    kb_digest: str
    queries_digest: str
    chunks: int
    overall: Metrics
    by_tag: dict[str, Metrics]
    misses: list[str]  # answerable questions with no relevant chunk in the top 5
    answerable_top_score: float | None  # mean best score
    unanswerable_top_score: float | None
    unanswerable_max_score: float | None
    answerable_min_score: float | None
    cutoff: Cutoff | None

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = json.loads(json.dumps(self, default=lambda o: o.__dict__))
        return document


def build_report(
    outcomes: Sequence[Outcome],
    retriever: str,
    chunks: Sequence[Chunk],
    cases: Sequence[RetrievalCase],
) -> Report:
    tags: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        for tag in o.case.tags:
            if tag != "unanswerable":
                tags[tag].append(o)
    unanswerable = [o for o in outcomes if not o.case.relevant]
    answerable = [o for o in outcomes if o.case.relevant]
    return Report(
        retriever=retriever,
        kb_digest=kb_digest(list(chunks)),
        queries_digest=queries_digest(cases),
        chunks=len(chunks),
        overall=compute_metrics(outcomes),
        by_tag={tag: compute_metrics(group) for tag, group in sorted(tags.items())},
        misses=[o.case.id for o in answerable if not o.hit(5)],
        answerable_top_score=_mean([o.top_score for o in answerable]),
        unanswerable_top_score=_mean([o.top_score for o in unanswerable]),
        unanswerable_max_score=max((o.top_score for o in unanswerable), default=None),
        answerable_min_score=min((o.top_score for o in answerable), default=None),
        cutoff=best_cutoff(outcomes),
    )


def _f(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_report(report: Report) -> str:
    m = report.overall
    lines = [
        f"Retriever {report.retriever}, kb {report.kb_digest} ({report.chunks} chunks), "
        f"questions {report.queries_digest}",
        f"  {m.questions} answerable questions: hit@1 {_f(m.hit1)}  hit@3 {_f(m.hit3)}  "
        f"hit@5 {_f(m.hit5)}  recall@5 {_f(m.recall5)}  MRR@10 {_f(m.mrr)}",
    ]
    for tag, tm in report.by_tag.items():
        lines.append(
            f"  {tag:<11} n={tm.questions:<3} hit@1 {_f(tm.hit1)}  hit@3 {_f(tm.hit3)}  "
            f"hit@5 {_f(tm.hit5)}  MRR {_f(tm.mrr)}"
        )
    lines.append(
        f"  top score: answerable mean {_f(report.answerable_top_score, 2)}, "
        f"unanswerable mean {_f(report.unanswerable_top_score, 2)} "
        f"(max {_f(report.unanswerable_max_score, 2)}); lowest answerable "
        f"{_f(report.answerable_min_score, 2)}"
    )
    if report.cutoff is not None:
        c = report.cutoff
        lines.append(
            f"  best no-answer cut-off {c.score:.3f}: "
            f"keeps {c.answerable_kept}/{c.answerable_total} answerable, "
            f"rejects {c.unanswerable_rejected}/{c.unanswerable_total} unanswerable"
        )
    lines.append("  not in top 5: " + (", ".join(report.misses) if report.misses else "none"))
    return "\n".join(lines)


def format_outcome(o: Outcome) -> str:
    rank = o.first_relevant_rank
    if not o.case.relevant:
        mark = "n/a "
    elif rank == 1:
        mark = "ok  "
    elif rank is not None and rank <= 5:
        mark = "low "
    else:
        mark = "MISS"
    top = ", ".join(o.ranked[:3]) or "(nothing)"
    want = ", ".join(o.case.relevant) or "(unanswerable)"
    return f"[{mark}] {o.case.id} {o.case.query!r}\n        want {want}\n        got  {top}"


_LOG_COLUMNS = (
    "timestamp_utc", "retriever", "kb_digest", "queries_digest", "chunks", "questions", "hit1",
    "hit3", "hit5", "recall5", "mrr", "note", "results_file",
)  # fmt: skip


def _cell(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def save_run(
    outcomes: Sequence[Outcome],
    report: Report,
    out_dir: Path,
    *,
    note: str = "",
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", report.retriever)
    path = out_dir / f"{now:%Y%m%d-%H%M%S}_retrieval_{safe}.json"
    document = {
        "timestamp_utc": now.isoformat(timespec="seconds"),
        "note": note,
        "report": report.to_dict(),
        "results": [
            {"id": o.case.id, "query": o.case.query, "want": o.case.relevant, "got": o.ranked[:5]}
            for o in outcomes
        ],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    log = out_dir / "retrieval_experiments.csv"
    is_new = not log.exists()
    m = report.overall
    with log.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(_LOG_COLUMNS)
        writer.writerow(
            [
                now.isoformat(timespec="seconds"), report.retriever, report.kb_digest,
                report.queries_digest, report.chunks, m.questions, _cell(m.hit1), _cell(m.hit3),
                _cell(m.hit5), _cell(m.recall5), _cell(m.mrr), note, path.name,
            ]
        )  # fmt: skip
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kartly.evals.retrieval", description=__doc__)
    parser.add_argument("--retriever", default="bm25", choices=sorted(RETRIEVERS))
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--kb", default=DEFAULT_KB, help="directory of knowledge-base *.md files")
    parser.add_argument("--cases", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--show", action="store_true", help="print every question and its results")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--note", default="")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    try:
        chunks = load_kb(args.kb)
        cases = load_queries(args.cases, chunks)
    except (KnowledgeBaseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)

    def make_embedder() -> Embedder:
        return OllamaEmbedder(
            args.embed_model, args.ollama_url, cache_path=out_dir / "embedding_cache.json"
        )

    try:
        retriever = RETRIEVERS[args.retriever](chunks, make_embedder)
        outcomes = run_retrieval_eval(cases, retriever)
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.show:
        for outcome in outcomes:
            print(format_outcome(outcome))
        print()
    report = build_report(outcomes, retriever.name, chunks, cases)
    print(format_report(report))
    if not args.no_save:
        path = save_run(outcomes, report, out_dir, note=args.note)
        print(f"\nSaved {path} and a row in {out_dir / 'retrieval_experiments.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
