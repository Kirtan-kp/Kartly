"""The retrieval eval: metric arithmetic on made-up rankings, and the real BM25 baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from kartly.evals.retrieval import (
    Outcome,
    RetrievalCase,
    build_report,
    compute_metrics,
    format_outcome,
    format_report,
    load_queries,
    main,
    queries_digest,
    run_retrieval_eval,
    save_run,
)
from kartly.retrieval import BM25Retriever, Chunk, Hit, load_kb

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "kb"
QUERIES = ROOT / "evals" / "retrieval" / "queries.yaml"
CHUNKS = load_kb(KB)


def case(id_: str, relevant: list[str], tags: list[str] | None = None) -> RetrievalCase:
    return RetrievalCase(id=id_, query=f"query {id_}", relevant=relevant, tags=tags or [])


def outcome(relevant: list[str], ranked: list[str], scores: list[float] | None = None) -> Outcome:
    return Outcome(case("q", relevant), ranked, scores or [1.0] * len(ranked))


# ----------------------------------------------------------------------------- the metrics


def test_hit_rank_and_reciprocal_rank() -> None:
    o = outcome(["c"], ["a", "b", "c", "d"])
    assert o.first_relevant_rank == 3 and o.reciprocal_rank == pytest.approx(1 / 3)
    assert not o.hit(2) and o.hit(3) and o.hit(5)
    miss = outcome(["z"], ["a", "b"])
    assert miss.first_relevant_rank is None and miss.reciprocal_rank == 0.0 and not miss.hit(5)


def test_recall_counts_the_share_of_relevant_chunks_found() -> None:
    o = outcome(["a", "b", "c", "d"], ["a", "x", "b", "y", "z", "c"])
    assert o.recall(5) == 0.5 and o.recall(6) == 0.75


def test_metrics_average_over_answerable_questions_only() -> None:
    outcomes = [
        outcome(["a"], ["a", "b"]),  # rank 1
        outcome(["a"], ["b", "a"]),  # rank 2
        outcome(["a"], ["b", "c"]),  # miss
        outcome([], ["a"]),  # unanswerable: not scored
    ]
    m = compute_metrics(outcomes)
    assert m.questions == 3
    assert m.hit1 == pytest.approx(1 / 3) and m.hit3 == pytest.approx(2 / 3)
    assert m.mrr == pytest.approx((1 + 0.5 + 0) / 3)


def test_no_answerable_questions_gives_none_not_a_crash() -> None:
    m = compute_metrics([outcome([], ["a"])])
    assert m.questions == 0 and m.hit1 is None and m.mrr is None


class FixedRetriever:
    name = "fixed"

    def __init__(self, ranking: dict[str, list[str]]) -> None:
        self.ranking = ranking

    def search(self, query: str, k: int = 5) -> list[Hit]:
        by_id = {c.id: c for c in CHUNKS}
        return [Hit(by_id[i], 10.0 - n) for n, i in enumerate(self.ranking.get(query, [])[:k])]


def test_run_and_report_with_tags_misses_and_score_ranges() -> None:
    a, b, c = CHUNKS[0].id, CHUNKS[1].id, CHUNKS[2].id
    cases = [
        case("hit", [a], ["plain"]),
        case("miss", [a], ["paraphrase"]),
        case("none", [], ["unanswerable"]),
    ]
    retriever = FixedRetriever({"query hit": [a, b], "query miss": [b, c], "query none": [c]})
    outcomes = run_retrieval_eval(cases, retriever)
    report = build_report(outcomes, "fixed", CHUNKS, cases)
    assert report.overall.hit1 == 0.5 and report.misses == ["miss"]
    assert set(report.by_tag) == {"plain", "paraphrase"}  # unanswerable is not a scored tag
    assert report.by_tag["plain"].hit1 == 1.0 and report.by_tag["paraphrase"].hit5 == 0.0
    assert report.answerable_top_score == 10.0 and report.unanswerable_max_score == 10.0
    text = format_report(report)
    assert "hit@1 0.500" in text and "not in top 5: miss" in text and "paraphrase" in text
    assert "[MISS]" in format_outcome(outcomes[1]) and "[ok  ]" in format_outcome(outcomes[0])
    assert "(unanswerable)" in format_outcome(outcomes[2])


def test_a_hit_below_rank_one_is_marked_low() -> None:
    o = outcome(["a"], ["b", "a"])
    assert format_outcome(o).startswith("[low ]")


# ------------------------------------------------------------------------------ the dataset


def test_the_real_questions_load_against_the_real_knowledge_base() -> None:
    cases = load_queries(QUERIES, CHUNKS)
    assert len(cases) >= 40
    assert sum(1 for c in cases if not c.relevant) >= 3
    assert {t for c in cases for t in c.tags} >= {"plain", "paraphrase", "messy", "multi"}
    assert sum(1 for c in cases if len(c.relevant) > 1) >= 5


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("x: 1", "expected a list"),
        ("- {id: a, query: q, relevant: [nope#x]}", "unknown chunk id"),
        ("- {id: a, query: q, relevant: []}\n" * 2, "duplicate"),
        (f"- {{id: a, query: q, relevant: [{CHUNKS[0].id}, {CHUNKS[0].id}]}}", "listed twice"),
        ("- {id: a, query: q, relevant: [], extra: 1}", "extra"),
        ("{", "cannot read"),
    ],
)
def test_bad_question_files_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_queries(path, CHUNKS)


def test_the_questions_digest_follows_the_labels() -> None:
    a = [case("a", [CHUNKS[0].id])]
    assert queries_digest(a) == queries_digest(list(a))
    assert queries_digest(a) != queries_digest([case("a", [CHUNKS[1].id])])


# --------------------------------------------------------------------- the real BM25 baseline


def test_bm25_stays_above_a_floor_on_the_real_questions() -> None:
    """A regression guard, not a target: if a change to chunking, stemming or the knowledge base
    drops these, find out why before adjusting them."""
    cases = load_queries(QUERIES, CHUNKS)
    report = build_report(run_retrieval_eval(cases, BM25Retriever(CHUNKS)), "bm25", CHUNKS, cases)
    assert report.overall.hit5 is not None and report.overall.hit5 >= 0.85
    assert report.overall.mrr is not None and report.overall.mrr >= 0.75
    assert report.by_tag["plain"].hit5 == 1.0
    # answerable questions score clearly higher than ones the knowledge base cannot answer
    assert report.answerable_top_score is not None and report.unanswerable_top_score is not None
    assert report.answerable_top_score > 2 * report.unanswerable_top_score


# ------------------------------------------------------------------------------ files and CLI


def test_save_run_writes_the_results_and_an_experiment_row(tmp_path: Path) -> None:
    cases = load_queries(QUERIES, CHUNKS)
    outcomes = run_retrieval_eval(cases, BM25Retriever(CHUNKS))
    report = build_report(outcomes, "bm25", CHUNKS, cases)
    path = save_run(outcomes, report, tmp_path / "runs", note="baseline")
    document = json.loads(path.read_text())
    assert document["note"] == "baseline" and len(document["results"]) == len(cases)
    assert document["report"]["retriever"] == "bm25"
    assert document["results"][0]["want"] == cases[0].relevant
    save_run(outcomes, report, tmp_path / "runs")
    rows = list(csv.DictReader((tmp_path / "runs" / "retrieval_experiments.csv").open()))
    assert len(rows) == 2 and rows[0]["retriever"] == "bm25" and rows[0]["note"] == "baseline"
    assert rows[0]["kb_digest"] == report.kb_digest and rows[0]["hit5"] != ""


def test_the_cli_prints_the_report_and_saves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "runs"
    assert main(["--out", str(out), "--note", "cli"]) == 0
    text = capsys.readouterr().out
    assert "Retriever bm25" in text and "hit@1" in text and "Saved" in text
    assert (out / "retrieval_experiments.csv").exists()


def test_the_cli_can_show_every_question_and_skip_saving(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--show", "--no-save", "--out", str(tmp_path / "runs")]) == 0
    text = capsys.readouterr().out
    assert "want returns#return-window" in text and "Saved" not in text
    assert not (tmp_path / "runs").exists()


def test_the_cli_reports_a_bad_knowledge_base_or_question_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--kb", str(tmp_path)]) == 2
    assert "no *.md files" in capsys.readouterr().err
    assert main(["--cases", str(tmp_path / "missing.yaml")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_chunks_are_hashable_values() -> None:
    c = Chunk("d#x", "d", "T", "X", "text")
    assert {c: 1}[Chunk("d#x", "d", "T", "X", "text")] == 1
