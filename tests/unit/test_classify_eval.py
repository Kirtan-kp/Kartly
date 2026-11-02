"""The classifier eval over the labelled messages, with models that answer from the labels."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from kartly.evals.classify import (
    ClassifierSummary,
    IntentCase,
    format_confusion,
    format_outcome,
    format_summary,
    intents_digest,
    load_intent_cases,
    main,
    run_classifier_eval,
    save_classifier_run,
    summarize_classifier,
)
from kartly.llm.errors import RateLimitError
from kartly.llm.types import ChatResponse
from kartly.routing.classifier import INTENT_MIN_TIER, INTENTS, PROMPTS
from tests.llm_support import (
    FakeOpenAIServer,
    completion_payload,
    make_response,
    write_models_config,
)

DATASET = Path(__file__).resolve().parents[2] / "evals" / "routing" / "intents.yaml"
CASES = load_intent_cases(DATASET)
BY_MESSAGE = {c.message: c for c in CASES}


def label_json(case: IntentCase, **overrides: Any) -> str:
    body = {
        "intent": case.intent,
        "tier": case.tier,
        "complexity": "simple",
        "sentiment": "neutral",
        "confidence": 0.9,
    }
    return json.dumps({**body, **overrides})


class LabelClient:
    """A 'model' that reads the quoted ticket and answers from the dataset's labels."""

    name = "label"

    def __init__(self, answer: Any = None) -> None:
        self.answer = answer or (lambda case: label_json(case))
        self.calls = 0

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls += 1
        text = messages[1]["content"].removeprefix("<ticket>\n").removesuffix("\n</ticket>")
        result = self.answer(BY_MESSAGE[text])
        if isinstance(result, Exception):
            raise result
        return make_response(result, latency_ms=10 * self.calls)


# ------------------------------------------------------------------------------ the data


def test_the_dataset_is_big_complete_and_consistent() -> None:
    assert len(CASES) >= 50
    assert {c.intent for c in CASES} == set(INTENTS) - {"other"}
    for case in CASES:
        # a label must never sit below its intent's floor, or the eval contradicts the design
        assert case.tier >= INTENT_MIN_TIER[case.intent], case.id
    assert sum(1 for c in CASES if c.tier == 3) >= 8
    assert len({c.message for c in CASES}) == len(CASES)


def test_the_digest_follows_the_labels() -> None:
    changed = [*CASES[:-1], CASES[-1].model_copy(update={"tier": 2})]
    assert intents_digest(CASES) == intents_digest(list(CASES))
    assert intents_digest(changed) != intents_digest(CASES)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("x: 1", "expected a list"),
        ("- {id: a, intent: tracking, tier: 0, message: m, extra: 1}", "extra"),
        ("- {id: a, intent: nonsense, tier: 0, message: m}", "unknown intent"),
        ("- {id: a, intent: tracking, tier: 5, message: m}", "tier must be 0 to 3"),
        ("- {id: a, intent: tracking, tier: 0, message: m}\n" * 2, "duplicate"),
        ("- {id: a, intent: tracking, tier: '0', message: m}", "tier"),
        ("{", "cannot read"),
    ],
)
def test_bad_datasets_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "d.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_intent_cases(path)


# ----------------------------------------------------------------------------- the metrics


def evaluate(client: LabelClient, cases: list[IntentCase] | None = None) -> ClassifierSummary:
    chosen = cases or CASES
    return summarize_classifier(run_classifier_eval(chosen, client), chosen, "label")


def test_a_model_that_answers_from_the_labels_scores_perfectly() -> None:
    summary = evaluate(LabelClient())
    assert summary.intent_accuracy == 1.0 and summary.tier_accuracy == 1.0
    assert summary.tier3_recall == summary.tier3_recall_classifier == 1.0
    assert summary.under_routed == [] and summary.over_routed == [] and summary.unusable == 0
    assert summary.tier3_false_alarm_rate == 0.0 and summary.gate_passed
    tracking = sum(1 for c in CASES if c.intent == "tracking")
    assert summary.confusion["tracking"] == {"tracking": tracking}


def test_a_lazy_model_is_caught_by_the_rules_but_misses_the_rest_of_tier_3() -> None:
    lazy = LabelClient(lambda case: label_json(case, intent="order_status", tier=0))
    summary = evaluate(lazy)
    refunds = [c.id for c in CASES if c.intent == "refund"]
    assert summary.tier3_recall_classifier == 0.0
    assert summary.tier3_recall_rules is not None and 0 < summary.tier3_recall_rules < 1
    assert summary.tier3_recall == summary.tier3_recall_rules  # rules rescue what they can
    assert summary.tier3_missed and not summary.gate_passed
    assert set(summary.tier3_missed) <= set(summary.under_routed)
    assert set(refunds) <= set(summary.under_routed)  # money requests went through as tier 0


def test_over_routing_is_counted_separately_from_under_routing() -> None:
    paranoid = LabelClient(lambda case: label_json(case, intent="escalation", tier=3))
    summary = evaluate(paranoid)
    assert summary.tier3_recall == 1.0 and summary.under_routed == []
    assert len(summary.over_routed) == sum(1 for c in CASES if c.tier < 3)
    assert summary.tier3_false_alarm_rate == 1.0 and summary.gate_passed


def test_unusable_output_fails_the_gate_and_is_counted_as_wrong() -> None:
    summary = evaluate(LabelClient(lambda case: "no idea"))
    assert summary.unusable == len(CASES) and summary.intent_accuracy == 0.0
    assert summary.clarification_rate == 1.0 and not summary.gate_passed
    tracking = sum(1 for c in CASES if c.intent == "tracking")
    assert summary.confusion["tracking"] == {"(unusable)": tracking}
    assert summary.tier3_recall_classifier == 0.0
    assert summary.tier3_recall == summary.tier3_recall_rules  # only the rules remained


def test_a_provider_error_is_unusable_output_not_a_crash() -> None:
    err = LabelClient(lambda case: RateLimitError("429", model="m", status=429))
    outcomes = run_classifier_eval(CASES[:2], err)
    assert all(o.result.error == "llm_error: RateLimitError" for o in outcomes)


def test_tokens_cost_and_latency_percentiles() -> None:
    summary = evaluate(LabelClient(), CASES[:10])
    assert summary.avg_tokens == 120 and summary.total_cost_usd > 0
    assert (summary.p50_ms, summary.p95_ms) == (50, 100)


def test_an_empty_run_does_not_divide_by_zero() -> None:
    summary = summarize_classifier([], [], "m")
    assert summary.intent_accuracy is None and summary.avg_tokens == 0.0 and not summary.gate_passed
    assert "n/a" in format_summary(summary)


def test_the_pause_is_between_calls_not_after_the_last() -> None:
    slept: list[float] = []
    run_classifier_eval(CASES[:3], LabelClient(), pause_s=2.5, sleep=slept.append)
    assert slept == [2.5, 2.5]


# ----------------------------------------------------------------------------- the output


def test_the_report_flags_dangerous_misroutes() -> None:
    lazy = LabelClient(lambda case: label_json(case, intent="order_status", tier=0))
    outcomes = run_classifier_eval(CASES, lazy)
    summary = summarize_classifier(outcomes, CASES, "label")
    text = format_summary(summary)
    assert (
        "TIER 3 MISSED" in text and "GATE (every Tier 3 caught, no unusable output): FAILED" in text
    )
    assert any("ROUTED TOO LOW" in format_outcome(o) for o in outcomes)
    assert "refund-1" in format_confusion(summary.confusion) or "refund" in format_confusion(
        summary.confusion
    )


def test_a_perfect_report_has_no_confusion_and_a_passing_gate() -> None:
    outcomes = run_classifier_eval(CASES, LabelClient())
    summary = summarize_classifier(outcomes, CASES, "label")
    assert "(none)" in format_confusion(summary.confusion)
    assert "passed" in format_summary(summary).splitlines()[-1]
    assert format_outcome(outcomes[0]).startswith("[ok  ]")


def test_unusable_outcomes_are_shown_with_their_reason() -> None:
    [outcome] = run_classifier_eval(CASES[:1], LabelClient(lambda c: "nope"))
    assert format_outcome(outcome).startswith("[BAD ]") and "unusable_output" in format_outcome(
        outcome
    )


def test_save_writes_results_and_an_experiment_row(tmp_path: Path) -> None:
    outcomes = run_classifier_eval(CASES, LabelClient())
    summary = summarize_classifier(outcomes, CASES, "a/b model")
    path = save_classifier_run(outcomes, summary, tmp_path / "runs", note="baseline")
    assert path.name.endswith("_routing_a_b_model.json")
    document = json.loads(path.read_text())
    assert document["note"] == "baseline" and len(document["results"]) == len(CASES)
    assert document["results"][0]["want"] == {"intent": "order_status", "tier": 0}
    save_classifier_run(outcomes, summary, tmp_path / "runs")
    rows = list(csv.DictReader((tmp_path / "runs" / "routing_experiments.csv").open()))
    assert len(rows) == 2 and rows[0]["model"] == "a/b model"
    assert rows[0]["tier3_recall"] == "1.000" and rows[0]["classifier_version"] == "c2"
    assert rows[0]["note"] == "baseline" and rows[0]["results_file"] == path.name


# ------------------------------------------------------------------------------------ CLI


def answer_from_labels(body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
    text = body["messages"][1]["content"].removeprefix("<ticket>\n").removesuffix("\n</ticket>")
    return 200, completion_payload(label_json(BY_MESSAGE[text])), {}


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kartly.evals.classify.load_env", lambda: None)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def test_the_cli_runs_over_http_prints_and_saves(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = answer_from_labels
    config = write_models_config(tmp_path, fake_server.base_url)
    out = tmp_path / "runs"
    code = main(["--model", "fake_small", "--config", str(config), "--out", str(out)])
    text = capsys.readouterr().out
    assert code == 0 and "intent accuracy 100.0%" in text and "Saved" in text
    assert len(fake_server.requests) == len(CASES)
    assert fake_server.requests[0]["body"]["response_format"] == {"type": "json_object"}
    assert (out / "routing_experiments.csv").exists()


def test_the_cli_can_run_an_older_prompt_on_the_same_cases(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = answer_from_labels
    config = write_models_config(tmp_path, fake_server.base_url)
    args = ["--model", "fake_small", "--config", str(config), "--out", str(tmp_path / "runs")]
    assert main([*args, "--classifier-version", "c1", "--only", "tracking-1"]) == 0
    assert main([*args, "--only", "tracking-1"]) == 0
    first, second = (r["body"]["messages"][0]["content"] for r in fake_server.requests)
    assert first == PROMPTS["c1"] and second == PROMPTS["c2"] and first != second
    rows = list(csv.DictReader((tmp_path / "runs" / "routing_experiments.csv").open()))
    assert [r["classifier_version"] for r in rows] == ["c1", "c2"]
    assert rows[0]["cases_digest"] == rows[1]["cases_digest"]  # comparable on purpose
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main([*args, "--classifier-version", "c9"])


def test_the_cli_exits_one_when_the_gate_fails(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = lambda body: (200, completion_payload("garbage"), {})
    config = write_models_config(tmp_path, fake_server.base_url)
    args = ["--model", "fake_small", "--config", str(config), "--no-save", "--only", "tracking-1"]
    assert main(args) == 1
    assert "[BAD ]" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--only", "nope"], "unknown case id"),
        (["--pause", "-1"], "--pause cannot be negative"),
        (["--cases", "missing.yaml"], "cannot read"),
        (["--model", "nope"], "nope"),
    ],
)
def test_bad_options_exit_two(
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    args = ["--model", "fake_small", "--config", str(config), *extra]
    if "--model" in extra:
        args = ["--config", str(config), *extra]
    assert main(args) == 2
    assert message in capsys.readouterr().err
    assert fake_server.requests == []


def test_a_model_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "--model is required" in capsys.readouterr().err
