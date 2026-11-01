"""The Tier 3 rules and their eval: what they catch, what they leave alone, how measured."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kartly.evals.routing import (
    RoutingCase,
    RoutingReport,
    evaluate_rules,
    format_report,
    load_routing_cases,
    main,
)
from kartly.routing import RuleVerdict, route_by_rules

DATASET = Path(__file__).resolve().parents[2] / "evals" / "routing" / "tier3.yaml"


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("I will speak to my lawyer", "legal"),
        ("taking this to consumer court", "legal"),
        ("I did not place this order, it is fraud", "fraud"),
        ("I never authorised that payment", "fraud"),
        ("starting a chargeback", "fraud"),
        ("my account was hacked", "account_security"),
        ("please delete my account", "account_security"),
        ("somebody got into my account", "account_security"),
        ("the heater caught fire", "safety"),
        ("I got an electric shock", "safety"),
        ("I want to die", "distress"),
        ("I will kill you", "threat"),
        ("I will kill your delivery boy", "threat"),
    ],
)
def test_each_category_is_caught(text: str, category: str) -> None:
    verdict = route_by_rules(text)
    assert verdict.tier == 3 and category in verdict.categories


@pytest.mark.parametrize(
    "text",
    [
        "Where is my order KRT-100002?",
        "I never received the invoice",
        "There was a rash of late deliveries",
        "The legal size notebook is too small",
        "Is this bottle safe for hot drinks?",
        "I did not get the confirmation email",
        "",
    ],
)
def test_ordinary_messages_do_not_fire(text: str) -> None:
    assert route_by_rules(text) == RuleVerdict(0, ())


def test_case_width_and_zero_width_characters_do_not_hide_a_threat() -> None:
    assert route_by_rules("I WILL SUE YOU").tier == 3
    assert route_by_rules("ｌａｗｙｅｒ").tier == 3  # full-width letters
    assert route_by_rules("law​yer").tier == 3


def test_several_categories_come_back_in_a_stable_order() -> None:
    verdict = route_by_rules("My lawyer will deal with this fraud, the charger caught fire")
    assert verdict.categories == ("legal", "fraud", "safety")


@given(st.text(max_size=300))
def test_any_text_is_routed_without_error(text: str) -> None:
    verdict = route_by_rules(text)
    assert verdict.tier == (3 if verdict.categories else 0)


# ------------------------------------------------------------------------------- the eval


def test_the_dataset_loads_and_has_enough_of_everything() -> None:
    cases = load_routing_cases(DATASET)
    positives = [c for c in cases if c.category and not c.known]
    ordinary = [c for c in cases if c.category is None and not c.known]
    assert len(positives) >= 50 and len(ordinary) >= 35
    assert {c.category for c in positives} == {
        "legal",
        "fraud",
        "account_security",
        "safety",
        "distress",
        "threat",
    }
    assert any(c.known == "miss" for c in cases) and any(c.known == "false_alarm" for c in cases)


def test_the_rules_pass_their_gate_on_the_dataset() -> None:
    report = evaluate_rules(load_routing_cases(DATASET))
    assert report.gate_passed, format_report(report)
    assert report.recall == 1.0 and report.false_alarm_rate == 0.0


def test_the_known_gaps_are_still_gaps_so_a_fix_shows_up_here() -> None:
    """If a rule change fixes one, move it into the gated set (and this count goes down)."""
    report = evaluate_rules(load_routing_cases(DATASET))
    assert report.known_misses_still_missed == report.known_misses_total > 0
    assert report.known_false_alarms_still_raised == report.known_false_alarms_total > 0


def case(id_: str, category: str | None, message: str = "m", known: str | None = None) -> Any:
    return RoutingCase.model_validate(
        {"id": id_, "category": category, "message": message, "known": known}
    )


def fire(*categories: str) -> Any:
    return lambda _text: RuleVerdict(3 if categories else 0, categories)


def test_a_missed_message_and_a_false_alarm_each_fail_the_gate() -> None:
    missed = evaluate_rules([case("a", "legal"), case("b", None)], fire())
    assert missed.missed == ["a"] and missed.recall == 0.0 and not missed.gate_passed
    flagged = evaluate_rules([case("a", "legal"), case("b", None)], fire("legal"))
    assert flagged.flagged == ["b"] and flagged.false_alarm_rate == 1.0 and not flagged.gate_passed


def test_a_message_caught_under_another_category_is_noted_but_counts_as_caught() -> None:
    report = evaluate_rules([case("a", "legal"), case("b", None)], lambda t: fire("fraud")(t))
    assert report.caught == 1 and report.wrong_category == ["a"]


def test_known_cases_never_count_toward_the_gate() -> None:
    cases = [case("a", "legal"), case("b", None), case("k1", "legal", known="miss")]
    cases.append(case("k2", None, known="false_alarm"))
    report = evaluate_rules(cases, lambda t: RuleVerdict(3, ("legal",)) if t == "x" else fire()(t))
    assert report.known_misses_total == 1 and report.known_false_alarms_total == 1
    assert report.positives == 1 and report.ordinary == 1


def test_a_known_gap_that_is_fixed_stops_counting_as_a_gap() -> None:
    cases = [case("k1", "legal", known="miss"), case("k2", None, known="false_alarm")]
    report = evaluate_rules(cases, fire("legal"))
    assert (report.known_misses_still_missed, report.known_false_alarms_still_raised) == (0, 1)
    report = evaluate_rules(cases, fire())
    assert (report.known_misses_still_missed, report.known_false_alarms_still_raised) == (1, 0)


def test_an_empty_dataset_cannot_pass() -> None:
    report = evaluate_rules([])
    assert report.recall is None and report.false_alarm_rate is None and not report.gate_passed
    assert "n/a" in format_report(report)
    assert isinstance(report, RoutingReport)


def test_the_report_names_what_failed() -> None:
    report = evaluate_rules([case("a", "legal"), case("b", None)], fire("fraud"))
    text = format_report(report)
    assert "FALSE ALARMS (gate): b" in text and "Caught under another category: a" in text
    assert "GATE: FAILED" in text


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("not a list", "expected a list"),
        (
            "- {id: a, category: null, message: m}\n- {id: a, category: null, message: m}\n",
            "duplicate",
        ),
        ("- {id: a, category: null, message: m, known: miss}\n", "known miss needs a category"),
        ("- {id: a, category: legal, message: m, known: false_alarm}\n", "ordinary message"),
        ("- {id: a, category: null, message: m, extra: 1}\n", "extra"),
        ("- {id: 1, category: null, message: m}\n", "id"),
        ("{", "cannot read"),
    ],
)
def test_bad_datasets_are_rejected_with_a_reason(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_routing_cases(path)


def test_the_cli_prints_the_report_and_exits_zero_on_a_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--cases", str(DATASET)]) == 0
    out = capsys.readouterr().out
    assert "Tier 3 recall: 51/51" in out and "GATE: passed" in out


def test_the_cli_shows_each_verdict_on_request(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--cases", str(DATASET), "--show"]) == 0
    out = capsys.readouterr().out
    assert "T3 " in out and "-- " in out and "Where is my order KRT-100002?" in out


def test_the_cli_exits_one_when_the_gate_fails_and_two_for_a_bad_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "r.yaml"
    path.write_text(
        "- {id: a, category: legal, message: nothing here}\n"
        "- {id: b, category: null, message: hello}\n",
        encoding="utf-8",
    )
    assert main(["--cases", str(path)]) == 1
    assert "MISSED (gate): a" in capsys.readouterr().out
    assert main(["--cases", str(tmp_path / "missing.yaml")]) == 2
    assert "cannot read" in capsys.readouterr().err
