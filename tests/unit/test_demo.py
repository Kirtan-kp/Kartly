"""The demo tells the truth: what it narrates is what the real stack did."""

from __future__ import annotations

import json

import pytest

from kartly.demo import Demo, Scene, main, render


@pytest.fixture(scope="module")
def scenes() -> list[Scene]:
    return Demo().run()


def test_the_six_scenes_end_as_designed(scenes: list[Scene]) -> None:
    assert [s.outcome for s in scenes] == [
        "drafted",
        "escalated",
        "escalated",
        "drafted",
        "awaiting_approval",
        "escalated",
    ]


@pytest.mark.parametrize("index", [1, 2, 5])
def test_screened_routed_and_switched_off_tickets_never_reach_the_model(
    scenes: list[Scene], index: int
) -> None:
    assert scenes[index].model_calls == 0 and scenes[index].tool_calls == 0


def test_the_small_refund_is_paid_once_and_the_reply_says_so(scenes: list[Scene]) -> None:
    refund = scenes[3]
    assert refund.refunds_paid == 1
    assert refund.reply is not None and "has been issued" in refund.reply
    assert "refunded on the order now: INR 599.00" in refund.notes


def test_the_large_refund_waits_for_a_human_then_runs_exactly_once(scenes: list[Scene]) -> None:
    large = scenes[4]
    assert "refunded on the order so far: INR 0.00" in large.notes
    assert "human:maya approves: executed" in large.notes
    assert "refunded on the order after approval: INR 2,898.00" in large.notes
    assert any("already_decided" in n for n in large.notes)
    assert large.refunds_paid == 2  # the small one and this one, never a third
    assert large.reply is not None and "pending approval" in large.reply


def test_the_reply_never_claims_money_moved_before_it_did(scenes: list[Scene]) -> None:
    large = scenes[4]
    assert large.reply is not None
    assert "has been issued" not in large.reply and "refunded" not in large.reply.lower()


def test_every_scene_leaves_audit_rows_that_only_grow(scenes: list[Scene]) -> None:
    rows = [s.audit_rows for s in scenes]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)


def test_the_kill_switch_scene_names_who_stopped_the_agent(scenes: list[Scene]) -> None:
    assert any("switched off by maya" in n for n in scenes[5].notes)


def test_two_runs_give_the_same_story() -> None:
    first = [(s.outcome, s.reply, s.model_calls, s.refunds_paid) for s in Demo().run()]
    second = [(s.outcome, s.reply, s.model_calls, s.refunds_paid) for s in Demo().run()]
    assert first == second


def test_the_text_and_json_forms_work(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    text = capsys.readouterr().out
    assert "6. The kill switch" in text and "why it matters" in text
    assert main(["--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6
    assert render(Demo().run()).startswith("Kartly demo")
