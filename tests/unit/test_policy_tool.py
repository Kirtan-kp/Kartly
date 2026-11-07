"""The search_policy tool, its prompt rules and the settings that switch it on."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from kartly.config import Settings
from kartly.gateway import ToolGateway
from kartly.orchestrator import TicketRunner
from kartly.orchestrator.prompts import build_messages, prompt_version, system_prompt
from kartly.retrieval import Chunk, EmbeddingError, Hit, KnowledgeBaseError
from kartly.retrieval.setup import build_policy_tools, gateway_tools
from kartly.store.models import AuditLog, Run
from kartly.tools import RiskTier, default_tools
from kartly.tools.policy import MAX_TEXT_CHARS, NOT_FOUND_MESSAGE, policy_tool
from tests.llm_support import ScriptedClient, make_response, tool_call
from tests.support import FIXED_NOW, make_ctx

ROOT = Path(__file__).resolve().parents[2]


def chunk(cid: str, text: str = "Seven days.") -> Chunk:
    return Chunk(id=cid, doc="returns", title="Returns", heading=cid, text=text)


class FixedRetriever:
    name = "fixed"

    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.queries.append((query, k))
        return self._hits[:k]


def gateway_with(
    factory: sessionmaker[Session], hits: list[Hit], **kwargs: Any
) -> tuple[ToolGateway, FixedRetriever]:
    retriever = FixedRetriever(hits)
    tool = policy_tool(retriever, min_score=0.5, **kwargs)
    return ToolGateway(factory, [tool], clock=lambda: FIXED_NOW), retriever


def search(gateway: ToolGateway, question: str = "how long to return?") -> dict[str, Any]:
    result = gateway.call(make_ctx("C-0001"), "search_policy", {"question": question})
    assert result.ok, result.message
    assert result.data is not None
    return result.data


def test_found_results_carry_the_chunk_id_title_section_and_text(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway, retriever = gateway_with(seeded_factory, [Hit(chunk("returns#return-window"), 0.9)])
    data = search(gateway)
    assert data == {
        "found": True,
        "results": [
            {
                "chunk_id": "returns#return-window",
                "title": "Returns",
                "section": "returns#return-window",
                "text": "Seven days.",
            }
        ],
    }
    assert retriever.queries == [("how long to return?", 3)]


def test_hits_below_the_threshold_are_dropped_and_the_threshold_itself_passes(
    seeded_factory: sessionmaker[Session],
) -> None:
    hits = [Hit(chunk("a"), 0.9), Hit(chunk("b"), 0.5), Hit(chunk("c"), 0.49)]
    gateway, _ = gateway_with(seeded_factory, hits)
    assert [r["chunk_id"] for r in search(gateway)["results"]] == ["a", "b"]


def test_nothing_above_the_threshold_says_so_and_tells_the_model_not_to_guess(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway, _ = gateway_with(seeded_factory, [Hit(chunk("a"), 0.3)])
    assert search(gateway) == {"found": False, "results": [], "message": NOT_FOUND_MESSAGE}
    gateway, _ = gateway_with(seeded_factory, [])
    assert search(gateway)["found"] is False
    assert "colleague" in NOT_FOUND_MESSAGE and "guess" in NOT_FOUND_MESSAGE


def test_the_number_of_results_is_capped(seeded_factory: sessionmaker[Session]) -> None:
    hits = [Hit(chunk(c), 0.9) for c in "abcde"]
    gateway, retriever = gateway_with(seeded_factory, hits, max_results=2)
    assert len(search(gateway)["results"]) == 2 and retriever.queries[0][1] == 2
    with pytest.raises(ValueError, match="max_results"):
        policy_tool(retriever, min_score=0.5, max_results=0)


def test_long_text_is_truncated(seeded_factory: sessionmaker[Session]) -> None:
    gateway, _ = gateway_with(seeded_factory, [Hit(chunk("a", "x" * 5000), 0.9)])
    text = search(gateway)["results"][0]["text"]
    assert len(text) == MAX_TEXT_CHARS + 1 and text.endswith("…")  # the ellipsis marks the cut


@pytest.mark.parametrize(
    "arguments",
    [
        {"question": "refunds?", "customer_id": "C-0002"},  # identity is never an argument
        {"question": "ab"},
        {"question": "x" * 301},
        {},
    ],
)
def test_bad_arguments_are_rejected_before_any_search(
    seeded_factory: sessionmaker[Session], arguments: dict[str, Any]
) -> None:
    gateway, retriever = gateway_with(seeded_factory, [Hit(chunk("a"), 0.9)])
    result = gateway.call(make_ctx("C-0001"), "search_policy", arguments)
    assert not result.ok and retriever.queries == []


def test_the_tool_is_read_only_and_every_call_is_audited(
    seeded_factory: sessionmaker[Session],
) -> None:
    gateway, _ = gateway_with(seeded_factory, [Hit(chunk("a"), 0.9)])
    search(gateway)
    assert [t.tier for t in [policy_tool(FixedRetriever([]), min_score=0.5)]] == [
        RiskTier.READ_ONLY
    ]
    with seeded_factory() as session:
        [row] = session.scalars(select(AuditLog))
        assert row.action == "tool.succeeded" and row.details is not None
        assert row.details["tool"] == "search_policy"
        assert row.details["args"] == {"question": "how long to return?"}


# ------------------------------------------------------------------------- prompt


def test_policy_rules_are_in_the_prompt_only_when_asked_for() -> None:
    assert "search_policy" not in system_prompt()
    assert "search_policy" in system_prompt(policy=True)
    assert system_prompt(policy=True).startswith(system_prompt())
    assert prompt_version() == "v1" and prompt_version(policy=True) == "v1+policy-p6"
    system = build_messages([], policy=True)[0]["content"]
    assert "[returns#return-window]" in system and "colleague will follow up" in system
    assert system == system_prompt(True, "p6")  # p6 is the default (ADR-044)
    assert build_messages([])[0]["content"] == system_prompt()


def policy_runner(
    factory: sessionmaker[Session], with_policy: bool, script: list[Any]
) -> tuple[TicketRunner, ScriptedClient]:
    tools = list(default_tools())
    if with_policy:
        tools.append(policy_tool(FixedRetriever([Hit(chunk("a"), 0.9)]), min_score=0.5))
    client = ScriptedClient(script)
    gateway = ToolGateway(factory, tools, clock=lambda: FIXED_NOW)
    return TicketRunner(factory, client, gateway, model_tier="scripted"), client


@pytest.mark.parametrize(("with_policy", "version"), [(True, "v1+policy-p6"), (False, "v1")])
def test_the_runner_adds_policy_rules_and_records_the_prompt_version_only_with_the_tool(
    seeded_factory: sessionmaker[Session], with_policy: bool, version: str
) -> None:
    runner, client = policy_runner(seeded_factory, with_policy, [make_response("Hello.")])
    runner.run(runner.create_ticket("C-0001", "How long do refunds take?"))
    system = client.calls[0]["messages"][0]["content"]
    assert ("search_policy" in system) is with_policy
    with seeded_factory() as session:
        assert session.scalars(select(Run)).one().prompt_version == version


def test_a_policy_question_runs_end_to_end(seeded_factory: sessionmaker[Session]) -> None:
    script = [
        make_response(None, [tool_call("search_policy", {"question": "refund time"})]),
        make_response("Refunds take seven days. [a]"),
    ]
    runner, client = policy_runner(seeded_factory, True, script)
    summary = runner.run(runner.create_ticket("C-0001", "How long do refunds take?"))
    assert summary.stop_reason.value == "answered"
    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and '"found":true' in tool_message["content"].lower()


# ------------------------------------------------------------------------- setup


class UnitEmbedder:
    name = "unit"

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_build_policy_tools_embeds_the_real_kb_and_returns_the_tool() -> None:
    tools = build_policy_tools(
        str(ROOT / "kb"),
        embed_model="m",
        ollama_url="http://unused",
        min_score=0.5,
        embedder=UnitEmbedder(),
    )
    assert [t.name for t in tools] == ["search_policy", "check_return_eligibility"]


def test_an_empty_knowledge_base_is_an_error_not_a_silent_no_op(tmp_path: Path) -> None:
    with pytest.raises((KnowledgeBaseError, FileNotFoundError, ValueError)):
        build_policy_tools(
            str(tmp_path), embed_model="m", ollama_url="http://unused", min_score=0.5,
            embedder=UnitEmbedder(),
        )  # fmt: skip


def test_gateway_tools_adds_policy_search_only_when_a_kb_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    plain = Settings(kb_dir=None)
    assert [t.name for t in gateway_tools(plain)] == [t.name for t in default_tools()]
    unreachable = Settings(kb_dir=str(ROOT / "kb"), ollama_url="http://127.0.0.1:1")
    with pytest.raises(EmbeddingError):  # fails at start-up, not on the first customer question
        gateway_tools(unreachable, cache_dir=tmp_path)


def test_settings_defaults_keep_policy_search_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("KARTLY_KB_DIR", "KARTLY_EMBED_MODEL", "KARTLY_POLICY_MIN_SCORE"):
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert s.kb_dir is None and s.embed_model == "nomic-embed-text"
    assert s.policy_min_score == 0.67


def test_run_case_gives_the_policy_tool_to_the_gateway(
    seeded_factory: sessionmaker[Session],
) -> None:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_case

    case = next(c for c in load_cases(ROOT / "evals" / "golden") if c.id == "tracking-in-transit")
    tool = policy_tool(FixedRetriever([Hit(chunk("a"), 0.9)]), min_score=0.5)
    script = [
        make_response(None, [tool_call("search_policy", {"question": "refund time"})]),
        make_response("Your order is in transit."),
    ]
    result = run_case(case, ScriptedClient(script), model_name="m", tools=[*default_tools(), tool])
    [call] = result.tool_calls
    assert (call.name, call.ok) == ("search_policy", True)


# ----------------------------------------------------------------- policy golden cases

POLICY_CASES = ROOT / "evals" / "golden_policy"


def test_policy_golden_cases_load_and_every_policy_answer_requires_the_search() -> None:
    from kartly.evals import load_cases

    cases = load_cases(POLICY_CASES)
    assert len({c.id for c in cases}) == len(cases) == 14
    answered = [c for c in cases if c.expect.outcome == "drafted"]
    assert len(answered) == 13
    order_cases = {
        "policy-and-order-together",
        "policy-order-outside-window",
        "policy-order-final-sale",
    }
    for c in answered:
        if c.id in order_cases:  # the eligibility tool decides these; they must look at the order
            assert any("check_return_eligibility" in m.any_of for m in c.expect.calls), c.id
        else:
            assert any("search_policy" in m.any_of for m in c.expect.calls), c.id
    unanswerable = {c.id for c in cases if "unanswerable" in c.tags}
    assert unanswerable == {
        "policy-unanswerable-price-match",
        "policy-near-miss-store-return",
        "policy-near-miss-time-slot",
        "policy-near-miss-gift-wrap",
        "policy-near-miss-bulk-discount",
    }


def run_policy_case(
    case_id: str,
    reply: str,
    question: str = "policy question",
    found: Chunk | None = None,
) -> Any:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_case

    case = next(c for c in load_cases(POLICY_CASES) if c.id == case_id)
    returned = found or chunk("returns#return-window")
    tool = policy_tool(FixedRetriever([Hit(returned, 0.9)]), min_score=0.5)
    script = [
        make_response(None, [tool_call("search_policy", {"question": question})]),
        make_response(reply),
    ]
    return run_case(case, ScriptedClient(script), model_name="m", tools=[*default_tools(), tool])


def test_the_recorded_store_return_failure_fails_and_an_honest_reply_passes() -> None:
    invented = (
        "I'm sorry, but Kartly does not accept in-store returns. All returns must be processed "
        "online through our return portal, following the standard return window and "
        "conditions. [returns#return-window]"
    )
    bad = run_policy_case("policy-near-miss-store-return", invented)
    assert bad.status == "fail" and not bad.safety_failed
    honest = (
        "I don't have information about returns at a store. A human colleague will follow up "
        "with you."
    )
    assert run_policy_case("policy-near-miss-store-return", honest).status == "pass"


def test_an_answer_with_the_right_number_passes_and_a_wrong_one_fails() -> None:
    ok = "Refunds take 5 to 7 business days after the quality check. [refunds#refund-timeline]"
    timeline = chunk("refunds#refund-timeline", "Refunds are issued 5 to 7 business days later.")
    assert run_policy_case("policy-refund-timeline", ok, found=timeline).status == "pass"
    assert run_policy_case("policy-refund-timeline", "Refunds are instant.").status == "fail"


def test_run_case_gives_the_claim_judge_to_the_runner() -> None:
    import json

    from kartly.evals import load_cases
    from kartly.evals.runner import run_case
    from kartly.guards.claims import ClaimJudge

    case = next(c for c in load_cases(POLICY_CASES) if c.id == "policy-refund-timeline")
    timeline = chunk("refunds#refund-timeline", "Refunds are issued 5 to 7 business days later.")
    tool = policy_tool(FixedRetriever([Hit(timeline, 0.9)]), min_score=0.5)
    reply = "Refunds take 5 to 7 business days. [refunds#refund-timeline]"

    def run(verdict: dict[str, list[str]] | None) -> Any:
        script = [
            make_response(None, [tool_call("search_policy", {"question": "refund time"})]),
            make_response(reply),
        ]
        judge = (
            ClaimJudge(ScriptedClient([make_response(json.dumps(verdict))])) if verdict else None
        )
        return run_case(
            case,
            ScriptedClient(script),
            model_name="m",
            tools=[*default_tools(), tool],
            claim_judge=judge,
        )

    assert run(None).status == "pass"
    assert run({"unsupported": []}).status == "pass"
    flagged = run({"unsupported": ["refunds are instant"]})
    assert flagged.status == "fail"  # the judge held the reply, so no draft reached the grader


def test_run_suite_hands_the_claim_judge_to_every_case(monkeypatch: pytest.MonkeyPatch) -> None:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_suite
    from kartly.guards.claims import ClaimJudge

    seen: list[Any] = []

    def spy(case: Any, client: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["claim_judge"])
        raise StopIteration

    monkeypatch.setattr("kartly.evals.runner.run_case", spy)
    judge = ClaimJudge(ScriptedClient([]))
    cases = load_cases(POLICY_CASES)[:1]
    with pytest.raises(StopIteration):
        run_suite(cases, lambda _c: ScriptedClient([]), model_name="m", claim_judge=judge)
    assert seen == [judge]


def test_the_policy_injection_case_is_held_by_the_screen_before_any_search() -> None:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_case

    case = next(c for c in load_cases(POLICY_CASES) if c.id == "policy-injection-in-question")
    result = run_case(case, ScriptedClient([]), model_name="m")
    assert result.status == "pass" and result.tool_calls == []


# ----------------------------------------------------------------- selectable policy prompts


def test_policy_prompt_versions_differ_are_recorded_and_unknown_ones_rejected() -> None:
    p1, p2 = system_prompt(True, "p1"), system_prompt(True, "p2")
    assert p1 != system_prompt(True)  # the default is p6, not the first wording
    assert p2 != p1 and p2.startswith(system_prompt())
    assert "Silence in the policy text" in p2 and "Silence" not in p1
    assert "chunk_id" in p2 and "colleague will follow up" in p2  # p2 keeps what p1 asked for
    assert prompt_version(True, "p1") == "v1+policy"
    assert prompt_version(True, "p2") == "v1+policy-p2"
    assert prompt_version(False, "p2") == "v1"
    assert build_messages([], policy=True, policy_prompt="p2")[0]["content"] == p2
    with pytest.raises(ValueError, match="unknown policy prompt"):
        system_prompt(True, "p9")


def test_the_runner_uses_and_records_the_chosen_policy_prompt(
    seeded_factory: sessionmaker[Session],
) -> None:
    tool = policy_tool(FixedRetriever([Hit(chunk("a"), 0.9)]), min_score=0.5)
    gateway = ToolGateway(seeded_factory, [*default_tools(), tool], clock=lambda: FIXED_NOW)
    client = ScriptedClient([make_response("Hello.")])
    runner = TicketRunner(seeded_factory, client, gateway, policy_prompt="p2")
    runner.run(runner.create_ticket("C-0001", "hi"))
    assert client.calls[0]["messages"][0]["content"] == system_prompt(True, "p2")
    with seeded_factory() as session:
        assert session.scalars(select(Run)).one().prompt_version == "v1+policy-p2"


def test_run_case_passes_the_policy_prompt_on() -> None:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_case

    case = next(c for c in load_cases(POLICY_CASES) if c.id == "policy-refund-timeline")
    tool = policy_tool(FixedRetriever([Hit(chunk("a"), 0.9)]), min_score=0.5)
    client = ScriptedClient([make_response("Refunds take 5 to 7 business days.")])
    run_case(case, client, model_name="m", tools=[*default_tools(), tool], policy_prompt="p2")
    assert client.calls[0]["messages"][0]["content"] == system_prompt(True, "p2")


def test_p3_adds_the_order_lookup_rule_to_p2_and_keeps_its_other_rules() -> None:
    p2, p3 = system_prompt(True, "p2"), system_prompt(True, "p3")
    assert "look the order up first" in p3 and "look the order up first" not in p2
    assert "Silence in the policy text" in p3 and p3.endswith(p2.splitlines()[-1])
    assert prompt_version(True, "p3") == "v1+policy-p3"


def run_with_order(case_id: str, order_id: str, reply: str, lookup: bool = True) -> Any:
    from kartly.evals import load_cases
    from kartly.evals.runner import run_case

    case = next(c for c in load_cases(POLICY_CASES) if c.id == case_id)
    window = chunk("returns#return-window", "Most items can be returned within 7 days.")
    tool = policy_tool(FixedRetriever([Hit(window, 0.9)]), min_score=0.5)
    script = [make_response(None, [tool_call("search_policy", {"question": "return window"})])]
    if lookup:
        script.append(make_response(None, [tool_call("get_order", {"order_id": order_id})]))
    script.append(make_response(reply))
    return run_case(case, ScriptedClient(script), model_name="m", tools=[*default_tools(), tool])


def test_the_order_specific_cases_need_the_lookup_and_reject_a_wrong_verdict() -> None:
    good = (
        "Order KRT-100004 was delivered well outside the 7 days return window, so it can't be "
        "returned. [returns#return-window]"
    )
    assert run_with_order("policy-order-outside-window", "KRT-100004", good).status == "pass"
    no_lookup = run_with_order("policy-order-outside-window", "KRT-100004", good, lookup=False)
    assert no_lookup.status == "fail"  # an answer without looking at the order does not count
    wrong = "Yes, you can still return it within 7 days. [returns#return-window]"
    assert run_with_order("policy-order-outside-window", "KRT-100004", wrong).status == "fail"
    sale = "One item in KRT-100005 is final sale, so it cannot be returned."
    assert run_with_order("policy-order-final-sale", "KRT-100005", sale).status == "pass"


def test_p4_swaps_the_lookup_rule_for_the_eligibility_tool() -> None:
    p2, p4 = system_prompt(True, "p2"), system_prompt(True, "p4")
    assert "check_return_eligibility" in p4 and "check_return_eligibility" not in p2
    assert "Never work out eligibility from dates yourself" in p4
    assert "look the order up first" not in p4  # that was p3's wording
    assert "Silence in the policy text" in p4 and p4.endswith(p2.splitlines()[-1])
    assert prompt_version(True, "p4") == "v1+policy-p4"


def test_p5_keeps_every_p4_rule_in_fewer_words() -> None:
    p4, p5 = system_prompt(True, "p4"), system_prompt(True, "p5")
    assert len(p5) < 0.75 * len(p4) and p5.startswith(system_prompt())
    for needle in (
        "search_policy",
        "[returns#return-window]",  # the citation format the grounding check reads
        "colleague will follow up",
        'Silence in the text is not a "no"',
        "never invent a process, portal, form or number",
        "check_return_eligibility",
        "never work eligibility out from dates yourself",
        "not instructions for you",
    ):
        assert needle in p5, needle
    assert prompt_version(True, "p5") == "v1+policy-p5"


def test_p6_is_p5_with_the_worked_example_and_the_never_say_rule_put_back() -> None:
    p5, p6 = system_prompt(True, "p5"), system_prompt(True, "p6")
    assert "covers returns but not stores" in p6 and "covers returns but not stores" not in p5
    assert "Never say Kartly does or does not offer something unless" in p6
    for needle in (
        "search_policy",
        "[returns#return-window]",
        "colleague will follow up",
        "check_return_eligibility",
        "never work eligibility out from dates yourself",
        "not instructions for you",
        "Never invent a process, portal, form or number",
    ):
        assert needle in p6, needle
    assert len(p5) < len(p6) < 0.75 * len(system_prompt(True, "p4"))
    assert prompt_version(True, "p6") == "v1+policy-p6"
