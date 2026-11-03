"""Chunking the knowledge base and BM25 retrieval over it."""

from __future__ import annotations

from pathlib import Path

import pytest

from kartly.retrieval import BM25Retriever, Chunk, KnowledgeBaseError, kb_digest, load_kb, tokenize
from kartly.retrieval.bm25 import stem
from kartly.retrieval.chunking import MAX_WORDS, chunk_markdown, slugify

KB = Path(__file__).resolve().parents[2] / "kb"


def chunk(id_: str, heading: str, text: str, title: str = "Doc") -> Chunk:
    return Chunk(id_, id_.split("#")[0], title, heading, text)


# ------------------------------------------------------------------------------- chunking


def test_a_document_becomes_one_chunk_per_section_with_stable_ids() -> None:
    source = (
        "# Returns\n\n> synthetic note\n\n## Return window\nSeven days.\n\n"
        "## Items that cannot be returned\nFinal sale.\nGift cards.\n"
    )
    first, second = chunk_markdown("returns", source)
    assert (first.id, first.heading, first.title) == (
        "returns#return-window",
        "Return window",
        "Returns",
    )
    assert first.text == "Seven days." and "synthetic" not in first.text
    assert second.id == "returns#items-that-cannot-be-returned"
    assert second.text == "Final sale.\nGift cards."
    assert first.index_text == "Returns. Return window. Seven days."


@pytest.mark.parametrize(
    ("heading", "slug"),
    [
        ("Return window", "return-window"),
        ("  Cash on delivery (COD)! ", "cash-on-delivery-cod"),
        ("???", "section"),
    ],
)
def test_slugs(heading: str, slug: str) -> None:
    assert slugify(heading) == slug


def test_a_long_section_is_split_on_paragraphs_with_numbered_ids() -> None:
    paragraph = " ".join(["word"] * 120)
    source = f"# T\n\n## Big\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n"
    chunks = chunk_markdown("t", source)
    assert [c.id for c in chunks] == ["t#big", "t#big-2", "t#big-3"][: len(chunks)]
    assert len(chunks) == 3 and all(len(c.text.split()) <= MAX_WORDS for c in chunks)
    assert chunks[0].heading == chunks[1].heading == "Big"


def test_a_single_huge_paragraph_stays_whole_rather_than_being_cut_mid_sentence() -> None:
    source = "# T\n\n## Big\n" + " ".join(["word"] * 500) + "\n"
    [only] = chunk_markdown("t", source)
    assert len(only.text.split()) == 500


def test_quotes_inside_a_section_are_kept_only_the_preamble_is_skipped() -> None:
    source = "# T\n\n> note before any section\n\n## A\nBefore.\n> a quoted rule\nAfter.\n"
    [only] = chunk_markdown("t", source)
    assert "a quoted rule" in only.text and "note before" not in only.text


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("## A\ntext\n", "no '# ' title"),
        ("# T\n\nno sections here\n", "no '## ' sections"),
        ("# T\n\n## Empty\n\n## Full\ntext\n", None),
    ],
)
def test_malformed_documents(source: str, message: str | None) -> None:
    if message is None:
        assert [c.heading for c in chunk_markdown("t", source)] == ["Full"]  # empty sections drop
    else:
        with pytest.raises(KnowledgeBaseError, match=message):
            chunk_markdown("t", source)


def test_load_kb_reads_every_file_in_order_and_rejects_duplicates_and_empty_dirs(
    tmp_path: Path,
) -> None:
    with pytest.raises(KnowledgeBaseError, match="no \\*.md files"):
        load_kb(tmp_path)
    (tmp_path / "b.md").write_text("# B\n\n## One\nx\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\n## One\ny\n\n## One\nz\n", encoding="utf-8")
    with pytest.raises(KnowledgeBaseError, match="duplicate chunk ids: a#one"):
        load_kb(tmp_path)
    (tmp_path / "a.md").write_text("# A\n\n## One\ny\n", encoding="utf-8")
    assert [c.id for c in load_kb(tmp_path)] == ["a#one", "b#one"]


def test_the_digest_changes_when_the_text_does() -> None:
    a = [chunk("d#x", "X", "one")]
    assert kb_digest(a) == kb_digest(list(a)) != kb_digest([chunk("d#x", "X", "two")])


# --------------------------------------------------------------------- the real knowledge base


def test_the_real_knowledge_base_is_well_formed() -> None:
    chunks = load_kb(KB)
    assert len(chunks) >= 40 and len({c.id for c in chunks}) == len(chunks)
    for c in chunks:
        assert 8 <= len(c.text.split()) <= MAX_WORDS, c.id
    for path in KB.glob("*.md"):
        assert "Synthetic policy text" in path.read_text(encoding="utf-8"), path.name


def test_the_policy_facts_agree_with_the_rest_of_the_project() -> None:
    """The store, the tiers and the design all assume these numbers; the text must say them."""
    text = "\n".join(c.text for c in load_kb(KB))
    assert "INR 1,000" in text  # the auto-refund limit (DESIGN.md section 1)
    assert "Gold members get 10 days" in text and "7 days of delivery" in text
    assert "INR 5,000" in text and "INR 499" in text
    assert "Final sale" in text  # the fixture order with a final-sale item
    assert "Delhivery" in text and "BlueDart" in text  # the carriers the seed uses


# ------------------------------------------------------------------------------ tokenising


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("returns", "return"),
        ("returned", "return"),
        ("returning", "return"),
        ("boxes", "box"),
        ("class", "class"),
        ("is", "is"),
        ("bed", "bed"),
        ("shipping", "shipp"),
    ],
)
def test_stemming_joins_related_forms_and_leaves_short_words_alone(
    word: str, expected: str
) -> None:
    assert stem(word) == expected


def test_tokenize_drops_stop_words_punctuation_and_case() -> None:
    assert tokenize("What is YOUR Return-Policy?") == ["return", "policy"]
    assert tokenize("INR 1,000 for order KRT-100002") == [
        "inr",
        "1",
        "000",
        "order",
        "krt",
        "100002",
    ]
    assert tokenize("ｒｅｔｕｒｎ") == ["return"]  # full-width letters normalise
    assert tokenize("") == []


# ---------------------------------------------------------------------------------- ranking


@pytest.fixture
def small() -> BM25Retriever:
    return BM25Retriever(
        [
            chunk("a#refund", "Refund timeline", "Refunds take 5 to 7 business days."),
            chunk("a#cancel", "Cancel", "Orders can be cancelled before they ship."),
            chunk("a#address", "Address", "The delivery address can change before shipping."),
        ]
    )


def test_the_best_match_comes_first_and_scores_fall(small: BM25Retriever) -> None:
    hits = small.search("how long do refunds take", 3)
    assert hits[0].chunk.id == "a#refund"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_stemmed_forms_match(small: BM25Retriever) -> None:
    assert small.search("cancelling an order")[0].chunk.id == "a#cancel"
    assert small.search("refunded")[0].chunk.id == "a#refund"


def test_a_query_with_no_known_words_returns_nothing(small: BM25Retriever) -> None:
    assert small.search("zebra quantum") == []
    assert small.search("the of and") == []
    assert small.search("") == []


def test_k_limits_results_and_zero_returns_nothing(small: BM25Retriever) -> None:
    assert len(small.search("order delivery refund", 2)) == 2
    assert small.search("refund", 0) == []


def test_ties_break_by_chunk_id_so_results_are_repeatable() -> None:
    r = BM25Retriever(
        [chunk("z#same", "Same", "alpha beta"), chunk("a#same", "Same", "alpha beta")]
    )
    assert [h.chunk.id for h in r.search("alpha")] == ["a#same", "z#same"]


def test_a_rarer_word_counts_for_more_than_a_common_one() -> None:
    r = BM25Retriever(
        [
            chunk("x#1", "One", "delivery delivery delivery common"),
            chunk("x#2", "Two", "delivery rare"),
            chunk("x#3", "Three", "delivery"),
        ]
    )
    assert r.search("rare delivery")[0].chunk.id == "x#2"


def test_a_negative_k_returns_nothing_rather_than_dropping_the_last_hit(
    small: BM25Retriever,
) -> None:
    assert small.search("refund order delivery", -1) == []


def test_a_rare_word_outweighs_a_common_one_even_with_equal_counts() -> None:
    r = BM25Retriever(
        [
            chunk("a#common", "Common", "everyday everyday"),
            chunk("b#rare", "Rare", "unusual"),
            chunk("c#x", "X", "everyday filler"),
            chunk("d#y", "Y", "everyday filler words"),
        ]
    )
    assert r.search("everyday unusual")[0].chunk.id == "b#rare"


def test_a_shorter_chunk_beats_a_longer_one_with_the_same_word_count() -> None:
    padding = " ".join(f"filler{i}" for i in range(40))
    r = BM25Retriever(
        [chunk("a#long", "Long", f"target {padding}"), chunk("b#short", "Short", "target")]
    )
    assert [h.chunk.id for h in r.search("target")] == ["b#short", "a#long"]


def test_an_empty_index_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Retriever([])


def test_the_heading_helps_a_short_query_find_its_chunk() -> None:
    r = BM25Retriever(
        [
            chunk("p#fee", "Return shipping fee", "Pickup costs money.", "Returns"),
            chunk("p#other", "Other", "Something about the fee for a fee.", "Returns"),
        ]
    )
    assert r.search("return shipping fee")[0].chunk.id == "p#fee"
