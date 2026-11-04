"""Dense retrieval and fusion, with a fake embedder and a mocked Ollama (no network)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from kartly.evals.retrieval import main
from kartly.retrieval import (
    BM25Retriever,
    Chunk,
    DenseRetriever,
    EmbeddingError,
    Hit,
    HybridRetriever,
    OllamaEmbedder,
)

WORDS = ["refund", "return", "ship", "cancel", "card"]


def chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, doc="f", title="Doc", heading=cid, text=text)


def vec(text: str) -> list[float]:
    low = text.lower()
    return [float(low.count(w)) for w in WORDS]


class FakeEmbedder:
    name = "fake"

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        return [vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return vec(text)


CHUNKS = [
    chunk("a", "refund refund money back"),
    chunk("b", "ship ship courier"),
    chunk("c", "cancel the order before it ships"),
]


def test_dense_ranks_by_cosine_and_is_length_insensitive() -> None:
    retriever = DenseRetriever(CHUNKS, FakeEmbedder())
    hits = retriever.search("refund", 3)
    assert [h.chunk.id for h in hits][0] == "a"
    assert hits[0].score == pytest.approx(1.0)
    assert retriever.name == "dense:fake"
    assert [h.chunk.id for h in retriever.search("refund refund refund refund", 1)] == ["a"]


def test_dense_edge_cases() -> None:
    retriever = DenseRetriever(CHUNKS, FakeEmbedder())
    assert retriever.search("refund", 0) == []
    assert retriever.search("refund", -1) == []
    assert retriever.search("   ", 3) == []
    assert DenseRetriever([], FakeEmbedder()).search("refund") == []
    zero = retriever.search("zzz", 3)  # no overlap: zero vector must not divide by zero
    assert [h.score for h in zero] == [0.0, 0.0, 0.0]
    assert [h.chunk.id for h in zero] == ["a", "b", "c"]  # ties break by id


def test_embeds_the_title_and_heading_with_the_body() -> None:
    seen: list[str] = []

    class Spy(FakeEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            seen.extend(texts)
            return super().embed_documents(texts)

    DenseRetriever([chunk("Return window", "seven days")], Spy())
    assert seen == ["Doc. Return window. seven days"]


class Fixed:
    def __init__(self, name: str, ids: list[str]) -> None:
        self.name = name
        self._ids = ids

    def search(self, query: str, k: int = 5) -> list[Hit]:
        by_id = {c.id: c for c in CHUNKS}
        return [Hit(by_id[i], 1.0) for i in self._ids][:k]


def test_rrf_rewards_agreement_over_a_single_first_place() -> None:
    hybrid = HybridRetriever([Fixed("x", ["a", "b"]), Fixed("y", ["c", "b"])])
    hits = hybrid.search("q", 3)
    assert [h.chunk.id for h in hits] == ["b", "a", "c"]  # b is 2nd twice; a/c tie, id order
    assert hits[0].score == pytest.approx(2 / 62)
    assert hybrid.name == "hybrid(x+y)"


def test_rrf_includes_chunks_only_one_retriever_found_and_respects_k() -> None:
    hybrid = HybridRetriever([Fixed("x", ["a"]), Fixed("y", ["b"])])
    assert {h.chunk.id for h in hybrid.search("q", 5)} == {"a", "b"}
    assert len(hybrid.search("q", 1)) == 1
    assert hybrid.search("q", 0) == []


def test_rrf_depth_limits_what_each_retriever_contributes() -> None:
    hybrid = HybridRetriever([Fixed("x", ["a", "b", "c"])], depth=2)
    assert [h.chunk.id for h in hybrid.search("q", 5)] == ["a", "b"]


@pytest.mark.parametrize("kwargs", [{"depth": 0}, {"rrf_k": 0}])
def test_rrf_rejects_bad_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        HybridRetriever([Fixed("x", [])], **kwargs)
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_hybrid_over_real_retrievers_returns_chunks() -> None:
    hybrid = HybridRetriever([BM25Retriever(CHUNKS), DenseRetriever(CHUNKS, FakeEmbedder())])
    assert hybrid.search("cancel", 1)[0].chunk.id == "c"


# ------------------------------------------------------------------ Ollama embedder


def ollama(handler: httpx.MockTransport, **kwargs: object) -> OllamaEmbedder:
    return OllamaEmbedder("nomic-embed-text", http_client=httpx.Client(transport=handler), **kwargs)  # type: ignore[arg-type]


def test_ollama_embedder_sends_prefixed_batches_and_parses_vectors() -> None:
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"embeddings": [[1, 2]] * len(body["input"])})

    emb = ollama(httpx.MockTransport(handle))
    assert emb.embed_documents(["x", "y"]) == [[1.0, 2.0], [1.0, 2.0]]
    emb.embed_query("q")
    assert bodies[0]["input"] == ["search_document: x", "search_document: y"]
    assert bodies[1]["input"] == ["search_query: q"]
    assert bodies[0]["model"] == "nomic-embed-text"


def test_other_models_get_no_prefix() -> None:
    bodies: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    emb = OllamaEmbedder("other", http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    emb.embed_query("q")
    assert bodies[0]["input"] == ["q"]


def test_cache_avoids_repeat_requests_across_instances(tmp_path: Path) -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"embeddings": [[0.5, 0.5]] * len(json.loads(request.content)["input"])}
        )

    cache = tmp_path / "sub" / "cache.json"
    ollama(httpx.MockTransport(handle), cache_path=cache).embed_documents(["x", "y"])
    again = ollama(httpx.MockTransport(handle), cache_path=cache)
    assert again.embed_documents(["x", "y"]) == [[0.5, 0.5], [0.5, 0.5]]
    assert calls == 1
    again.embed_documents(["x", "z"])  # only the new text is requested
    assert calls == 2


def test_a_damaged_cache_is_ignored(tmp_path: Path) -> None:
    cache = tmp_path / "c.json"
    cache.write_text("{not json", encoding="utf-8")
    emb = ollama(
        httpx.MockTransport(lambda r: httpx.Response(200, json={"embeddings": [[1.0]]})),
        cache_path=cache,
    )
    assert emb.embed_query("q") == [1.0]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, json={"error": "model not found"}), "ollama pull nomic-embed-text"),
        (httpx.Response(500), "HTTP 500"),
        (httpx.Response(200, json={"nope": 1}), "unexpected reply"),
        (httpx.Response(200, json={"embeddings": [[1.0], [2.0]]}), "got 2"),
        (httpx.Response(200, json={"embeddings": [[]]}), "got 1"),
        (httpx.Response(200, text="not json"), "unexpected reply"),
    ],
)
def test_embedding_errors_are_specific(response: httpx.Response, message: str) -> None:
    emb = ollama(httpx.MockTransport(lambda r: response))
    with pytest.raises(EmbeddingError, match=message):
        emb.embed_query("q")


def test_connection_failure_says_ollama_may_not_be_running() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(EmbeddingError, match="is it running"):
        ollama(httpx.MockTransport(handle)).embed_query("q")


def test_cli_reports_a_missing_embedding_service_without_a_traceback(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    code = main(
        [
            "--retriever",
            "dense",
            "--ollama-url",
            "http://127.0.0.1:1",
            "--no-save",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err
