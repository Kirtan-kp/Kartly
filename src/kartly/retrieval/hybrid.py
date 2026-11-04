"""Reciprocal rank fusion: combine rankings without comparing their incomparable scores."""

from __future__ import annotations

from collections.abc import Sequence

from kartly.retrieval.base import Hit, Retriever

RRF_K = 60  # the constant from the original RRF paper; larger flattens rank differences


class HybridRetriever:
    """Each retriever votes ``1 / (k + rank)`` for the chunks it returns; votes are summed.

    The fused score is a sum of votes, not a similarity, so it cannot be compared with a BM25 or
    cosine score and is not a good "no answer" signal on its own.
    """

    def __init__(
        self, retrievers: Sequence[Retriever], *, depth: int = 20, rrf_k: int = RRF_K
    ) -> None:
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one retriever")
        if depth < 1 or rrf_k < 1:
            raise ValueError("depth and rrf_k must be at least 1")
        self._retrievers = list(retrievers)
        self._depth = depth
        self._rrf_k = rrf_k
        self.name = "hybrid(" + "+".join(r.name for r in self._retrievers) + ")"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1:
            return []
        scores: dict[str, float] = {}
        chunks = {}
        for retriever in self._retrievers:
            for rank, hit in enumerate(retriever.search(query, self._depth), start=1):
                chunks[hit.chunk.id] = hit.chunk
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1 / (self._rrf_k + rank)
        ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [Hit(chunks[cid], scores[cid]) for cid in ordered[:k]]
