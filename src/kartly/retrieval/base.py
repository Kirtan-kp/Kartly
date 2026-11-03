"""What every retriever returns, so BM25, dense search and fusion can be swapped and compared."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kartly.retrieval.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """The best ``k`` chunks for ``query``, best first. May return fewer (or none)."""
        ...
