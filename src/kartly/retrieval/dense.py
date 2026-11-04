"""Dense retrieval: embed chunks and questions with a local model, rank by cosine similarity.

The embedder is a protocol so tests and later providers can replace Ollama. Vectors for the
knowledge base are computed once when the retriever is built; with 44 chunks a plain Python scan
is fast enough, and it keeps the measurement honest (no index library hiding what is ranked).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx

from kartly.retrieval.base import Hit
from kartly.retrieval.chunking import Chunk

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# nomic-embed-text was trained with task prefixes; without them quality drops noticeably.
_PREFIXES = {"nomic-embed-text": ("search_document: ", "search_query: ")}


class EmbeddingError(RuntimeError):
    """The embedding service could not produce vectors (down, model missing, bad reply)."""


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Embeddings from Ollama's native ``/api/embed`` endpoint (the /v1 one is chat-oriented)."""

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        http_client: httpx.Client | None = None,
        timeout_s: float = 120.0,
        cache_path: Path | None = None,
    ) -> None:
        self.name = model
        self._model = model
        self._url = base_url.rstrip("/") + "/api/embed"
        self._http = http_client or httpx.Client()
        self._timeout = timeout_s
        self._doc_prefix, self._query_prefix = _PREFIXES.get(model, ("", ""))
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = self._load_cache()

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache_path is None or not self._cache_path.exists():
            return {}
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            return {str(k): [float(x) for x in v] for k, v in data.items()}
        except (OSError, ValueError, AttributeError, TypeError):
            return {}  # a damaged cache is only a slower run

    def _save_cache(self) -> None:
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except OSError:
            pass  # the cache is an optimisation, never a failure

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._model}\x00{text}".encode()).hexdigest()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._cache]
        if missing:
            vectors = self._request([t for _, t in missing])
            for (key, _), vector in zip(missing, vectors, strict=True):
                self._cache[key] = vector
            self._save_cache()
        return [self._cache[k] for k in keys]

    def _request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._http.post(
                self._url, json={"model": self._model, "input": texts}, timeout=self._timeout
            )
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"cannot connect to Ollama at {self._url} (is it running?)"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed ({type(exc).__name__})") from exc
        if response.status_code == 404:
            raise EmbeddingError(
                f"Ollama has no model {self._model!r}; run: ollama pull {self._model}"
            )
        if response.status_code != 200:
            raise EmbeddingError(f"Ollama returned HTTP {response.status_code}")
        try:
            vectors = response.json()["embeddings"]
            result = [[float(x) for x in v] for v in vectors]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError("unexpected reply from Ollama /api/embed") from exc
        if len(result) != len(texts) or any(not v for v in result):
            raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(result)}")
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self._query_prefix + text])[0]


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.name = f"dense:{embedder.name}"
        self._chunks = list(chunks)
        self._embedder = embedder
        # Same text BM25 indexes: the title and heading are part of what a chunk is about.
        texts = [c.index_text for c in self._chunks]
        self._vectors = [_normalise(v) for v in embedder.embed_documents(texts)] if texts else []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if k < 1 or not self._chunks or not query.strip():
            return []
        q = _normalise(self._embedder.embed_query(query))
        scored = [
            Hit(chunk, sum(a * b for a, b in zip(q, vec, strict=True)))
            for chunk, vec in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda h: (-h.score, h.chunk.id))
        return scored[:k]
