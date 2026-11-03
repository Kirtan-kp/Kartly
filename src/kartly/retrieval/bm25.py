"""BM25 keyword retrieval, in plain Python (no index server, nothing to install).

Good at exact words ("INR 499", "Final sale", "pincode") and weak at paraphrase ("money back" for
"refund"), which is what the dense retriever added later is for. Text is lower-cased, split into
words, stripped of common stop words and lightly stemmed so "returns", "returned" and "returning"
match.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from kartly.guards.screen import normalise_text
from kartly.retrieval.base import Hit
from kartly.retrieval.chunking import Chunk

STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)
_WORD = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """A very small suffix stripper. It only has to map related words to the same token."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(normalise_text(text)) if w not in STOP_WORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self._chunks = list(chunks)
        self._k1, self._b = k1, b
        # The heading counts twice: it is the most specific description of what a chunk is about.
        docs = [tokenize(f"{c.heading}. {c.index_text}") for c in self._chunks]
        self._tf = [Counter(tokens) for tokens in docs]
        self._len = [len(tokens) for tokens in docs]
        self._avg = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        if not terms or k < 1:
            return []
        scored: list[Hit] = []
        for chunk, tf, length in zip(self._chunks, self._tf, self._len, strict=True):
            score = 0.0
            for term in set(terms):
                f = tf.get(term, 0)
                if f:
                    norm = f + self._k1 * (1 - self._b + self._b * length / self._avg)
                    score += self._idf[term] * f * (self._k1 + 1) / norm
            if score > 0:
                scored.append(Hit(chunk, score))
        scored.sort(key=lambda h: (-h.score, h.chunk.id))
        return scored[:k]
