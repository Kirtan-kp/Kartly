"""Knowledge-base retrieval: chunks with citable ids, and retrievers over them (DESIGN.md s7)."""

from kartly.retrieval.base import Hit, Retriever
from kartly.retrieval.bm25 import BM25Retriever, tokenize
from kartly.retrieval.chunking import Chunk, KnowledgeBaseError, kb_digest, load_kb
from kartly.retrieval.dense import (
    DenseRetriever,
    Embedder,
    EmbeddingError,
    OllamaEmbedder,
)
from kartly.retrieval.hybrid import HybridRetriever

__all__ = [
    "BM25Retriever",
    "Chunk",
    "DenseRetriever",
    "Embedder",
    "EmbeddingError",
    "HybridRetriever",
    "OllamaEmbedder",
    "Hit",
    "KnowledgeBaseError",
    "Retriever",
    "kb_digest",
    "load_kb",
    "tokenize",
]
