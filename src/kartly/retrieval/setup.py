"""Build the policy-search tool from settings, for the API, the CLI and the evals."""

from __future__ import annotations

from pathlib import Path

from kartly.config import Settings
from kartly.retrieval.chunking import KnowledgeBaseError, load_kb
from kartly.retrieval.dense import DenseRetriever, Embedder, OllamaEmbedder
from kartly.tools import default_tools, write_tools
from kartly.tools.base import Tool
from kartly.tools.policy import policy_tool
from kartly.tools.returns import RETURN_TOOLS


def build_policy_tools(
    kb_dir: str,
    *,
    embed_model: str,
    ollama_url: str,
    min_score: float,
    cache_path: Path | None = None,
    embedder: Embedder | None = None,
) -> tuple[Tool, ...]:
    """Embed the knowledge base and return ``search_policy`` and ``check_return_eligibility``.

    Raises ``KnowledgeBaseError`` for a bad KB and ``EmbeddingError`` if Ollama cannot embed it,
    so a misconfigured server fails at start-up, not on the first customer question.
    """
    chunks = load_kb(kb_dir)
    if not chunks:
        raise KnowledgeBaseError(f"{kb_dir}: no knowledge-base chunks found")
    emb = embedder or OllamaEmbedder(embed_model, ollama_url, cache_path=cache_path)
    return (policy_tool(DenseRetriever(chunks, emb), min_score=min_score), *RETURN_TOOLS)


def gateway_tools(settings: Settings, *, cache_dir: Path | None = None) -> tuple[Tool, ...]:
    """The read tools and the write tools, plus ``search_policy`` when ``KARTLY_KB_DIR`` is set.

    A write tool is only offered to the model while its tier is not ``off`` (the gateway decides).
    """
    tools = default_tools() + write_tools(
        settings.refund_auto_limit_inr, settings.refund_daily_cap_inr
    )
    if not settings.kb_dir:
        return tools
    cache = (cache_dir or Path("evals/runs")) / "embedding_cache.json"
    return tools + build_policy_tools(
        settings.kb_dir,
        embed_model=settings.embed_model,
        ollama_url=settings.ollama_url,
        min_score=settings.policy_min_score,
        cache_path=cache,
    )
