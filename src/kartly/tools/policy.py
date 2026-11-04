"""``search_policy`` (risk tier 0): look up Kartly's own policy text.

Unlike the order tools it touches no customer data, so it needs no ownership check; the gateway
still validates its arguments, audits the call and enforces the allow-list like any other tool.

The tool answers "which policy text is relevant", never "what should the customer be told": the
model must base its reply on the returned text and cite the chunk id. When nothing scores above
the threshold the result says so explicitly, so "no answer" is a first-class outcome rather than
the least-bad match presented as an answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kartly.retrieval import Retriever
from kartly.sanitize import clean_str
from kartly.tools.base import RiskTier, Tool, ToolEnv

MAX_RESULTS = 3
MAX_TEXT_CHARS = 1500
NOT_FOUND_MESSAGE = (
    "No policy text matched this question. Do not answer it from memory or guess: tell the "
    "customer you do not have that information and that a human colleague will follow up."
)


class PolicyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=3,
        max_length=300,
        description="The policy question in plain words, for example: how long do refunds take",
    )


def policy_tool(retriever: Retriever, *, min_score: float, max_results: int = MAX_RESULTS) -> Tool:
    """Build the tool around ``retriever``; only hits scoring at least ``min_score`` are shown."""
    if max_results < 1:
        raise ValueError("max_results must be at least 1")

    def search_policy(env: ToolEnv, args: PolicyArgs) -> dict[str, Any]:
        hits = [h for h in retriever.search(args.question, max_results) if h.score >= min_score]
        if not hits:
            return {"found": False, "results": [], "message": NOT_FOUND_MESSAGE}
        return {
            "found": True,
            "results": [
                {
                    "chunk_id": h.chunk.id,
                    "title": h.chunk.title,
                    "section": h.chunk.heading,
                    "text": clean_str(h.chunk.text, MAX_TEXT_CHARS),
                }
                for h in hits
            ],
        }

    return Tool(
        name="search_policy",
        tier=RiskTier.READ_ONLY,
        description=(
            "Search Kartly's policies (returns, refunds, shipping, cancellation, address "
            "changes, payments, damaged items, account and membership). Use it for any "
            "question about how Kartly works, not for a specific order."
        ),
        args_model=PolicyArgs,
        handler=search_policy,
    )
