"""Token estimates and shadow-cost calculation (ADR-013).

Real spend is $0 on free tiers and local models, but cost is still a core engineering metric.
Every call is priced at *reference* per-token prices from config/models.yaml, so routing and
caching experiments produce meaningful numbers. Costs are Decimal and left unrounded here; the
caller rounds once, when it totals a run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kartly.llm.types import Usage

ONE_MILLION = Decimal(1_000_000)


class ModelPrice(BaseModel):
    """USD per 1M tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(ge=0)
    cached_input_per_mtok: Decimal | None = Field(default=None, ge=0)
    note: str = ""


def compute_cost(price: ModelPrice, usage: Usage) -> Decimal:
    """Shadow cost of one call. Cached prompt tokens use the cached rate when one is set."""
    cached = min(usage.cached_tokens, usage.prompt_tokens)
    fresh = usage.prompt_tokens - cached
    cached_rate = (
        price.cached_input_per_mtok
        if price.cached_input_per_mtok is not None
        else price.input_per_mtok
    )
    total = (
        fresh * price.input_per_mtok
        + cached * cached_rate
        + usage.completion_tokens * price.output_per_mtok
    )
    return total / ONE_MILLION


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English). Used only when a provider
    returns no usage, and for budget pre-checks. Real counts come from the provider."""
    return (len(text) + 3) // 4


def estimate_prompt_tokens(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] | None = None
) -> int:
    """Rough prompt size, counting each message's JSON plus a few tokens of framing."""
    total = sum(
        4 + estimate_tokens(json.dumps(m, ensure_ascii=False, default=str)) for m in messages
    )
    if tools:
        total += estimate_tokens(json.dumps(tools, ensure_ascii=False, default=str))
    return total
