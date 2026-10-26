from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from kartly.llm import ModelPrice, Usage, compute_cost, estimate_prompt_tokens, estimate_tokens

SMALL = ModelPrice(input_per_mtok=Decimal("0.05"), output_per_mtok=Decimal("0.08"))


def test_cost_at_reference_prices() -> None:
    assert compute_cost(SMALL, Usage(1_000_000, 1_000_000)) == Decimal("0.13")
    assert compute_cost(SMALL, Usage(1_200, 150)) == Decimal("0.000072")


def test_zero_usage_is_free() -> None:
    assert compute_cost(SMALL, Usage(0, 0)) == 0


def test_cached_tokens_use_the_cached_rate() -> None:
    price = ModelPrice(
        input_per_mtok=Decimal("0.05"),
        output_per_mtok=Decimal("0.08"),
        cached_input_per_mtok=Decimal("0.025"),
    )
    # 600 fresh * 0.05 + 400 cached * 0.025 = 40 (per million)
    assert compute_cost(price, Usage(1_000, 0, cached_tokens=400)) == Decimal("0.00004")


def test_cached_tokens_without_a_cached_rate_cost_the_normal_rate() -> None:
    assert compute_cost(SMALL, Usage(1_000, 0, cached_tokens=400)) == compute_cost(
        SMALL, Usage(1_000, 0)
    )


def test_more_cached_than_prompt_tokens_cannot_make_cost_negative() -> None:
    price = ModelPrice(
        input_per_mtok=Decimal("1"),
        output_per_mtok=Decimal("1"),
        cached_input_per_mtok=Decimal("0"),
    )
    assert compute_cost(price, Usage(100, 0, cached_tokens=500)) == 0


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelPrice(input_per_mtok=Decimal("-1"), output_per_mtok=Decimal("0"))


def test_unknown_price_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelPrice(
            input_per_mtok=Decimal(1),
            output_per_mtok=Decimal(1),
            output_per_ktok=Decimal(1),  # type: ignore[call-arg]
        )


def test_usage_total() -> None:
    assert Usage(10, 5).total_tokens == 15


def test_token_estimates() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    short = estimate_prompt_tokens([{"role": "user", "content": "hi"}])
    longer = estimate_prompt_tokens([{"role": "user", "content": "hi " * 500}])
    with_tools = estimate_prompt_tokens(
        [{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "x"}}]
    )
    assert short < with_tools
    assert longer > 10 * short
