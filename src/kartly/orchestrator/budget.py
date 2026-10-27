"""Hard limits for one agent run (DESIGN.md section 5, denial-of-wallet in section 10)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RunBudget:
    max_steps: int = 6  # model calls, including ones that failed
    max_tool_calls: int = 8  # calls that reached the gateway
    max_total_tokens: int = 20_000  # prompt + completion, summed over every model call
    max_cost_usd: Decimal = Decimal("0.30")  # shadow cost at reference prices
    timeout_s: float = 120.0  # wall clock, checked before every model call
    max_repairs: int = 1  # re-prompts after a broken tool call or an empty reply
    max_duplicate_calls: int = 1  # identical repeated tool calls answered with an error
    max_output_tokens: int = 1024  # per model call
    min_output_tokens: int = 128  # below this much room, stop instead of risking a cut-off reply

    def __post_init__(self) -> None:
        positive = {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_total_tokens": self.max_total_tokens,
            "timeout_s": self.timeout_s,
            "max_output_tokens": self.max_output_tokens,
            "min_output_tokens": self.min_output_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_cost_usd <= 0:
            raise ValueError(f"max_cost_usd must be positive, got {self.max_cost_usd}")
        if self.max_repairs < 0 or self.max_duplicate_calls < 0:
            raise ValueError("max_repairs and max_duplicate_calls cannot be negative")
        if self.min_output_tokens > self.max_output_tokens:
            raise ValueError("min_output_tokens cannot exceed max_output_tokens")
