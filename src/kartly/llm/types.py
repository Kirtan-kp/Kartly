"""Provider-independent request/response types and the ChatClient interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    estimated: bool = False  # True when the provider reported no usage and we guessed

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class ToolCall:
    """One tool call requested by the model.

    Small models often emit broken JSON. That is not an exception: ``arguments`` is None and
    ``parse_error`` says why, so the agent loop can ask the model to repair the call.
    """

    id: str
    name: str
    arguments: dict[str, Any] | None
    raw_arguments: str
    parse_error: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    model: str  # the logical name from models.yaml
    provider_model: str  # the model ID the provider reports
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    usage: Usage
    cost_usd: Decimal  # shadow cost at reference prices, unrounded
    latency_ms: int
    ignored_params: tuple[str, ...] = ()  # request options this provider cannot take
    rate_limit: dict[str, str] = field(default_factory=dict)  # x-ratelimit-* response headers
    attempts: int = 1  # model calls it took to get this reply (more than 1: retries or fallback)
    failed_over_from: tuple[str, ...] = ()  # models tried and given up on before this one

    def assistant_message(self) -> dict[str, Any]:
        """This reply as a chat message, ready to append to the history for the next call."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.raw_arguments},
                }
                for call in self.tool_calls
            ]
        return message


class ChatClient(Protocol):
    """What the rest of the system needs from a model. One implementation per wire protocol;
    the resilient fallback wrapper (Week 5) implements it too."""

    name: str

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResponse: ...
