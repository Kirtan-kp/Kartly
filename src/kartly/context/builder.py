"""Assemble the first prompt of a run within a token budget (DESIGN.md section 8).

What is in a prompt, measured on real policy answers: the system prompt and the tool definitions
are about 85% of the tokens and are sent again on every model call; retrieved policy text is about
10%. So the budget is spent in this order: the system prompt and tool definitions are fixed and
counted, the ticket thread gets what is left. When the thread does not fit, the first customer
message (the original request) and the newest messages are kept and the middle is replaced by one
line saying how many messages were left out, never silently dropped.

Token counts are estimates (four characters per token), recorded next to the provider's real
count in the trace so the error is visible instead of assumed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from kartly.context.prompts import DEFAULT_POLICY_PROMPT, build_messages
from kartly.store.models import MessageRole

CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_TOKENS = 6000  # of Groq's 8k window, leaving room for tool results and the reply
MIN_NEWEST_CHARS = 400  # the newest message is never cut below this, whatever the budget
OMITTED_NOTE = "(earlier messages in this conversation were left out to save space)"


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN) if text else 0


@dataclass(frozen=True)
class ContextBreakdown:
    """Estimated tokens in the first prompt of a run, and what was trimmed to fit."""

    system: int
    tools: int
    thread: int
    total: int
    budget: int
    omitted_messages: int  # thread messages replaced by the one-line note
    truncated_newest: bool  # the newest message was cut to fit
    over_budget: bool  # even the fixed part plus a clipped newest message exceeds the budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BuiltContext:
    messages: list[dict[str, Any]]
    breakdown: ContextBreakdown


def _tokens(message: Mapping[str, Any]) -> int:
    return estimate_tokens(str(message.get("content") or ""))


class ContextBuilder:
    def __init__(self, budget_tokens: int = DEFAULT_BUDGET_TOKENS) -> None:
        if budget_tokens < 1:
            raise ValueError("budget_tokens must be at least 1")
        self._budget = budget_tokens

    def build(
        self,
        thread: Iterable[tuple[MessageRole, str]],
        *,
        tool_specs: Sequence[Mapping[str, Any]] = (),
        policy: bool = False,
        policy_prompt: str = DEFAULT_POLICY_PROMPT,
    ) -> BuiltContext:
        built = build_messages(thread, policy=policy, policy_prompt=policy_prompt)
        system, turns = built[0], built[1:]
        system_tokens = _tokens(system)
        tool_tokens = estimate_tokens(json.dumps(list(tool_specs), separators=(",", ":")))
        room = self._budget - system_tokens - tool_tokens

        kept, omitted, truncated = self._fit(turns, room)
        messages = [system, *kept]
        thread_tokens = sum(_tokens(m) for m in kept)
        total = system_tokens + tool_tokens + thread_tokens
        return BuiltContext(
            messages,
            ContextBreakdown(
                system=system_tokens,
                tools=tool_tokens,
                thread=thread_tokens,
                total=total,
                budget=self._budget,
                omitted_messages=omitted,
                truncated_newest=truncated,
                over_budget=total > self._budget,
            ),
        )

    @staticmethod
    def _fit(turns: list[dict[str, Any]], room: int) -> tuple[list[dict[str, Any]], int, bool]:
        """Keep the first message and as many of the newest as fit; note what was left out."""
        if not turns or sum(_tokens(m) for m in turns) <= room:
            return turns, 0, False
        first, newest = turns[0], turns[-1]
        if len(turns) == 1:
            return [ContextBuilder._clip(newest, room)], 0, True
        note = {"role": "user", "content": OMITTED_NOTE}
        # Fixed cost of any trimmed thread: the first message, the note and the newest message.
        newest_clipped = ContextBuilder._clip(newest, room - _tokens(first) - _tokens(note))
        tail: list[dict[str, Any]] = [newest_clipped]
        used = _tokens(first) + _tokens(note) + _tokens(newest_clipped)
        for message in reversed(turns[1:-1]):
            if used + _tokens(message) > room:
                break
            tail.insert(0, message)
            used += _tokens(message)
        omitted = len(turns) - 2 - (len(tail) - 1)
        middle = [note] if omitted else []
        return [first, *middle, *tail], omitted, newest_clipped is not newest

    @staticmethod
    def _clip(message: dict[str, Any], room: int) -> dict[str, Any]:
        """Cut a message to about ``room`` tokens, never below MIN_NEWEST_CHARS characters."""
        text = str(message.get("content") or "")
        limit = max(room * CHARS_PER_TOKEN, MIN_NEWEST_CHARS)
        if len(text) <= limit:
            return message
        return {**message, "content": text[: limit - 1] + "…"}  # the ellipsis is inside the limit
