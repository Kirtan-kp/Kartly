"""A cheap, deterministic screen for prompt injection in a customer's message.

It is a tripwire, not a wall. It catches the common, lazy attacks (pretending to be the system,
telling the model to ignore its rules, naming tools or ``customer_id``) without spending a model
call, and sends those tickets to a human. A determined attacker can word things to get past it,
which is why the Tool Gateway stays the real defence: whatever the model is talked into, it can
only read the ticket's own customer's data (ADR-007, ADR-021).

Rules are deliberately specific so ordinary customers are not flagged: a message must contain
instruction-like phrasing aimed at the assistant, not merely a word like "system" or "ignore".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")

# name -> pattern, matched against normalised (casefolded, NFKC, single-spaced) text.
_RULES: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass|skip)\b[^.\n]{0,40}\b"
        r"(instructions?|rules?|guidelines?|polic(y|ies)|prompts?|restrictions?|safeguards?)\b"
    ),
    "rules_suspended": re.compile(
        r"\b(rules?|instructions?|restrictions?|guidelines?|safeguards?)\b[^.\n]{0,30}"
        r"\b(suspended|disabled|lifted|void|no longer (apply|apply to you)|do(es)? not apply)\b"
    ),
    "fake_system_message": re.compile(
        r"(\[\s*(system|admin|developer)\s*\]|<\s*/?\s*(system|admin|developer)\s*>|"
        r"^#{1,3}\s*system\b|\b(system|admin(istrator)?|developer) (notice|message|override|"
        r"instruction|prompt|command)s?\b|\bmessage from (the )?(system|admin|administrator|"
        r"developer|kartly (admin|team|staff))\b)",
        re.MULTILINE,
    ),
    "new_role": re.compile(
        r"\b(you are now|from now on you|act as|pretend (to be|you are)|developer mode|"
        r"jailbreak|do anything now)\b"
    ),
    "names_internal_tool": re.compile(
        r"\b(call|run|invoke|execute|use)\b[^.\n]{0,30}\b(get_order|get_tracking)\b"
    ),
    "names_identity_field": re.compile(r"\bcustomer_id\b"),
    "reveal_prompt": re.compile(
        r"\b(reveal|print|show|repeat|output|paste|tell me)\b[^.\n]{0,40}"
        r"\b(system prompt|your (instructions|rules|prompt)|hidden (instructions|prompt))\b"
    ),
}


@dataclass(frozen=True)
class ScreenResult:
    suspicious: bool
    rules: tuple[str, ...] = ()


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text).casefold()
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n+", "\n", text)


def screen_text(text: str) -> ScreenResult:
    """Names of the rules the text trips, in a stable order; suspicious if any."""
    normalised = _normalise(text)
    hits = tuple(name for name, pattern in _RULES.items() if pattern.search(normalised))
    return ScreenResult(bool(hits), hits)
