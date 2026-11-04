"""Check a drafted policy reply against what the tools actually returned (DESIGN.md section 10).

Two things are checked, both deterministic:

- every citation in square brackets must be a chunk that a ``search_policy`` call returned in
  this run (a model that never searched, or invented a section name, cannot cite one);
- every quantity with a unit (``30 days``, ``INR 5,000``, ``12 hours``, ``20%``) must appear in a
  tool result or in what the customer wrote.

It does NOT judge claims without numbers: "Kartly has no stores" next to a real chunk about
returns passes. That needs a model-based check; this guard only removes the cheap, certain
failures. A reply is checked only when it cites something or a search was made, so order answers
are untouched.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# A bracketed token with no spaces, such as [returns#return-window]. Only these are removed from
# the customer-facing text, so ordinary brackets like "[order KRT-100001]" are left alone.
CITATION = re.compile(r"[ \t]*\[([A-Za-z0-9_#.\-]{2,80})\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_UNIT = (
    r"(?:%|(?:(?:business|working|calendar)\s+)?(?:days?|hours?|hrs?|weeks?|months?)\b"
    r"|(?:rupees|inr)\b)"
)
_AFTER = re.compile(rf"(?<![\w.,])(\d[\d,]*(?:\.\d+)?)\s*{_UNIT}", re.IGNORECASE)
_CURRENCY = re.compile(r"(?:inr|rs\.?|₹)\s*(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)


@dataclass(frozen=True)
class GroundingResult:
    applicable: bool  # False when the reply neither cites nor followed a search
    citations: tuple[str, ...]
    text: str  # the reply without its citations (the original if there was nothing to strip)
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def _normal(number: str) -> str:
    """ "5,000", "5000.00" and "05000" compare equal."""
    cleaned = number.replace(",", "")
    trimmed = cleaned.rstrip("0").rstrip(".") if "." in cleaned else cleaned.lstrip("0")
    return trimmed or "0"


def _numbers(text: str) -> set[str]:
    return {_normal(n) for n in _NUMBER.findall(text)}


def _quantities(reply: str) -> list[tuple[str, str]]:
    """(label, normalised number) for each number the reply states with a unit."""
    found: list[tuple[str, str]] = []
    for match in _AFTER.finditer(reply):
        found.append((match.group(0).strip(), _normal(match.group(1))))
    for match in _CURRENCY.finditer(reply):
        found.append((match.group(0).strip(), _normal(match.group(1))))
    return found


def _tool_payloads(messages: Iterable[Mapping[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        if isinstance(payload, dict):
            out.append((content, payload))
    return out


def check_grounding(
    reply: str, messages: Sequence[Mapping[str, Any]], customer_text: str = ""
) -> GroundingResult:
    """``messages`` is the loop's final history (tool results are the tool-role entries)."""
    retrieved: set[str] = set()
    searched = False
    evidence_numbers = _numbers(customer_text)
    for content, payload in _tool_payloads(messages):
        evidence_numbers |= _numbers(content)
        data = payload.get("data")
        if isinstance(data, dict) and "found" in data:  # a search_policy result
            searched = True
            for item in data.get("results") or []:
                if isinstance(item, dict) and isinstance(item.get("chunk_id"), str):
                    retrieved.add(item["chunk_id"])
    citations = tuple(dict.fromkeys(CITATION.findall(reply)))
    if not citations and not searched:
        return GroundingResult(False, (), reply, ())
    problems = [
        f"cites [{c}], which no policy search returned in this run"
        for c in citations
        if c not in retrieved
    ]
    for label, number in _quantities(reply):
        if number not in evidence_numbers:
            problems.append(f"states '{label}', which is in no policy text or tool result")
    stripped = CITATION.sub("", reply).strip()
    return GroundingResult(True, citations, stripped or reply, tuple(dict.fromkeys(problems)))
