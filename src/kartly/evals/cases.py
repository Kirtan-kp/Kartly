"""Golden eval cases: the ticket, whose it is, and what a correct run looks like.

Cases live in ``evals/golden/*.yaml`` (a list of cases per file). They are written against the
mock store's fixture customers and orders (README, "Mock store and eval fixtures"), which are
stable, and graded by ``kartly.evals.grader`` mostly on what happened (tool calls, gateway
security events, database state) and only then on what the reply says.

Everything unknown is rejected, so a typo in a case file fails loudly instead of silently
loosening a check.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Outcome = Literal["drafted", "escalated", "awaiting_approval"]
Intent = Literal[
    "order_status", "tracking", "return", "refund", "cancel", "address_change", "policy_faq",
    "escalation",
]  # fmt: skip
_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class CaseFileError(ValueError):
    """A case file is unreadable or invalid; the message says which file and case."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CallMatch(_Strict):
    """Matches tool calls: the tool is one of ``any_of`` (empty: any tool) and every ``args``
    entry equals what the model sent (compared ignoring case and surrounding spaces)."""

    any_of: list[str] = Field(default_factory=list)
    args: dict[str, str] = Field(default_factory=dict)

    def describe(self) -> str:
        tools = "|".join(self.any_of) or "any tool"
        args = " ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{tools} {args}".strip()


class AnswerExpect(_Strict):
    """Checks on the draft reply (substrings, ignoring case, quotes and repeated spaces)."""

    # Every group needs at least one of its alternatives: [["in transit", "on its way"], ["25 Oct"]]
    must_mention: list[list[str]] = Field(default_factory=list)
    # Wrong or invented claims (a quality failure).
    must_not_mention: list[str] = Field(default_factory=list)
    # Another customer's data, or a claim that something unauthorised was done (a SAFETY failure).
    must_not_leak: list[str] = Field(default_factory=list)

    @field_validator("must_mention")
    @classmethod
    def _groups_not_empty(cls, groups: list[list[str]]) -> list[list[str]]:
        for group in groups:
            if not group or any(not option.strip() for option in group):
                raise ValueError("each must_mention group needs non-blank alternatives")
        return groups

    @field_validator("must_not_mention", "must_not_leak")
    @classmethod
    def _phrases_not_blank(cls, phrases: list[str]) -> list[str]:
        if any(not phrase.strip() for phrase in phrases):
            raise ValueError("blank phrases would match every reply")
        return phrases


StoreTable = Literal["customers", "orders", "order_items", "shipments", "actions"]


class StateCheck(_Strict):
    """A value one order row must hold after the run (a write case's proof that the store ended
    as it should). ``severity: safety`` is for rows that must NOT have changed."""

    table: Literal["orders"] = "orders"
    id: str
    field: Literal["status", "refunded_paise"]
    equals: str | int
    severity: Literal["safety", "quality"] = "quality"


class Expect(_Strict):
    outcome: Outcome
    # Each entry must be matched by at least one tool call the model made.
    calls: list[CallMatch] = Field(default_factory=list)
    # Any call matching one of these fails the case (a SAFETY failure).
    forbidden_calls: list[CallMatch] = Field(default_factory=list)
    max_tool_calls: int | None = Field(default=None, ge=0)
    # The security events the gateway should have recorded: exactly this set (default none).
    security_events: list[str] = Field(default_factory=list)
    # Store tables that should differ afterwards (default none). Any other table changing is a
    # SAFETY failure, and an expected one that did not change is a quality failure.
    changed_tables: list[StoreTable] = Field(default_factory=list)
    # Rows that must hold a value afterwards.
    state: list[StateCheck] = Field(default_factory=list)
    # How many approval requests the run should have created (more than this is a SAFETY failure).
    approvals: int = Field(default=0, ge=0)
    answer: AnswerExpect = Field(default_factory=AnswerExpect)

    @field_validator("forbidden_calls")
    @classmethod
    def _forbidden_not_empty(cls, matches: list[CallMatch]) -> list[CallMatch]:
        if any(not m.any_of and not m.args for m in matches):
            raise ValueError(
                "a forbidden call must name a tool or an argument (use max_tool_calls: 0)"
            )
        return matches


class Case(_Strict):
    id: str
    description: str
    tier: int = Field(ge=0, le=3)
    intent: Intent
    tags: list[str] = Field(default_factory=list)
    customer_id: str
    message: str = Field(min_length=1)
    # Offer the write tools (cancel_order, issue_refund) with both write tiers in this mode. Left
    # out, the case runs with reads only, as every case did before Week 4.
    write_mode: Literal["shadow", "auto"] | None = None
    expect: Expect
    # Other behaviours that are just as right (a model may legitimately take another safe path).
    # The run passes if it fully meets ``expect`` or any one of these.
    also_acceptable: list[Expect] = Field(default_factory=list)
    # What a human reviewer should know about why the case exists.
    notes: str = ""

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError(
                "id must be lower-case words joined by hyphens, e.g. 'idor-order-lookup'"
            )
        return value

    @model_validator(mode="after")
    def _writes_need_a_write_mode(self) -> Case:
        wants_writes = any(
            e.outcome == "awaiting_approval"
            or e.approvals > 0
            or any(t != "customers" for t in e.changed_tables)
            or any(c.severity == "quality" and c.field == "refunded_paise" for c in e.state)
            for e in (self.expect, *self.also_acceptable)
        )
        if wants_writes and self.write_mode is None:
            raise ValueError("this case expects a write or an approval, so set write_mode")
        return self

    @model_validator(mode="after")
    def _escalation_has_no_answer_checks(self) -> Case:
        answer = self.expect.answer
        if self.expect.outcome == "escalated" and (answer.must_mention or answer.must_not_mention):
            raise ValueError("an escalated run has no reply; remove the answer checks")
        return self


def _read_file(path: Path) -> list[Case]:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CaseFileError(f"{path}: cannot read ({exc})") from exc
    if not isinstance(raw, list) or not raw:
        raise CaseFileError(f"{path}: expected a non-empty list of cases")
    cases: list[Case] = []
    for index, item in enumerate(raw, start=1):
        label = item.get("id", f"#{index}") if isinstance(item, dict) else f"#{index}"
        try:
            cases.append(Case.model_validate(item))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'case'}: {e['msg']}" for e in exc.errors()
            )
            raise CaseFileError(f"{path}: case {label}: {problems}") from exc
    return cases


def load_cases(path: Path | str) -> list[Case]:
    """Load one YAML file, or every ``*.yaml`` in a directory (by name). Ids must be unique."""
    root = Path(path)
    files = sorted(root.glob("*.yaml")) if root.is_dir() else [root]
    if not files:
        raise CaseFileError(f"{root}: no *.yaml case files found")
    cases = [case for file in files for case in _read_file(file)]
    duplicates = sorted(i for i, n in Counter(c.id for c in cases).items() if n > 1)
    if duplicates:
        raise CaseFileError(f"duplicate case ids: {', '.join(duplicates)}")
    return cases
