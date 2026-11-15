"""Request and response bodies. Requests reject unknown fields; responses are typed so the
OpenAPI document (/docs) describes exactly what comes back."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_MESSAGE_CHARS = 4000


class TicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(
        min_length=1,
        max_length=16,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="The authenticated customer this ticket belongs to (set by the storefront).",
    )
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    external_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Your own id for this ticket. Sending it twice returns the first ticket.",
    )

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message cannot be blank")
        return value


class TicketCreated(BaseModel):
    ticket_id: uuid.UUID
    status: str
    duplicate: bool = False
    run_id: uuid.UUID | None = None
    outcome: str | None = None  # drafted | awaiting_approval | escalated, once the run has finished
    stop_reason: str | None = None
    answer: str | None = None  # the draft reply (shadow mode: a human sends it)


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class RunOut(BaseModel):
    id: uuid.UUID
    status: str
    outcome: str | None
    model_tier: str | None
    prompt_version: str | None
    tokens_in: int
    tokens_out: int
    shadow_cost_usd: str
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class TicketOut(BaseModel):
    id: uuid.UUID
    customer_id: str
    external_id: str | None
    channel: str
    status: str
    created_at: datetime
    messages: list[MessageOut]
    runs: list[RunOut]


class TraceRun(BaseModel):
    id: uuid.UUID
    ticket_id: uuid.UUID
    status: str
    outcome: str | None
    mode: str
    model_tier: str | None
    prompt_version: str | None
    tokens_in: int
    tokens_out: int
    shadow_cost_usd: str
    error: str | None
    started_at: str
    finished_at: str | None


class TraceTicket(BaseModel):
    id: uuid.UUID
    customer_id: str
    status: str


class TraceStep(BaseModel):
    idx: int
    kind: str
    name: str
    model: str | None
    input: dict[str, Any] | None
    output: dict[str, Any] | None
    tokens_in: int
    tokens_out: int
    latency_ms: int | None
    shadow_cost_usd: str
    error: str | None


class TraceAudit(BaseModel):
    id: int
    ts: str
    action: str
    details: dict[str, Any] | None


class TraceOut(BaseModel):
    run: TraceRun
    ticket: TraceTicket | None
    steps: list[TraceStep]
    audit: list[TraceAudit]


class Health(BaseModel):
    status: str


class ApprovalDecide(BaseModel):
    """Who is deciding and, optionally, why. The API key is shared, so the reviewer name is
    asserted by the caller (the admin tool that authenticated the person) and recorded as is."""

    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
    note: str | None = Field(default=None, max_length=500)


class ApprovalOut(BaseModel):
    id: uuid.UUID
    ticket_id: uuid.UUID
    customer_id: str
    status: str
    reason: str
    tool: str
    args: dict[str, Any]
    preview: dict[str, Any] | None  # what it would do (while pending)
    outcome: dict[str, Any] | None  # what happened, once decided
    action_status: str
    requested_at: datetime
    expires_at: datetime
    decided_by: str | None
    decided_at: datetime | None
    decision_note: str | None


class ApprovalList(BaseModel):
    approvals: list[ApprovalOut]
    expired_now: int  # requests this call found past their expiry and closed


class DecisionOut(BaseModel):
    approval_id: uuid.UUID
    ok: bool
    result: Literal["executed", "not_executed", "rejected", "expired", "already_decided", "refused"]
    message: str
    data: dict[str, Any] | None
    ticket_status: str | None


class ControlSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1, max_length=20)
    reviewer: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")
    reason: str | None = Field(default=None, max_length=300)


class ControlOut(BaseModel):
    name: str
    value: str
    allowed: list[str]
    reason: str | None
    updated_by: str | None
    updated_at: datetime | None  # None: never changed, the default applies


class ControlList(BaseModel):
    controls: list[ControlOut]


class ControlChanged(ControlOut):
    changed: bool


class JobRequeue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.@-]+$")


class DeadLetterOut(BaseModel):
    job_id: uuid.UUID
    ticket_id: uuid.UUID
    attempts: int
    last_error: str | None
    died_at: datetime | None


class DeadLetterList(BaseModel):
    jobs: list[DeadLetterOut]
    backlog: int  # jobs waiting or running now


class AlertOut(BaseModel):
    name: str
    status: str  # firing | ok
    severity: str  # warning | critical
    message: str
    details: dict[str, Any] | None
    since: datetime
    last_checked: datetime


class AlertList(BaseModel):
    alerts: list[AlertOut]


class AlertCheckOut(BaseModel):
    firing: list[str]  # rule names
    fired: list[str]  # newly announced
    cleared: list[str]
