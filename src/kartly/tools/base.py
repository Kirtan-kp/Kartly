"""Types shared by the Tool Gateway and the tool implementations.

This module must not import from ``kartly.gateway`` (the gateway imports this module).
"""

from __future__ import annotations

import enum
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session


class RiskTier(enum.IntEnum):
    """Mirrors the risk tiers in DESIGN.md section 1."""

    READ_ONLY = 0
    LOW_RISK_WRITE = 1
    MONEY = 2
    NEVER_AUTONOMOUS = 3


@dataclass(frozen=True)
class CallContext:
    """Who a tool call is for. Every field comes from authenticated server state.

    ``customer_id`` is taken from the ticket, never from the model's output or the ticket text.
    It is the only source of identity a tool may use, which is what stops a prompt-injected
    request from reaching another customer's data.
    """

    ticket_id: uuid.UUID
    run_id: uuid.UUID
    customer_id: str


@dataclass(frozen=True)
class ToolEnv:
    """What a handler may use: a session inside the gateway's savepoint, the caller, a clock."""

    session: Session
    ctx: CallContext
    now: datetime


class ToolError(Exception):
    """An expected, model-visible failure.

    ``message`` is shown to the model, so it must never contain data the caller may not see.
    ``reason`` and ``security_event`` are internal: they go to the audit log only.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        reason: str | None = None,
        security_event: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.reason = reason
        self.security_event = security_event


ToolHandler = Callable[[ToolEnv, Any], dict[str, Any]]
# A rule that says why a human must approve this particular call, or None if none is needed.
ApprovalRule = Callable[[ToolEnv, Any], str | None]

_TOOL_NAME = re.compile(r"[a-z][a-z0-9_]{0,59}")


@dataclass(frozen=True)
class Tool:
    name: str
    tier: RiskTier
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler
    # Writes only. ``preview`` runs every check the handler runs, changes nothing, and returns a
    # short description of what would happen. It is what a shadow-mode proposal and an approval
    # request show. ``needs_approval`` can require a human even in auto mode (money over a limit).
    preview: ToolHandler | None = None
    needs_approval: ApprovalRule | None = None

    @property
    def is_write(self) -> bool:
        return self.tier >= RiskTier.LOW_RISK_WRITE

    def __post_init__(self) -> None:
        if self.is_write and self.preview is None:
            raise ValueError(f"{self.name}: a write tool needs a preview")
        if not self.is_write and (self.preview is not None or self.needs_approval is not None):
            raise ValueError(f"{self.name}: only write tools take a preview or approval rule")
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"invalid tool name: {self.name!r}")
        if self.args_model.model_config.get("extra") != "forbid":
            # Unknown arguments, above all a customer_id, must be rejected, not ignored.
            raise ValueError(f"{self.name}: args_model must set extra='forbid'")
