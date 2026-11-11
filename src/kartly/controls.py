"""Operator switches that work without a deploy (ADR-051).

Two controls, both stored in the database so every worker sees a change on its next use:

- ``agent``: ``on`` (default) or ``off``. Off: a new ticket is handed to a human before any model
  call, and no tool runs for it. A run already in progress finishes its current step first.
- ``writes``: ``on`` (default) or ``frozen``. Frozen: a write that would run on its own becomes a
  proposal instead, and a human approval cannot execute anything (rejecting still works). Reads
  are not affected.

A stored value that is not one of the allowed ones is treated as the safe state (off, frozen), so
a bad edit stops things instead of enabling them. Every change is audited, in the same
transaction as the change.

    python -m kartly.controls status
    python -m kartly.controls set agent off --by maya --reason "model returning wrong refunds"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from kartly.sanitize import clean_str
from kartly.store.models import ActorType, AuditLog, Control
from kartly.tools.orders import utc

AGENT = "agent"
WRITES = "writes"
# name -> (default, safe value, every allowed value)
CONTROLS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    AGENT: ("on", "off", ("on", "off")),
    WRITES: ("on", "frozen", ("on", "frozen")),
}
MAX_REASON_CHARS = 300


class ControlError(ValueError):
    """An unknown control or a value it does not allow."""


@dataclass(frozen=True)
class ControlState:
    name: str
    value: str
    allowed: tuple[str, ...]
    reason: str | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None  # None: never changed, so the default applies


class Controls:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._factory = session_factory
        self._clock = clock or (lambda: datetime.now(UTC))

    # ---------------------------------------------------------------- reading

    def get(self, name: str, session: Session | None = None) -> ControlState:
        """The control's state. Pass ``session`` when one is already open, so no second
        connection (and, on a test database, no second transaction) is needed."""
        if name not in CONTROLS:
            raise ControlError(f"unknown control '{name}'. Known: {', '.join(CONTROLS)}")
        if session is not None:
            return self._read(session, name)
        with self._factory() as own:
            return self._read(own, name)

    @staticmethod
    def _read(session: Session, name: str) -> ControlState:
        default, safe, allowed = CONTROLS[name]
        row = session.get(Control, name, populate_existing=True)
        if row is None:
            return ControlState(name, default, allowed)
        value = row.value if row.value in allowed else safe  # a bad stored value is "stop"
        return ControlState(name, value, allowed, row.reason, row.updated_by, utc(row.updated_at))

    def all(self) -> list[ControlState]:
        with self._factory() as session:
            return [self._read(session, name) for name in CONTROLS]

    def agent_on(self, session: Session | None = None) -> bool:
        return self.get(AGENT, session).value == "on"

    def writes_on(self, session: Session | None = None) -> bool:
        return self.get(WRITES, session).value == "on"

    # ---------------------------------------------------------------- changing

    def set(
        self, name: str, value: str, actor: str, reason: str | None = None
    ) -> tuple[ControlState, bool]:
        """Set a control; returns its new state and whether anything changed. A change is
        audited as ``control.set``; setting the value it already has records nothing."""
        if name not in CONTROLS:
            raise ControlError(f"unknown control '{name}'. Known: {', '.join(CONTROLS)}")
        allowed = CONTROLS[name][2]
        if value not in allowed:
            raise ControlError(
                f"'{value}' is not allowed for {name}. Use one of: {', '.join(allowed)}"
            )
        who = clean_str(actor, 64).strip()
        if not who:
            raise ControlError("who is changing this control?")
        why = clean_str(reason, MAX_REASON_CHARS) if reason else None
        for attempt in (1, 2):
            try:
                return self._set_once(name, value, who, why)
            except IntegrityError:
                if attempt == 2:  # the other writer created the row first: look again, once
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _set_once(
        self, name: str, value: str, who: str, why: str | None
    ) -> tuple[ControlState, bool]:
        with self._factory() as session:
            row = session.get(Control, name, with_for_update=True, populate_existing=True)
            before = self._read(session, name).value
            if row is None:
                row = Control(name=name, value=value, updated_by=who, updated_at=self._clock())
                session.add(row)
            elif before == value and row.value == value:
                return self._read(session, name), False
            row.value, row.reason, row.updated_by = value, why, who
            row.updated_at = self._clock()
            session.add(
                AuditLog(
                    actor_type=ActorType.HUMAN,
                    actor_id=who,
                    action="control.set",
                    target_type="control",
                    target_id=name,
                    details={"from": before, "to": value, "reason": why},
                )
            )
            session.commit()
            return self._read(session, name), True


def _format(state: ControlState) -> str:
    line = f"{state.name}: {state.value}"
    if state.updated_at is None:
        return line + " (default)"
    line += f"  (set by {state.updated_by} at {state.updated_at:%Y-%m-%d %H:%M} UTC"
    return line + (f', "{state.reason}")' if state.reason else ")")


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.config import load_env
    from kartly.store.db import make_engine, make_session_factory

    parser = argparse.ArgumentParser(description="Show or flip the operator switches.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="show every control")
    change = sub.add_parser("set", help="change a control")
    change.add_argument("name", choices=list(CONTROLS))
    change.add_argument("value")
    change.add_argument("--by", required=True, help="who is doing this (goes in the audit log)")
    change.add_argument("--reason", help="why (shown in the status and the audit log)")
    args = parser.parse_args(argv)

    load_env()
    engine = make_engine()
    try:
        controls = Controls(make_session_factory(engine))
        if args.command == "set":
            try:
                state, changed = controls.set(args.name, args.value, args.by, args.reason)
            except ControlError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(("changed: " if changed else "already: ") + _format(state))
        else:
            for state in controls.all():
                print(_format(state))
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
