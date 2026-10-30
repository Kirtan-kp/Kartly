"""Shared test helpers (import as ``tests.support``)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic.config import Config
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from kartly.store.models import Customer
from kartly.store.seed import seed_mock_store
from kartly.store.sqlite import make_sqlite_engine as make_sqlite_engine
from kartly.tools.base import CallContext, RiskTier, Tool, ToolEnv, ToolError

ROOT = Path(__file__).resolve().parents[1]
FIXED_NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)


def seeded_session_factory(engine: Engine) -> sessionmaker[Session]:
    """A session factory over an engine whose store holds the standard seed at FIXED_NOW."""
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        seed_mock_store(session, now=FIXED_NOW)
        session.commit()
    return factory


def make_ctx(customer_id: str) -> CallContext:
    return CallContext(ticket_id=uuid.uuid4(), run_id=uuid.uuid4(), customer_id=customer_id)


def alembic_config(conn: Connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "src/kartly/store/migrations"))
    cfg.attributes["connection"] = conn
    return cfg


# ---- tools used to test the gateway's failure handling (they write, then misbehave) ----


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _write_then_raise(env: ToolEnv, _args: Any) -> dict[str, Any]:
    env.session.add(Customer(id="C-BOOM", name="Boom", email="boom@example.com"))
    env.session.flush()
    raise RuntimeError("kaboom")


def _write_then_reject(env: ToolEnv, _args: Any) -> dict[str, Any]:
    env.session.add(Customer(id="C-NOPE", name="Nope", email="nope@example.com"))
    env.session.flush()
    raise ToolError("not_allowed", "That is not allowed.", reason="policy")


def _write_and_succeed(env: ToolEnv, _args: Any) -> dict[str, Any]:
    env.session.add(Customer(id="C-WROTE", name="Wrote", email="wrote@example.com"))
    env.session.flush()
    return {"written": True}


def _violate_unique_constraint(env: ToolEnv, _args: Any) -> dict[str, Any]:
    env.session.add(Customer(id="C-DUP1", name="Dup", email="dup@example.com"))
    env.session.add(Customer(id="C-DUP2", name="Dup", email="dup@example.com"))
    env.session.flush()  # IntegrityError
    return {}


def _tool(name: str, handler: Any) -> Tool:
    return Tool(
        name=name,
        tier=RiskTier.LOW_RISK_WRITE,
        description="test tool",
        args_model=NoArgs,
        handler=handler,
    )


EXPLODING_TOOL = _tool("explode", _write_then_raise)
REJECTING_TOOL = _tool("reject", _write_then_reject)
WRITING_TOOL = _tool("write", _write_and_succeed)
CONSTRAINT_TOOL = _tool("violate", _violate_unique_constraint)
