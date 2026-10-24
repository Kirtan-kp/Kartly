"""Migration and constraint tests against a real Postgres.

Set KARTLY_TEST_DATABASE_URL to a dedicated *test* database, for example
postgresql+psycopg://kartly:kartly@localhost:5432/kartly_test
These tests DROP and recreate the public schema, so the database name must contain "test".
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection, Engine, create_engine, inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from kartly.store.models import (
    Action,
    Base,
    Customer,
    Order,
    Run,
    RunMode,
    Ticket,
)
from kartly.store.seed import seed_mock_store

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 23, 12, 0, 0, tzinfo=UTC)
EXPECTED_TABLES = {
    "customers",
    "orders",
    "order_items",
    "shipments",
    "tickets",
    "messages",
    "runs",
    "run_steps",
    "actions",
    "audit_log",
}


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    url = os.environ.get("KARTLY_TEST_DATABASE_URL")
    if not url:
        pytest.skip("KARTLY_TEST_DATABASE_URL is not set")
    engine = create_engine(url)
    if "test" not in (engine.url.database or ""):
        engine.dispose()
        pytest.fail("refusing to run: the test database name must contain 'test'")
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    yield engine
    engine.dispose()


def _alembic(conn: Connection) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(ROOT / "src/kartly/store/migrations"))
    cfg.attributes["connection"] = conn
    return cfg


@pytest.fixture
def migrated(pg_engine: Engine) -> Engine:
    with pg_engine.begin() as conn:
        command.upgrade(_alembic(conn), "head")
    return pg_engine


def test_upgrade_creates_all_tables(migrated: Engine) -> None:
    assert set(inspect(migrated).get_table_names()) >= EXPECTED_TABLES


def test_models_match_the_migration(migrated: Engine) -> None:
    with migrated.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(ctx, Base.metadata) == []


def test_downgrade_removes_everything_and_upgrade_is_repeatable(pg_engine: Engine) -> None:
    with pg_engine.begin() as conn:
        command.upgrade(_alembic(conn), "head")
    with pg_engine.begin() as conn:
        command.downgrade(_alembic(conn), "base")
    assert EXPECTED_TABLES.isdisjoint(inspect(pg_engine).get_table_names())
    with pg_engine.connect() as conn:
        left = conn.execute(
            text("SELECT count(*) FROM pg_proc WHERE proname = 'audit_log_reject_changes'")
        ).scalar_one()
    assert left == 0
    with pg_engine.begin() as conn:
        command.upgrade(_alembic(conn), "head")
    assert set(inspect(pg_engine).get_table_names()) >= EXPECTED_TABLES


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE audit_log SET action = 'tampered'",
        "DELETE FROM audit_log",
        "TRUNCATE audit_log",
    ],
)
def test_audit_log_is_append_only(migrated: Engine, statement: str) -> None:
    with migrated.begin() as conn:
        conn.execute(
            text("INSERT INTO audit_log (actor_type, action) VALUES ('system', 'created')")
        )
    with pytest.raises(DBAPIError, match="append-only"), migrated.begin() as conn:
        conn.execute(text(statement))
    with migrated.connect() as conn:
        assert conn.execute(text("SELECT action FROM audit_log")).scalar_one() == "created"


def test_seed_loads_and_types_are_postgres_native(migrated: Engine) -> None:
    with Session(migrated) as session:
        result = seed_mock_store(session, now=NOW)
        session.commit()
        assert result.orders == 200
        order = session.get(Order, "KRT-100001")
        assert order is not None
        assert order.placed_at.tzinfo is not None
        assert order.shipping_address["city"]

    with migrated.connect() as conn:
        jsonb = conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'orders' AND column_name = 'shipping_address'"
            )
        ).scalar_one()
        assert jsonb == "jsonb"
        checks = (
            conn.execute(
                text(
                    "SELECT conname FROM pg_constraint WHERE contype = 'c' AND conname LIKE 'ck_%'"
                )
            )
            .scalars()
            .all()
        )
    assert "ck_orders_refund_within_total" in checks
    assert "ck_orders_order_status" in checks


def test_constraints_hold_on_postgres(migrated: Engine) -> None:
    with Session(migrated) as session:
        session.add(Customer(id="C-9001", name="T", email="t@example.com"))
        ticket = Ticket(customer_id="C-9001")
        run = Run(mode=RunMode.SHADOW)
        ticket.runs = [run]
        session.add(ticket)
        session.flush()
        assert run.shadow_cost_usd == Decimal("0")
        for _ in range(2):
            session.add(
                Action(
                    run_id=run.id,
                    ticket_id=ticket.id,
                    tool_name="issue_refund",
                    args={"amount_paise": 1},
                    idempotency_key="dup",
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                break
        else:
            pytest.fail("a duplicate idempotency key was accepted")
