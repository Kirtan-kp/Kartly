"""Fixtures for tests that need a real Postgres.

Set KARTLY_TEST_DATABASE_URL to a dedicated *test* database, for example
postgresql+psycopg://kartly:kartly@localhost:5432/kartly_test
The fixtures DROP and recreate the public schema, so the database name must contain "test".
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from alembic import command
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from tests.support import FIXED_NOW, alembic_config, seeded_session_factory


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


@pytest.fixture
def migrated(pg_engine: Engine) -> Engine:
    with pg_engine.begin() as conn:
        command.upgrade(alembic_config(conn), "head")
    return pg_engine


@pytest.fixture
def pg_factory(migrated: Engine) -> sessionmaker[Session]:
    """Migrated Postgres, seeded at FIXED_NOW."""
    return seeded_session_factory(migrated)


@pytest.fixture
def pg_gateway(pg_factory: sessionmaker[Session]) -> ToolGateway:
    return ToolGateway(pg_factory, clock=lambda: FIXED_NOW)
