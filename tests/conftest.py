"""Shared fixtures. Unit tests run on in-memory SQLite; see tests/integration for Postgres."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from kartly.store.models import Base


def _sqlite_engine() -> Engine:
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def engine_factory() -> Iterator[Callable[[], Engine]]:
    created: list[Engine] = []

    def make() -> Engine:
        engine = _sqlite_engine()
        created.append(engine)
        return engine

    yield make
    for engine in created:
        engine.dispose()


@pytest.fixture
def engine(engine_factory: Callable[[], Engine]) -> Engine:
    return engine_factory()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with sessionmaker(engine, expire_on_commit=False)() as s:
        yield s
