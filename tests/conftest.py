"""Shared fixtures. Unit tests run on in-memory SQLite; see tests/integration for Postgres."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from kartly.gateway import ToolGateway
from tests.support import FIXED_NOW, make_sqlite_engine, seeded_session_factory


@pytest.fixture
def engine_factory() -> Iterator[Callable[[], Engine]]:
    created: list[Engine] = []

    def make() -> Engine:
        engine = make_sqlite_engine()
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


@pytest.fixture
def seeded_factory(engine: Engine) -> sessionmaker[Session]:
    return seeded_session_factory(engine)


@pytest.fixture
def gateway(seeded_factory: sessionmaker[Session]) -> ToolGateway:
    return ToolGateway(seeded_factory, clock=lambda: FIXED_NOW)