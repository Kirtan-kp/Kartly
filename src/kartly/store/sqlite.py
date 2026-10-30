"""SQLite engines for tests and the eval runner. Production uses Postgres (see ``store.db``)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.pool import StaticPool

from kartly.store.models import Base


def make_sqlite_engine(path: str | None = None) -> Engine:
    """SQLite with foreign keys on and working SAVEPOINTs: in memory, or a file when ``path`` is
    given (several threads can then use separate connections, as with Postgres).

    pysqlite's default transaction handling breaks SAVEPOINT, so transaction control is handed to
    SQLAlchemy (the documented workaround). Production uses Postgres.
    """
    if path is None:
        engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
    else:
        engine = create_engine(
            f"sqlite:///{path}", connect_args={"check_same_thread": False, "timeout": 30}
        )

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.isolation_level = None
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    # A file database shared by several threads must take its write lock up front: with a plain
    # (deferred) BEGIN, a reader that later writes fails at once with "database is locked".
    begin = "BEGIN" if path is None else "BEGIN IMMEDIATE"

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Connection) -> None:
        conn.exec_driver_sql(begin)

    Base.metadata.create_all(engine)
    return engine
