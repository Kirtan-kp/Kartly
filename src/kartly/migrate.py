"""Bring the database schema to the latest version (the command a deployment runs).

    python -m kartly.migrate              # wait for the database, then upgrade to head
    python -m kartly.migrate --check      # exit 0 only if the database is already at head

It does what ``alembic upgrade head`` does, but works from an installed package (no alembic.ini,
no source tree) and waits for a database that is still starting, which is what happens when a
container stack comes up (ADR-060). Exit codes: 0 done / up to date, 1 not at head (--check),
2 could not reach the database or the migration failed.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

MIGRATIONS = Path(__file__).parent / "store" / "migrations"


def alembic_config() -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    config.set_main_option("sqlalchemy.url", "")  # env.py then reads KARTLY_DATABASE_URL
    return config


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert head is not None  # a project with no migrations cannot be deployed
    return head


def current_revision(url: str) -> str | None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            return MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()


def wait_for_database(
    url: str,
    wait_s: float,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> str | None:
    """Try to connect until ``wait_s`` has passed: None when it works, else the last error."""
    deadline = clock() + wait_s
    error = "no attempt made"
    while True:
        engine = create_engine(url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return None
        except SQLAlchemyError as exc:
            error = str(exc.__cause__ or exc).splitlines()[0]
        finally:
            engine.dispose()
        if clock() >= deadline:
            return error
        sleep(min(2.0, max(0.0, deadline - clock())))


def main(argv: Sequence[str] | None = None) -> int:
    from kartly.config import get_settings, load_env

    parser = argparse.ArgumentParser(description="Upgrade the Kartly database schema.")
    parser.add_argument("--check", action="store_true", help="only report whether it is at head")
    parser.add_argument(
        "--wait", type=float, default=60.0, help="seconds to wait for the database (default 60)"
    )
    args = parser.parse_args(argv)
    load_env()
    url = get_settings().database_url
    failure = wait_for_database(url, args.wait)
    if failure is not None:
        print(f"error: cannot reach the database: {failure}")
        return 2
    head = head_revision()
    if args.check:
        current = current_revision(url)
        print(f"database is at {current or 'nothing'}; latest is {head}")
        return 0 if current == head else 1
    try:
        command.upgrade(alembic_config(), "head")
    except Exception as exc:  # a failed migration must stop a deployment, whatever the cause
        print(f"error: migration failed: {exc}")
        return 2
    print(f"database is at {current_revision(url)} (latest {head})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
