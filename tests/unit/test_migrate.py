"""The deployment's migration command, and the packaging that lets it work when installed."""

from __future__ import annotations

import fnmatch
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from kartly.migrate import (
    MIGRATIONS,
    alembic_config,
    current_revision,
    head_revision,
    main,
    wait_for_database,
)
from tests.support import ROOT


@pytest.fixture
def db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = f"sqlite:///{tmp_path / 'migrate.db'}"
    monkeypatch.setenv("KARTLY_DATABASE_URL", url)
    monkeypatch.setattr("kartly.config.load_env", lambda: None)
    from kartly.config import get_settings

    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_it_upgrades_an_empty_database_to_the_latest_revision(
    db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--wait", "1"]) == 0
    assert {"tickets", "reviewers", "jobs", "alembic_version"} <= tables(db_url)
    assert current_revision(db_url) == head_revision()
    assert f"latest {head_revision()}" in capsys.readouterr().out


def test_running_it_twice_changes_nothing(db_url: str) -> None:
    assert main(["--wait", "1"]) == 0
    before = tables(db_url)
    assert main(["--wait", "1"]) == 0
    assert tables(db_url) == before and current_revision(db_url) == head_revision()


def test_check_reports_whether_the_database_is_at_the_latest_revision(
    db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--check", "--wait", "1"]) == 1  # a database with nothing in it
    assert "nothing" in capsys.readouterr().out
    assert tables(db_url) <= {"alembic_version"}  # and --check changed nothing
    main(["--wait", "1"])
    assert main(["--check", "--wait", "1"]) == 0
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = '0001'"))
    engine.dispose()
    assert main(["--check", "--wait", "1"]) == 1  # behind


def test_a_failing_migration_is_reported_and_stops_the_deployment(
    db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    engine = create_engine(db_url)
    with engine.begin() as conn:  # a database stamped with a revision this code has never heard of
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version VALUES ('from-a-newer-release')"))
    engine.dispose()
    assert main(["--wait", "1"]) == 2
    assert "migration failed" in capsys.readouterr().out


def test_an_unreachable_database_is_reported_and_stops_the_deployment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("KARTLY_DATABASE_URL", "sqlite:////nonexistent-dir/x/y.db")
    monkeypatch.setattr("kartly.config.load_env", lambda: None)
    from kartly.config import get_settings

    get_settings.cache_clear()
    try:
        assert main(["--wait", "0"]) == 2
    finally:
        get_settings.cache_clear()
    assert "cannot reach the database" in capsys.readouterr().out


def test_waiting_retries_until_the_database_answers_then_gives_up_at_the_deadline(
    tmp_path: Path,
) -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(s: float) -> None:
        sleeps.append(s)
        now[0] += s

    bad = "sqlite:////nonexistent-dir/x/y.db"
    assert wait_for_database(bad, 7.0, sleep=sleep, clock=lambda: now[0]) is not None
    assert sum(sleeps) == pytest.approx(7.0) and max(sleeps) <= 2.0 and len(sleeps) >= 4
    good = f"sqlite:///{tmp_path / 'ok.db'}"
    assert wait_for_database(good, 7.0, sleep=sleep, clock=lambda: now[0]) is None


def test_the_script_location_is_the_packaged_directory_not_the_working_directory() -> None:
    cfg = alembic_config()
    assert Path(cfg.get_main_option("script_location") or "") == MIGRATIONS
    assert (MIGRATIONS / "env.py").is_file() and (MIGRATIONS / "versions").is_dir()


def test_every_file_alembic_needs_is_listed_as_package_data() -> None:
    """Migrations are not a Python package, so setuptools leaves them out of the wheel unless the
    patterns say so; an image built from the wheel could otherwise not migrate."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"][
        "package-data"
    ]
    patterns = data["kartly"]
    base = ROOT / "src" / "kartly"
    needed = [p for p in (MIGRATIONS).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for path in needed:
        rel = path.relative_to(base).as_posix()
        assert any(fnmatch.fnmatch(rel, pat) for pat in patterns), rel
    assert any(p.name.startswith("0007") for p in needed)
