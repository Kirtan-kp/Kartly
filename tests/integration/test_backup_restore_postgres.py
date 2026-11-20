"""The backup and restore scripts against a real Postgres: lose everything, get it back."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from kartly.reviewers import Reviewers
from tests.support import ROOT

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(sys.platform == "win32", reason="the scripts run under a POSIX bash"),
]
PW = "correct horse battery"


@pytest.fixture
def server(migrated: Engine) -> Engine:
    if not (shutil.which("pg_dump") and shutil.which("pg_restore")):
        pytest.skip("pg_dump / pg_restore are not installed")
    with migrated.connect() as conn:
        server_major = int(conn.execute(text("SHOW server_version_num")).scalar_one()) // 10000
    version = subprocess.run(
        ["pg_dump", "--version"], capture_output=True, text=True, check=True
    ).stdout  # "pg_dump (PostgreSQL) 16.15 (...)"
    found = re.search(r"\)\s+(\d+)", version)
    assert found, version
    client_major = int(found.group(1))
    if client_major != server_major:
        # An older client cannot read a newer server, and a newer one writes SQL an older server
        # rejects. Deployments run the tools inside the database container, so they always match.
        reason = f"pg_dump {client_major} does not match the server {server_major}"
        if os.environ.get("CI"):
            pytest.fail(reason)  # in CI a mismatch must be fixed, not hidden by a skip
        pytest.skip(reason)
    return migrated


def script_env(engine: Engine, tmp_path: Path) -> dict[str, str]:
    url = engine.url
    host = url.host or str(
        url.query.get("host", "")
    )  # a unix-socket server has the host in the query
    env = {k: v for k, v in os.environ.items() if not k.startswith("KARTLY_")}
    log = tmp_path / "hooks.log"
    record = tmp_path / "record.sh"  # the stop and start hooks are plain commands, not shell text
    record.write_text(f'#!/bin/sh\necho "$1" >> "{log}"\n')
    record.chmod(0o755)
    env.update(
        {
            "KARTLY_PG_EXEC": (
                f"env PGHOST={host} PGPORT={url.port or 5432} PGPASSWORD={url.password or ''}"
            ),
            "KARTLY_PG_USER": str(url.username),
            "KARTLY_PG_DB": str(url.database),
            "KARTLY_STOP_CMD": f"{record} stop",
            "KARTLY_START_CMD": f"{record} start",
            "KARTLY_MIGRATE_CMD": f"{sys.executable} -m kartly.migrate --wait 5",
            "KARTLY_DATABASE_URL": url.render_as_string(hide_password=False),
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    return env


def run(cmd: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=120)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A copy of the scripts in a scratch directory, so backups/ lands there, not in the repo."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("backup.sh", "restore.sh"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    return tmp_path


def counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as conn:
        return {
            t: int(conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one())
            for t in ("customers", "orders", "reviewers", "audit_log", "alembic_version")
        }


def test_a_lost_database_comes_back_from_a_backup(
    server: Engine, pg_factory: sessionmaker[Session], repo: Path
) -> None:
    accounts = Reviewers(pg_factory, scrypt_n=2**10)
    accounts.create("maya", PW, "admin")
    before = counts(server)
    assert before["customers"] > 0 and before["orders"] > 0 and before["reviewers"] == 1
    env = script_env(server, repo)

    backup = run(["bash", "scripts/backup.sh", "backups"], env, repo)
    assert backup.returncode == 0, backup.stderr
    [dump] = (repo / "backups").glob("kartly-*.dump")
    assert backup.stdout.strip() == f"backups/{dump.name}"
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
    assert not list((repo / "backups").glob(".*partial"))

    with server.begin() as conn:  # the volume is gone
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    with pytest.raises(DBAPIError), server.connect() as conn:
        conn.execute(text("SELECT count(*) FROM customers"))

    restored = run(["bash", "scripts/restore.sh", f"backups/{dump.name}", "--yes"], env, repo)
    assert restored.returncode == 0, restored.stdout + restored.stderr

    assert counts(server) == before
    assert accounts.login("maya", PW).reviewer.role == "admin"  # the password hash came back too
    with pytest.raises(DBAPIError), server.begin() as conn:  # and so did the audit log's guard
        conn.execute(text("UPDATE audit_log SET action = 'tampered'"))
    assert (repo / "hooks.log").read_text().split() == ["stop", "start"]
    [safety] = (repo / "backups").glob("pre-restore-*.dump")
    assert safety.stat().st_size > 0
    assert len(list((repo / "backups").glob("kartly-*.dump"))) == 1  # the safety copy is not pruned


def test_restoring_over_live_data_replaces_it_and_keeps_the_old_data_in_a_safety_copy(
    server: Engine, pg_factory: sessionmaker[Session], repo: Path
) -> None:
    env = script_env(server, repo)
    assert run(["bash", "scripts/backup.sh", "backups"], env, repo).returncode == 0
    [dump] = (repo / "backups").glob("kartly-*.dump")
    Reviewers(pg_factory, scrypt_n=2**10).create("later", PW)  # made after the backup
    assert counts(server)["reviewers"] == 1

    done = run(["bash", "scripts/restore.sh", str(dump), "--yes"], env, repo)
    assert done.returncode == 0, done.stdout + done.stderr
    assert counts(server)["reviewers"] == 0  # the backup did not have them
    [safety] = (repo / "backups").glob("pre-restore-*.dump")
    again = run(["bash", "scripts/restore.sh", str(safety), "--yes"], env, repo)  # undo the restore
    assert again.returncode == 0, again.stdout + again.stderr
    assert counts(server)["reviewers"] == 1
