"""The backup and restore scripts' rules, driven with stand-ins for pg_dump and pg_restore.

The real round trip against Postgres is tests/integration/test_backup_restore_postgres.py.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import ROOT

# On Windows `bash` is usually WSL, which sees neither this process's environment (the stand-in
# tools and settings) nor Windows paths; the scripts are for Linux servers (use WSL or CI there).
pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or sys.platform == "win32",
    reason="needs a POSIX bash that shares the environment (Linux or macOS)",
)

FAKE_PG_DUMP = """#!/bin/sh
echo "pg_dump $*" >> "$FAKE_LOG"
[ -n "$FAKE_EMPTY_DUMP" ] && exit 0
[ -n "$FAKE_DUMP_FAILS" ] && { echo "pg_dump: connection refused" >&2; exit 1; }
echo "PGDMP-fake-$(date +%s%N)"
"""
FAKE_PG_RESTORE = """#!/bin/sh
cat > /dev/null
case "$*" in
  *--list*) [ -n "$FAKE_BAD_DUMP" ] && exit 1; exit 0 ;;
esac
echo "restore $*" >> "$FAKE_LOG"
[ -n "$FAKE_RESTORE_FAILS" ] && { echo "pg_restore: error" >&2; exit 1; }
exit 0
"""
HOOK = """#!/bin/sh
echo "$1" >> "$FAKE_LOG"
"""


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "scripts").mkdir()
        for name in ("backup.sh", "restore.sh"):
            shutil.copy(ROOT / "scripts" / name, root / "scripts" / name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name, body in (
            ("pg_dump", FAKE_PG_DUMP),
            ("pg_restore", FAKE_PG_RESTORE),
            ("hook", HOOK),
        ):
            path = bin_dir / name
            path.write_text(body)
            path.chmod(0o755)
        self.log = root / "calls.log"
        self.log.write_text("")
        self.env = {
            **{k: v for k, v in os.environ.items() if not k.startswith(("KARTLY_", "FAKE_"))},
            "KARTLY_PG_EXEC": f"env PATH={bin_dir}:{os.environ['PATH']}",
            "KARTLY_STOP_CMD": f"{bin_dir}/hook stop",
            "KARTLY_START_CMD": f"{bin_dir}/hook start",
            "KARTLY_MIGRATE_CMD": f"{bin_dir}/hook migrate",
            "FAKE_LOG": str(self.log),
        }

    def run(self, script: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", f"scripts/{script}", *args],
            cwd=self.root, env={**self.env, **env}, capture_output=True, text=True, timeout=60,
        )  # fmt: skip

    def calls(self) -> list[str]:
        return [line.split()[0] for line in self.log.read_text().splitlines()]

    def dumps(self, prefix: str = "kartly") -> list[Path]:
        return sorted((self.root / "backups").glob(f"{prefix}-*.dump"))


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


# ------------------------------------------------------------------------------ backup


def test_a_backup_is_private_named_by_time_and_reported(box: Sandbox) -> None:
    done = box.run("backup.sh")
    assert done.returncode == 0, done.stderr
    [dump] = box.dumps()
    assert done.stdout.strip() == f"backups/{dump.name}"
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
    assert stat.S_IMODE((box.root / "backups").stat().st_mode) == 0o700
    assert "--format=custom" in box.log.read_text() and "--no-owner" in box.log.read_text()
    assert not list((box.root / "backups").glob(".*"))  # nothing half-written is left


def test_only_the_newest_backups_are_kept(box: Sandbox) -> None:
    backups = box.root / "backups"
    backups.mkdir()
    for day in range(1, 6):
        (backups / f"kartly-2026110{day}-120000.dump").write_text("old")
    (backups / "pre-restore-20260101-000000.dump").write_text("safety")
    done = box.run("backup.sh", KARTLY_KEEP="3")
    assert done.returncode == 0
    newest = Path(done.stdout.strip()).name
    assert [p.name for p in box.dumps()] == [
        "kartly-20261104-120000.dump",
        "kartly-20261105-120000.dump",
        newest,
    ]
    assert (backups / "pre-restore-20260101-000000.dump").exists()  # safety copies are never pruned


def test_keeping_zero_or_junk_is_refused_before_anything_is_dumped(box: Sandbox) -> None:
    for bad in ("0", "-1", "abc", "1.5"):
        done = box.run("backup.sh", KARTLY_KEEP=bad)
        assert done.returncode == 2, bad
    assert box.dumps() == [] and box.log.read_text() == ""


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"FAKE_EMPTY_DUMP": "1"}, "dump is empty"),
        ({"FAKE_BAD_DUMP": "1"}, "could not be read back"),
        ({"FAKE_DUMP_FAILS": "1"}, "connection refused"),
    ],
)
def test_a_bad_dump_makes_no_backup_and_leaves_no_file_behind(
    box: Sandbox, env: dict[str, str], message: str
) -> None:
    done = box.run("backup.sh", **env)
    assert done.returncode != 0 and message in done.stderr
    assert box.dumps() == []
    assert list((box.root / "backups").glob(".*")) == []


def test_a_backup_made_in_the_same_second_never_replaces_another(box: Sandbox) -> None:
    first = box.run("backup.sh").stdout.strip()
    second = box.run("backup.sh").stdout.strip()
    third = box.run("backup.sh").stdout.strip()
    assert len({first, second, third}) == 3 and len(box.dumps()) == 3


# ------------------------------------------------------------------------------ restore


@pytest.fixture
def dump(box: Sandbox) -> Path:
    path = box.root / "good.dump"
    path.write_text("PGDMP-fake")
    return path


@pytest.mark.parametrize("args", [[], ["x.dump"], ["x.dump", "yes"], ["x.dump", "--YES"]])
def test_a_restore_that_was_not_confirmed_does_nothing(box: Sandbox, args: list[str]) -> None:
    (box.root / "x.dump").write_text("PGDMP")
    done = box.run("restore.sh", *args)
    assert done.returncode == 2 and "--yes" in done.stderr
    assert box.calls() == []


def test_a_missing_empty_or_unreadable_file_is_refused_before_touching_anything(
    box: Sandbox,
) -> None:
    (box.root / "empty.dump").write_text("")
    (box.root / "junk.dump").write_text("not a dump")
    for args, env, message in (
        (("nope.dump", "--yes"), {}, "no such file"),
        (("empty.dump", "--yes"), {}, "is empty"),
        (("junk.dump", "--yes"), {"FAKE_BAD_DUMP": "1"}, "not a readable backup"),
    ):
        done = box.run("restore.sh", *args, **env)
        assert done.returncode == 2 and message in done.stderr, args
    assert box.calls() == [] and not (box.root / "backups").exists()


def test_a_restore_stops_saves_restores_migrates_and_only_then_starts(
    box: Sandbox, dump: Path
) -> None:
    done = box.run("restore.sh", str(dump), "--yes")
    assert done.returncode == 0, done.stderr
    assert box.calls() == ["stop", "pg_dump", "restore", "migrate", "start"]
    [safety] = box.dumps("pre-restore")
    assert f"saved as backups/{safety.name}" in done.stdout
    assert "--clean" in box.log.read_text() and "--exit-on-error" in box.log.read_text()
    assert dump.exists()  # the file being restored is never touched


def test_restoring_a_file_that_lives_in_backups_survives_its_own_safety_copy(
    box: Sandbox,
) -> None:
    box.run("backup.sh")
    [source] = box.dumps()
    done = box.run("restore.sh", f"backups/{source.name}", "--yes")
    assert done.returncode == 0, done.stderr
    assert source.exists() and len(box.dumps("pre-restore")) == 1


def test_a_failed_restore_leaves_the_services_stopped_and_says_where_the_old_data_is(
    box: Sandbox, dump: Path
) -> None:
    done = box.run("restore.sh", str(dump), "--yes", FAKE_RESTORE_FAILS="1")
    assert done.returncode != 0
    assert box.calls() == ["stop", "pg_dump", "restore"]  # no migrate, no start
    [safety] = box.dumps("pre-restore")
    assert "RESTORE FAILED" in done.stderr and safety.name in done.stderr


def test_a_failed_safety_backup_stops_the_restore_before_it_overwrites_anything(
    box: Sandbox, dump: Path
) -> None:
    done = box.run("restore.sh", str(dump), "--yes", FAKE_EMPTY_DUMP="1")
    assert done.returncode != 0
    assert "restore" not in box.calls() and "start" not in box.calls()


def test_the_scripts_are_strict_and_executable() -> None:
    for name in ("backup.sh", "restore.sh"):
        path = ROOT / "scripts" / name
        assert os.access(path, os.X_OK), name
        assert "set -euo pipefail" in path.read_text(), name
