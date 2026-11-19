"""A worker stops cleanly when asked to (docker stop sends SIGTERM)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from kartly.store.models import Base
from kartly.worker import main
from tests.support import ROOT


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    db = tmp_path / "worker.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    engine.dispose()
    clean = {k: v for k, v in os.environ.items() if not k.startswith("KARTLY_")}
    return {
        **clean,
        "KARTLY_DATABASE_URL": f"sqlite:///{db}",
        "KARTLY_API_KEY": "test-api-key-123456",
        "KARTLY_QUEUE_MODE": "db",
        "KARTLY_ALERTS_ENABLED": "false",
        "PYTHONUNBUFFERED": "1",
    }


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows has no SIGTERM/SIGINT delivery to a child; this is the Linux container's stop",
)
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_worker_process_finishes_and_exits_cleanly_on_a_stop_signal(
    env: dict[str, str], sig: signal.Signals
) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "kartly.worker", "--workers", "1"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        assert proc.stdout is not None
        started = proc.stdout.readline()
        assert "worker(s) running" in started and "SIGTERM" in started, started
        proc.send_signal(sig)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 0, out
    assert "stopping: finishing the jobs in hand" in out


def test_main_stops_when_its_stop_event_is_set(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for key, value in env.items():
        if key.startswith("KARTLY_"):
            monkeypatch.setenv(key, value)
    monkeypatch.chdir(ROOT)
    from kartly.config import get_settings

    get_settings.cache_clear()
    from kartly.worker import WorkerPool

    drained: list[bool] = []
    real_stop = WorkerPool.stop

    def spy(self: WorkerPool, timeout_s: float = 150.0) -> None:
        drained.append(True)
        real_stop(self, timeout_s)

    monkeypatch.setattr(WorkerPool, "stop", spy)
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    started = time.monotonic()
    try:
        assert main(["--workers", "1"], stop=stop) == 0
    finally:
        get_settings.cache_clear()
    assert time.monotonic() - started < 20
    assert drained == [True]  # it waited for the jobs in hand before exiting
    assert "stopping" in capsys.readouterr().out
