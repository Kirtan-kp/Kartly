"""The CI workflow keeps the properties that make it safe and honest."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support import ROOT

WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def ci() -> dict[str, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(loaded, dict)
    loaded["triggers"] = loaded.pop(True, loaded.pop("on", None))  # YAML reads a bare `on` as True
    return loaded


def steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job["steps"])


def commands(job: dict[str, Any]) -> str:
    return "\n".join(str(s.get("run", "")) for s in steps(job))


def test_the_workflow_reads_the_repository_and_nothing_more(ci: dict[str, Any]) -> None:
    assert ci["permissions"] == {"contents": "read"}
    for name, job in ci["jobs"].items():
        assert "permissions" not in job or job["permissions"] == {"contents": "read"}, name


def test_it_runs_on_pushes_to_main_and_on_pull_requests(ci: dict[str, Any]) -> None:
    assert set(ci["triggers"]) >= {"push", "pull_request"}
    assert ci["triggers"]["push"]["branches"] == ["main"]


def test_every_job_has_a_time_limit(ci: dict[str, Any]) -> None:
    for name, job in ci["jobs"].items():
        assert 0 < job["timeout-minutes"] <= 30, name


def test_every_action_is_pinned_to_a_released_version_or_a_commit(ci: dict[str, Any]) -> None:
    for name, job in ci["jobs"].items():
        for step in steps(job):
            if "uses" in step:
                assert re.fullmatch(r"[\w./-]+@(v\d+(\.\d+){0,2}|[0-9a-f]{40})", step["uses"]), (
                    name,
                    step["uses"],
                )


def test_the_oldest_supported_python_is_tested(ci: dict[str, Any]) -> None:
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    oldest = spec.removeprefix(">=")
    assert oldest in ci["jobs"]["test"]["strategy"]["matrix"]["python-version"]


def test_the_checks_a_change_must_pass_are_all_present(ci: dict[str, Any]) -> None:
    assert "ruff check ." in commands(ci["jobs"]["lint"])
    assert "ruff format --check ." in commands(ci["jobs"]["lint"])
    assert "mypy" in commands(ci["jobs"]["types"])
    assert "pytest" in commands(ci["jobs"]["test"])


def test_the_postgres_job_really_runs_the_postgres_tests(ci: dict[str, Any]) -> None:
    job = ci["jobs"]["postgres"]
    url = job["env"]["KARTLY_TEST_DATABASE_URL"]
    assert url.startswith("postgresql+psycopg://") and url.endswith("/kartly_test")
    assert "postgres" in job["services"]
    assert "pytest" in commands(job) and "not postgres" not in commands(job)
    for cmd in (
        "kartly.migrate",
        "kartly.migrate --check",
        "alembic check",
        "kartly.alerts",
        "kartly.costs",
    ):
        assert cmd in commands(job), cmd


def test_the_browser_job_installs_a_browser_and_the_others_skip_those_tests(
    ci: dict[str, Any],
) -> None:
    job = ci["jobs"]["browser"]
    assert "playwright install" in commands(job) and "tests/browser" in commands(job)
    for name in ("test", "postgres"):
        assert "not browser" in commands(ci["jobs"][name]), name


def test_the_container_job_runs_the_smoke_script(ci: dict[str, Any]) -> None:
    assert "scripts/compose-smoke.sh" in commands(ci["jobs"]["container"])
    assert (ROOT / "scripts" / "compose-smoke.sh").is_file()


def test_the_audit_cannot_block_a_change_but_the_checks_can(ci: dict[str, Any]) -> None:
    assert ci["jobs"]["audit"]["continue-on-error"] is True
    for name, job in ci["jobs"].items():
        if name != "audit":
            assert not job.get("continue-on-error"), name


def test_dependabot_watches_python_and_the_actions_themselves() -> None:
    config = yaml.safe_load((Path(WORKFLOW).parents[1] / "dependabot.yml").read_text())
    assert {u["package-ecosystem"] for u in config["updates"]} == {"pip", "github-actions"}


def test_every_command_the_workflow_runs_exists_in_the_project() -> None:
    text = WORKFLOW.read_text()
    for module in re.findall(r"python -m (kartly\.\w+)", text):
        assert (ROOT / "src" / (module.replace(".", "/") + ".py")).exists(), module
