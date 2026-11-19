"""The Dockerfile, compose file and smoke script keep the properties a deployment depends on."""

from __future__ import annotations

import os
import re
from typing import Any

import pytest
import yaml

from kartly.worker import WorkerPool
from tests.support import ROOT

DOCKERFILE = (ROOT / "Dockerfile").read_text()
IGNORE = (ROOT / ".dockerignore").read_text().splitlines()
EXAMPLE = (ROOT / ".env.deploy.example").read_text()


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded = yaml.safe_load((ROOT / "docker-compose.deploy.yml").read_text())
    assert isinstance(loaded, dict)
    return loaded


def app_services(compose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n: compose["services"][n] for n in ("migrate", "api", "worker")}


# ------------------------------------------------------------------------------ the image


def test_the_image_is_built_in_two_stages_from_a_tagged_base() -> None:
    bases = re.findall(r"^FROM (\S+)", DOCKERFILE, flags=re.M)
    assert len(bases) == 2
    for base in bases:
        assert ":" in base and not base.endswith(":latest"), base
    assert len(set(bases)) == 1  # build and run on the same Python


def test_the_image_runs_as_a_numeric_non_root_user() -> None:
    users = re.findall(r"^USER (\S+)", DOCKERFILE, flags=re.M)
    assert users and users[-1] == "10001:10001"


def test_the_image_carries_no_secrets_and_downloads_nothing_unchecked() -> None:
    assert not re.search(r"^(COPY|ADD) .*\.env", DOCKERFILE, flags=re.M)
    assert not re.search(r"^ADD ", DOCKERFILE, flags=re.M)
    assert "curl" not in DOCKERFILE and "wget" not in DOCKERFILE
    assert not re.search(r"(password|secret|api_key)\s*=", DOCKERFILE, flags=re.I)


def test_the_image_has_what_the_app_reads_at_run_time() -> None:
    assert "COPY config ./config" in DOCKERFILE and "COPY kb ./kb" in DOCKERFILE
    assert (ROOT / "config" / "models.yaml").is_file() and (ROOT / "kb").is_dir()
    assert re.search(r"^CMD \[.*kartly\.api.*0\.0\.0\.0", DOCKERFILE, flags=re.M)


def test_the_build_context_leaves_out_secrets_and_what_the_image_does_not_run() -> None:
    for entry in (".git", ".venv", ".env", ".env.*", "tests", "**/__pycache__", "*.db"):
        assert entry in IGNORE, entry
    assert "!.env.deploy.example" in IGNORE


# ------------------------------------------------------------------------------ the stack


def test_there_is_a_database_a_one_off_migration_the_api_and_a_worker(
    compose: dict[str, Any],
) -> None:
    assert set(compose["services"]) == {"postgres", "migrate", "api", "worker"}


def test_nothing_starts_before_the_schema_is_current(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    for name in ("api", "worker"):
        assert (
            services[name]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
        )
    assert services["migrate"]["restart"] == "no"
    assert "kartly.migrate" in services["migrate"]["command"]


def test_only_the_api_is_reachable_and_only_from_this_machine(compose: dict[str, Any]) -> None:
    services = compose["services"]
    for name in ("postgres", "migrate", "worker"):
        assert "ports" not in services[name], name
    [port] = services["api"]["ports"]
    assert port.startswith("127.0.0.1:")


def test_every_container_that_runs_our_code_is_locked_down(compose: dict[str, Any]) -> None:
    for name, svc in app_services(compose).items():
        assert svc["read_only"] is True, name
        assert svc["cap_drop"] == ["ALL"], name
        assert "no-new-privileges:true" in svc["security_opt"], name
        assert svc["init"] is True, name
        assert "/tmp" in svc["tmpfs"], name
        assert svc["logging"]["options"]["max-size"], name


def test_secrets_are_required_and_never_written_into_the_file(compose: dict[str, Any]) -> None:
    text = (ROOT / "docker-compose.deploy.yml").read_text()
    assert "${POSTGRES_PASSWORD:?" in text and "${KARTLY_API_KEY:?" in text
    env = compose["services"]["api"]["environment"]
    assert env["KARTLY_API_KEY"].startswith("${KARTLY_API_KEY:?")
    assert "${POSTGRES_PASSWORD:?" in env["KARTLY_DATABASE_URL"]
    assert compose["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"].startswith(
        "${POSTGRES_PASSWORD:?"
    )
    assert "@postgres:5432/kartly" in env["KARTLY_DATABASE_URL"]
    assert not re.search(r"(PASSWORD|API_KEY): [^$\s\"']", text)


def test_the_services_use_the_durable_queue_and_the_api_runs_no_workers_of_its_own(
    compose: dict[str, Any],
) -> None:
    for name, svc in app_services(compose).items():
        assert svc["environment"]["KARTLY_QUEUE_MODE"] == "db", name
        assert svc["environment"]["KARTLY_EMBEDDED_WORKERS"] == "0", name


def test_the_api_has_a_healthcheck_that_fails_when_the_database_is_down(
    compose: dict[str, Any],
) -> None:
    check = compose["services"]["api"]["healthcheck"]
    assert "/healthz" in " ".join(check["test"]) and check["start_period"]
    assert "healthcheck" not in compose["services"]["worker"]  # it has no HTTP port to ask


def test_the_worker_is_given_longer_to_stop_than_it_takes_to_drain(compose: dict[str, Any]) -> None:
    grace = compose["services"]["worker"]["stop_grace_period"]
    seconds = int(str(grace).rstrip("s"))
    assert seconds > WorkerPool.stop.__defaults__[0]  # type: ignore[index]


def test_images_are_tagged_and_the_database_keeps_its_data_in_a_volume(
    compose: dict[str, Any],
) -> None:
    pg = compose["services"]["postgres"]
    assert ":" in pg["image"] and not pg["image"].endswith(":latest")
    assert "pgdata" in compose["volumes"] and any("pgdata:" in v for v in pg["volumes"])


def test_every_command_names_a_module_that_exists(compose: dict[str, Any]) -> None:
    for name, svc in app_services(compose).items():
        module = svc["command"][2]
        base = ROOT / "src" / module.replace(".", "/")
        assert base.with_suffix(".py").exists() or (base / "__main__.py").exists(), (name, module)


# ------------------------------------------------------------------------------ the settings


def test_every_variable_the_compose_file_needs_is_in_the_example() -> None:
    text = (ROOT / "docker-compose.deploy.yml").read_text()
    needed = set(re.findall(r"\$\{([A-Z_]+)[:?-]", text))
    shown = set(re.findall(r"^#?\s*([A-Z_]+)=", EXAMPLE, flags=re.M))
    assert needed <= shown, needed - shown


def test_the_example_contains_no_real_secrets() -> None:
    values = dict(re.findall(r"^([A-Z_]+)=(.*)$", EXAMPLE, flags=re.M))
    assert values["POSTGRES_PASSWORD"] == "" and values["KARTLY_API_KEY"] == ""
    assert values["GROQ_API_KEY"] == ""


def test_the_example_does_not_set_values_that_break_when_empty() -> None:
    """An empty KARTLY_* option reaches the app as an empty string, not as 'unset'."""
    for name, value in re.findall(r"^(KARTLY_[A-Z_]+)=(.*)$", EXAMPLE, flags=re.M):
        if name != "KARTLY_API_KEY":
            assert value != "", name


def test_the_deployment_secrets_file_is_ignored_by_git() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env.deploy" in ignored


# ------------------------------------------------------------------------------ the smoke script


def test_the_smoke_script_is_strict_executable_and_cleans_up() -> None:
    path = ROOT / "scripts" / "compose-smoke.sh"
    text = path.read_text()
    assert os.access(path, os.X_OK)
    assert "set -euo pipefail" in text and "trap cleanup EXIT" in text and "down -v" in text
    assert "docker-compose.deploy.yml" in text


def test_the_smoke_script_checks_what_the_stack_promises() -> None:
    text = (ROOT / "scripts" / "compose-smoke.sh").read_text()
    for promise in (
        "/healthz",
        "migrate --check",
        "/admin",
        "10001",
        "should-not-work",
        "stop worker",
    ):
        assert promise in text, promise
