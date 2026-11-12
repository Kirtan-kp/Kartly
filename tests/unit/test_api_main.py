"""``create_app_from_env``: the real wiring (settings, model registry, client, database) over a
real HTTP model server, driven through the API."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import sessionmaker

from kartly.api.main import create_app_from_env, main
from kartly.config import Settings
from kartly.llm import LLMConfigError
from kartly.llm.client import OpenAICompatClient
from kartly.store.models import AuditLog
from tests.llm_support import FakeOpenAIServer, write_models_config
from tests.support import make_sqlite_engine, seeded_session_factory

KEY = "env-api-key-123456"


def make_settings(**values: object) -> Settings:
    """Settings that ignore the real .env file and environment file lookups."""
    return Settings(_env_file=None, **values)  # type: ignore[call-arg,arg-type]


@pytest.fixture(autouse=True)
def _no_kartly_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own KARTLY_* settings (in .env or the shell: default model, claim check,
    tier modes, ...) must not leak in, so every one is removed."""
    for name in [n for n in os.environ if n.startswith("KARTLY_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def wired(
    monkeypatch: pytest.MonkeyPatch, fake_server: FakeOpenAIServer, tmp_path: Path
) -> Iterator[Engine]:
    engine = make_sqlite_engine()
    seeded_session_factory(engine)
    config = write_models_config(tmp_path, fake_server.base_url)

    def settings(**overrides: object) -> Settings:
        base: dict[str, object] = {
            "models_config_path": str(config),
            "api_key": KEY,
            "max_workers": 2,
            "max_pending_runs": 5,
        }
        return make_settings(**{**base, **overrides})

    monkeypatch.setattr("kartly.api.main.get_settings", settings)
    monkeypatch.setattr("kartly.api.main.load_env", lambda: None)
    monkeypatch.setattr("kartly.api.main.make_engine", lambda: engine)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield engine
    engine.dispose()


def test_a_ticket_goes_through_http_the_model_the_gateway_and_the_database(
    wired: Engine, fake_server: FakeOpenAIServer
) -> None:
    app = create_app_from_env()
    with TestClient(app) as http:
        response = http.post(
            "/v1/tickets?wait=true",
            json={"customer_id": "C-0001", "message": "Where is KRT-100002?"},
            headers={"X-API-Key": KEY},
        )
        body = response.json()
        assert response.status_code == 200, body
        assert body["outcome"] == "drafted"
        assert body["answer"] == "Your order KRT-100002 is in transit."

        trace = http.get(f"/v1/runs/{body['run_id']}/trace", headers={"X-API-Key": KEY}).json()
        assert trace["run"]["model_tier"] == "fake_small"  # first model of role "small"
        assert trace["steps"][1]["output"]["data"]["order_id"] == "KRT-100002"

        # Read inside the block: shutting the app down disposes the (in-memory) engine.
        with sessionmaker(wired)() as session:
            audit = session.scalars(select(AuditLog)).one()
        assert audit.actor_type.value == "agent"

    assert len(fake_server.requests) == 2


def test_a_router_model_from_the_settings_labels_each_ticket_and_is_closed_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(
            models_config_path=str(config), api_key=KEY, router_model="fake_small"
        ),
    )
    closed: list[str] = []
    real_close = OpenAICompatClient.close

    def close_client(self: OpenAICompatClient) -> None:
        closed.append(self.name)
        real_close(self)

    monkeypatch.setattr(OpenAICompatClient, "close", close_client)
    with TestClient(create_app_from_env()) as http:
        body = http.post(
            "/v1/tickets?wait=true",
            json={"customer_id": "C-0001", "message": "Where is KRT-100002?"},
            headers={"X-API-Key": KEY},
        ).json()
        trace = http.get(f"/v1/runs/{body['run_id']}/trace", headers={"X-API-Key": KEY}).json()
    # The first request is the label call (JSON mode); the fake model's reply is not a label,
    # so the ticket carried on and was answered, with the failed label recorded in the trace.
    assert fake_server.requests[0]["body"]["response_format"] == {"type": "json_object"}
    assert [s["name"] for s in trace["steps"]][:2] == ["classify", "chat"]
    assert trace["steps"][0]["output"]["failed"] is True and body["outcome"] == "drafted"
    assert closed.count("fake_small") == 2


def test_no_router_unless_configured(wired: Engine, fake_server: FakeOpenAIServer) -> None:
    with TestClient(create_app_from_env()) as http:
        http.post(
            "/v1/tickets?wait=true",
            json={"customer_id": "C-0001", "message": "hi"},
            headers={"X-API-Key": KEY},
        )
    assert "response_format" not in fake_server.requests[0]["body"]


def test_the_worker_limits_come_from_the_settings(wired: Engine) -> None:
    executor = create_app_from_env().state.kartly.executor
    assert (executor.max_workers, executor.max_pending) == (2, 5)


def test_stopping_the_app_closes_the_model_client_and_the_database(
    monkeypatch: pytest.MonkeyPatch, wired: Engine
) -> None:
    closed: list[str] = []
    real_close = OpenAICompatClient.close

    def close_client(self: OpenAICompatClient) -> None:
        closed.append("client")
        real_close(self)

    def dispose_engine(*_args: object, **_kwargs: object) -> None:
        closed.append("engine")

    monkeypatch.setattr(OpenAICompatClient, "close", close_client)
    monkeypatch.setattr(wired, "dispose", dispose_engine)
    with TestClient(create_app_from_env()):
        assert closed == []
    assert closed == ["client", "engine"]


def test_the_model_can_be_chosen_by_name(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, tmp_path: Path, fake_server: FakeOpenAIServer
) -> None:
    base = write_models_config(tmp_path, fake_server.base_url).read_text()
    extra = (
        "  fake_large:\n    provider: fake\n    model: other-id\n"
        "    price: {input_per_mtok: 1, output_per_mtok: 2}\n"
    )
    config = tmp_path / "two.yaml"
    config.write_text(base.replace("roles:", extra + "roles:"))
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(
            models_config_path=str(config), api_key=KEY, default_model="fake_large"
        ),
    )
    with TestClient(create_app_from_env()) as http:
        body = http.post(
            "/v1/tickets?wait=true",
            json={"customer_id": "C-0001", "message": "hi"},
            headers={"X-API-Key": KEY},
        ).json()
    assert fake_server.requests[0]["body"]["model"] == "other-id"
    assert body["run_id"] is not None


def test_startup_fails_clearly_without_an_api_key(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, tmp_path: Path, fake_server: FakeOpenAIServer
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(models_config_path=str(config), api_key=None),
    )
    with pytest.raises(ValueError, match="KARTLY_API_KEY is not set.*token_urlsafe"):
        create_app_from_env()


def test_startup_fails_clearly_without_a_model_key(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, tmp_path: Path, fake_server: FakeOpenAIServer
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url, key_env="FAKE_KEY")
    monkeypatch.delenv("FAKE_KEY", raising=False)
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(models_config_path=str(config), api_key=KEY),
    )
    with pytest.raises(LLMConfigError, match="FAKE_KEY"):
        create_app_from_env()


def test_the_launcher_prints_setup_problems_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("kartly.api.main.load_env", lambda: None)
    monkeypatch.setattr("kartly.api.main.get_settings", lambda: make_settings(api_key=None))
    assert main([]) == 2
    assert "KARTLY_API_KEY" in capsys.readouterr().out


def test_the_context_budget_setting_reaches_the_runner(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(
            models_config_path=str(config), api_key=KEY, context_budget_tokens=1234
        ),
    )
    with TestClient(create_app_from_env()) as http:
        runner = http.app.state.kartly.runner  # type: ignore[attr-defined]
        assert runner._context._budget == 1234


def test_a_claim_check_model_from_the_settings_reaches_the_runner_and_is_closed_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(
            models_config_path=str(config), api_key=KEY, claim_check_model="fake_small"
        ),
    )
    closed: list[str] = []
    real_close = OpenAICompatClient.close

    def close_client(self: OpenAICompatClient) -> None:
        closed.append(self.name)
        real_close(self)

    monkeypatch.setattr(OpenAICompatClient, "close", close_client)
    with TestClient(create_app_from_env()) as http:
        judge = http.app.state.kartly.runner._claim_judge  # type: ignore[attr-defined]
        assert judge is not None and judge.name == "fake_small"
    assert closed.count("fake_small") == 2


def test_no_claim_check_unless_configured(wired: Engine, fake_server: FakeOpenAIServer) -> None:
    with TestClient(create_app_from_env()) as http:
        assert http.app.state.kartly.runner._claim_judge is None  # type: ignore[attr-defined]


def test_a_down_default_model_falls_back_to_the_next_one_from_the_settings(
    monkeypatch: pytest.MonkeyPatch, wired: Engine, tmp_path: Path, fake_server: FakeOpenAIServer
) -> None:
    from tests.llm_support import completion_payload

    base = write_models_config(tmp_path, fake_server.base_url).read_text()
    extra = (
        "  fake_large:\n    provider: fake\n    model: other-id\n"
        "    price: {input_per_mtok: 1, output_per_mtok: 2}\n"
    )
    config = tmp_path / "two.yaml"
    config.write_text(base.replace("roles:", extra + "roles:"))
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(
            models_config_path=str(config),
            api_key=KEY,
            fallback_models=" fake_large , ",
            llm_max_attempts=1,
        ),
    )
    fake_server.responder = lambda body: (
        (503, {"error": {"message": "overloaded"}}, {})
        if body["model"] == "fake-model-id"
        else (200, completion_payload("Sorry for the wait, it is on its way."), {})
    )
    with TestClient(create_app_from_env()) as http:
        body = http.post(
            "/v1/tickets?wait=true",
            json={"customer_id": "C-0001", "message": "hi"},
            headers={"X-API-Key": KEY},
        ).json()
        trace = http.get(f"/v1/runs/{body['run_id']}/trace", headers={"X-API-Key": KEY}).json()
    assert body["outcome"] == "drafted" and "on its way" in body["answer"]
    assert [r["body"]["model"] for r in fake_server.requests] == ["fake-model-id", "other-id"]
    [call] = [s for s in trace["steps"] if s["kind"] == "llm_call"]
    assert call["model"] == "fake_large"
    assert call["output"]["failed_over_from"] == ["fake_small"]


def test_queue_mode_db_turns_tickets_into_jobs_and_a_worker_can_run_them(
    wired: Engine, fake_server: FakeOpenAIServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kartly.api.main import build_runtime
    from kartly.store.models import Job, JobStatus
    from kartly.worker import Worker

    config = write_models_config(tmp_path, fake_server.base_url)
    monkeypatch.setattr(
        "kartly.api.main.get_settings",
        lambda: make_settings(
            models_config_path=str(config),
            api_key=KEY,
            queue_mode="db",
            embedded_workers=0,
            job_max_attempts=2,
        ),
    )
    app = create_app_from_env()
    with TestClient(app) as http:
        created = http.post(
            "/v1/tickets",
            json={"customer_id": "C-0001", "message": "Where is my order KRT-100001?"},
            headers={"X-API-Key": KEY},
        )
        assert created.status_code == 202
        factory = sessionmaker(wired)
        with factory() as session:
            [job] = session.scalars(select(Job)).all()
            assert job.status == JobStatus.QUEUED and job.max_attempts == 2

        runtime = build_runtime()  # what `python -m kartly.worker` builds (not closed: the
        # in-memory test engine is shared and closing would dispose it)
        assert Worker(runtime.queue, runtime.runner.run, runtime.factory).run_once()
        with factory() as session:
            [done] = session.scalars(select(Job)).all()
            assert done.status == JobStatus.SUCCEEDED and done.result is not None
            assert done.result["outcome"] == "drafted"
