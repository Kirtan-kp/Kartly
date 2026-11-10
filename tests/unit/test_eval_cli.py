"""``python -m kartly.evals`` over a real HTTP model server (no mocks of the client)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from kartly.evals.cli import main
from kartly.evals.runner import run_suite
from tests.llm_support import FakeOpenAIServer, completion_payload, write_models_config

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "golden"


def good_tracking_model(body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Looks the order up, then answers the way the 'tracking-in-transit' case wants."""
    if any(m.get("role") == "tool" for m in body["messages"]):
        text = "Your order KRT-100002 is in transit and should arrive on 25 Oct."
        return 200, completion_payload(text), {}
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_tracking", "arguments": '{"order_id": "KRT-100002"}'},
    }
    return 200, completion_payload(None, [call]), {}


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kartly.evals.cli.load_env", lambda: None)  # never read the real .env
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("FAKE_KEY", raising=False)


def args(config: Path, tmp_path: Path, *more: str) -> list[str]:
    return [
        "--model",
        "fake_small",
        "--config",
        str(config),
        "--out",
        str(tmp_path / "runs"),
        *more,
    ]


def test_list_shows_the_cases_without_needing_a_model(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 29 and "idor-friends-order" in out and "tier 2" in out


def test_a_passing_run_prints_saves_and_exits_zero(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    code = main(args(config, tmp_path, "--only", "tracking-in-transit", "--note", "cli test"))
    out = capsys.readouterr().out
    assert code == 0
    assert "[PASS] tracking-in-transit" in out and "passed 1/1" in out
    assert "safety failures: 0" in out and "Saved " in out

    [saved] = (tmp_path / "runs").glob("*_fake_small.json")
    assert json.loads(saved.read_text())["note"] == "cli test"
    [row] = list(csv.DictReader((tmp_path / "runs" / "experiments.csv").open(encoding="utf-8")))
    assert (
        row["model"] == "fake_small" and row["pass_rate"] == "1.000" and row["note"] == "cli test"
    )
    assert len(fake_server.requests) == 2  # one tool call, one answer


def test_a_failing_case_exits_one_and_says_which_check_failed(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = lambda body: (200, completion_payload("I have no idea."), {})
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(args(config, tmp_path, "--only", "tracking-in-transit")) == 1
    out = capsys.readouterr().out
    assert "[FAIL] tracking-in-transit" in out
    assert "quality: call: get_tracking order_id=KRT-100002" in out
    assert "reply: 'I have no idea.'" in out


def test_no_save_writes_nothing(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save")) == 0
    assert not (tmp_path / "runs").exists() and "Saved" not in capsys.readouterr().out


def test_show_replies_prints_passing_replies_too(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save", "--show-replies"))
    assert "should arrive on 25 Oct" in capsys.readouterr().out


def test_repeat_runs_the_case_again(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(args(config, tmp_path, "--only", "tracking-in-transit", "--repeat", "2")) == 0
    out = capsys.readouterr().out
    assert "tracking-in-transit #2" in out and "passed 2/2" in out


def test_a_provider_error_is_reported_as_errored_runs(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = lambda body: (401, {"error": {"message": "Invalid API Key"}}, {})
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save")) == 1
    out = capsys.readouterr().out
    assert "[ERROR] tracking-in-transit" in out and "AuthenticationError" in out
    assert "ERRORED" in out and "passed 0/0" in out


def test_a_missing_model_key_names_the_variable_and_runs_nothing(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url, key_env="FAKE_KEY")
    assert main(args(config, tmp_path)) == 2
    assert "FAKE_KEY" in capsys.readouterr().err
    assert fake_server.requests == []


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--only", "nope"], "unknown case id(s): nope"),
        (["--repeat", "0"], "--repeat must be at least 1"),
        (["--pause", "-1"], "--pause cannot be negative"),
        (["--cases", "missing.yaml"], "cannot read"),
    ],
)
def test_bad_options_exit_two_with_a_reason(
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    message: str,
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(args(config, tmp_path, *extra)) == 2
    assert message in capsys.readouterr().err
    assert fake_server.requests == []


def test_a_model_name_is_required_to_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "--model is required" in capsys.readouterr().err


def test_an_unknown_model_name_lists_the_known_ones(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(["--model", "nope", "--config", str(config)]) == 2
    assert "fake_small" in capsys.readouterr().err


def test_options_reach_the_runner(
    monkeypatch: pytest.MonkeyPatch, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def spy(cases: Any, client_for: Any, **kwargs: Any) -> Any:
        seen.update(kwargs, cases=[c.id for c in cases])
        return run_suite(cases, client_for, **{**kwargs, "pause_s": 0.0})  # do not really wait

    monkeypatch.setattr("kartly.evals.cli.run_suite", spy)
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    options = ["--only", "tracking-in-transit", "--repeat", "2", "--pause", "9.5"]
    main(args(config, tmp_path, *options))
    assert seen["cases"] == ["tracking-in-transit"]
    assert (seen["model_name"], seen["repeat"], seen["pause_s"]) == ("fake_small", 2, 9.5)
    assert seen["screen"] is True

    main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-screen", "--no-save"))
    assert seen["screen"] is False


def test_the_model_client_is_closed_afterwards(
    monkeypatch: pytest.MonkeyPatch, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    from kartly.llm import OpenAICompatClient

    closed: list[str] = []
    real_close = OpenAICompatClient.close

    def close(self: OpenAICompatClient) -> None:
        closed.append(self.name)
        real_close(self)

    monkeypatch.setattr(OpenAICompatClient, "close", close)
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save"))
    assert closed == ["fake_small"]


def test_a_router_model_is_passed_to_the_runner_closed_afterwards_and_noted(
    monkeypatch: pytest.MonkeyPatch,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
) -> None:
    from kartly.llm import OpenAICompatClient

    seen: dict[str, Any] = {}
    closed: list[str] = []
    real_close = OpenAICompatClient.close

    def spy(cases: Any, client_for: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return run_suite(cases, client_for, **kwargs)

    def close(self: OpenAICompatClient) -> None:
        closed.append(self.name)
        real_close(self)

    monkeypatch.setattr("kartly.evals.cli.run_suite", spy)
    monkeypatch.setattr(OpenAICompatClient, "close", close)
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    code = main(args(config, tmp_path, "--only", "tracking-in-transit", "--router", "fake_small"))
    assert seen["router"] is not None and seen["router"].name == "fake_small"
    assert closed.count("fake_small") == 2  # the model and the router
    [row] = list(csv.DictReader((tmp_path / "runs" / "experiments.csv").open(encoding="utf-8")))
    assert row["note"] == "router=fake_small"
    assert code in (0, 1)  # the fake router's reply is not a label: the run carries on


def test_no_router_by_default(
    monkeypatch: pytest.MonkeyPatch, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def spy(cases: Any, client_for: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return run_suite(cases, client_for, **kwargs)

    monkeypatch.setattr("kartly.evals.cli.run_suite", spy)
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save"))
    assert seen["router"] is None


def test_kb_gives_the_runner_the_policy_tool_and_is_noted(
    monkeypatch: pytest.MonkeyPatch,
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
) -> None:
    from kartly.tools import Tool

    seen: dict[str, Any] = {}

    def spy(cases: Any, client_for: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return run_suite(cases, client_for, **{**kwargs, "pause_s": 0.0})

    class Unit:
        name = "unit"

        def embed_documents(self, texts: Any) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr("kartly.evals.cli.run_suite", spy)
    monkeypatch.setattr("kartly.retrieval.setup.OllamaEmbedder", lambda *a, **k: Unit())
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    kb = str(Path(__file__).resolve().parents[2] / "kb")
    main(args(config, tmp_path, "--only", "tracking-in-transit", "--kb", kb))
    tools: list[Tool] = list(seen["tools"])
    assert "search_policy" in [t.name for t in tools] and "get_order" in [t.name for t in tools]
    log = (tmp_path / "runs" / "experiments.csv").read_text(encoding="utf-8")
    assert "kb=" in log and "v1+policy-p6" in log

    assert seen["policy_prompt"] == "p6"
    main(
        args(
            config,
            tmp_path,
            "--only",
            "tracking-in-transit",
            "--kb",
            kb,
            "--policy-prompt",
            "p2",
            "--note",
            "p2 try",
        )  # fmt: skip
    )
    assert seen["policy_prompt"] == "p2"
    assert "v1+policy-p2" in (tmp_path / "runs" / "experiments.csv").read_text(encoding="utf-8")

    main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save"))
    assert seen["tools"] is None


def test_a_bad_kb_is_a_clean_error(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    assert main(args(config, tmp_path, "--kb", str(tmp_path / "missing"))) == 2
    assert "error:" in capsys.readouterr().err
    assert fake_server.requests == []


def test_cases_that_need_the_policy_tool_refuse_to_run_without_kb(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_models_config(tmp_path, fake_server.base_url)
    policy = str(Path(__file__).resolve().parents[2] / "evals" / "golden_policy")
    assert main(args(config, tmp_path, "--cases", policy)) == 2
    err = capsys.readouterr().err
    assert "--kb kb" in err and "policy-refund-timeline" in err
    assert fake_server.requests == []
    assert main(["--cases", policy, "--list"]) == 0  # listing needs no tool


def test_claim_check_gives_the_runner_a_judge_closes_it_and_is_noted(
    monkeypatch: pytest.MonkeyPatch, fake_server: FakeOpenAIServer, tmp_path: Path
) -> None:
    from kartly.llm import OpenAICompatClient

    seen: dict[str, Any] = {}
    closed: list[str] = []
    real_close = OpenAICompatClient.close

    def spy(cases: Any, client_for: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return run_suite(cases, client_for, **kwargs)

    def close(self: OpenAICompatClient) -> None:
        closed.append(self.name)
        real_close(self)

    monkeypatch.setattr("kartly.evals.cli.run_suite", spy)
    monkeypatch.setattr(OpenAICompatClient, "close", close)
    fake_server.responder = good_tracking_model
    config = write_models_config(tmp_path, fake_server.base_url)
    main(args(config, tmp_path, "--only", "tracking-in-transit", "--claim-check", "fake_small"))
    assert seen["claim_judge"] is not None and seen["claim_judge"].name == "fake_small"
    assert closed.count("fake_small") == 2  # the model and the judge
    [row] = list(csv.DictReader((tmp_path / "runs" / "experiments.csv").open(encoding="utf-8")))
    assert row["note"] == "claim-check=fake_small"
    main(args(config, tmp_path, "--only", "tracking-in-transit", "--no-save"))
    assert seen["claim_judge"] is None
