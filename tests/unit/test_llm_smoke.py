from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kartly.llm import AuthenticationError, LLMTimeoutError, ToolUseFailedError
from kartly.llm.smoke import format_report, main, run_smoke
from tests.llm_support import (
    FakeOpenAIServer,
    ScriptedClient,
    completion_payload,
    make_response,
    tool_call,
)


def names(results: list[Any]) -> list[tuple[str, str]]:
    return [(r.name, "skip" if r.skipped else ("pass" if r.ok else "fail")) for r in results]


def test_a_well_behaved_model_passes_all_three_steps() -> None:
    client = ScriptedClient(
        [
            make_response("OK"),
            make_response(None, [tool_call()]),
            make_response("Your order is in transit."),
        ]
    )
    results = run_smoke(client)
    assert names(results) == [
        ("plain_completion", "pass"),
        ("tool_call", "pass"),
        ("tool_round_trip", "pass"),
    ]
    assert all(r.cost_usd > 0 and r.tokens > 0 for r in results)


def test_the_round_trip_sends_the_call_back_with_a_matching_tool_message() -> None:
    client = ScriptedClient(
        [
            make_response("OK"),
            make_response(None, [tool_call("get_tracking", call_id="call_xyz")]),
            make_response("done"),
        ]
    )
    run_smoke(client)
    history = client.calls[2]["messages"]
    assistant, tool = history[-2], history[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_xyz"
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_xyz"
    assert json.loads(tool["content"])["data"]["order_id"] == "KRT-100002"
    assert client.calls[1]["tools"] and client.calls[2]["tools"]  # tools stay on every call


def test_a_dead_endpoint_fails_step_one_and_skips_the_rest() -> None:
    client = ScriptedClient([AuthenticationError("bad key", model="m", status=401)])
    results = run_smoke(client)
    assert names(results) == [
        ("plain_completion", "fail"),
        ("tool_call", "skip"),
        ("tool_round_trip", "skip"),
    ]
    assert "AuthenticationError" in results[0].detail


def test_not_saying_ok_fails_step_one_but_the_run_continues() -> None:
    client = ScriptedClient(
        [
            make_response("Certainly! How can I help?"),
            make_response(None, [tool_call()]),
            make_response("x"),
        ]
    )
    assert names(run_smoke(client))[0] == ("plain_completion", "fail")


def test_answering_without_a_tool_call_fails_step_two() -> None:
    client = ScriptedClient([make_response("OK"), make_response("It is on its way, I think.")])
    results = run_smoke(client)
    assert names(results)[1:] == [("tool_call", "fail"), ("tool_round_trip", "skip")]
    assert "without calling a tool" in results[1].detail


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (tool_call("delete_everything", {}), "unknown tool"),
        (
            tool_call("get_tracking", raw='{"order_id": "KRT-', parse_error="invalid JSON: eof"),
            "unparseable",
        ),
        (tool_call("get_tracking", {"order_id": "KRT-999999"}), "wrong order"),
        (tool_call("get_tracking", {}), "fail the tool schema"),
        (
            tool_call("get_tracking", {"order_id": "KRT-100002", "customer_id": "c1"}),
            "fail the tool schema",
        ),
        (tool_call("get_tracking", {"order_id": "not an id"}), "fail the tool schema"),
    ],
)
def test_bad_tool_calls_are_diagnosed(call: Any, expected: str) -> None:
    client = ScriptedClient([make_response("OK"), make_response(None, [call])])
    results = run_smoke(client)
    assert names(results)[1:] == [("tool_call", "fail"), ("tool_round_trip", "skip")]
    assert expected in results[1].detail


def test_one_good_call_among_several_is_enough_but_a_bad_one_fails_the_step() -> None:
    good, extra = tool_call("get_order", call_id="a"), tool_call("get_tracking", call_id="b")
    ok = run_smoke(
        ScriptedClient(
            [make_response("OK"), make_response(None, [good, extra]), make_response("done")]
        )
    )
    assert names(ok)[1] == ("tool_call", "pass")

    bad = tool_call("nope", {}, call_id="c")
    failed = run_smoke(ScriptedClient([make_response("OK"), make_response(None, [good, bad])]))
    assert names(failed)[1] == ("tool_call", "fail")


def test_tool_use_failed_shows_what_the_model_wrote() -> None:
    client = ScriptedClient(
        [
            make_response("OK"),
            ToolUseFailedError("HTTP 400", model="m", failed_generation="<function=get_order{oops"),
        ]
    )
    results = run_smoke(client)
    assert names(results)[1:] == [("tool_call", "fail"), ("tool_round_trip", "skip")]
    assert "<function=get_order{oops" in results[1].detail


def test_a_timeout_on_the_last_step_is_reported_not_raised() -> None:
    client = ScriptedClient(
        [
            make_response("OK"),
            make_response(None, [tool_call()]),
            LLMTimeoutError("slow", model="m"),
        ]
    )
    results = run_smoke(client)
    assert names(results)[2] == ("tool_round_trip", "fail")


@pytest.mark.parametrize(
    ("final", "expected"),
    [
        (make_response(""), "empty reply"),
        (make_response(None), "empty reply"),
        (make_response(None, [tool_call()]), "another tool call"),
    ],
)
def test_the_round_trip_needs_a_written_answer(final: Any, expected: str) -> None:
    client = ScriptedClient([make_response("OK"), make_response(None, [tool_call()]), final])
    results = run_smoke(client)
    assert names(results)[2] == ("tool_round_trip", "fail")
    assert expected in results[2].detail


def cut_off(content: str | None) -> Any:
    response = make_response(content)
    return replace(response, finish_reason="length")


def test_a_reply_cut_off_by_the_token_cap_says_so() -> None:
    # What a reasoning model does when it spends the whole cap thinking.
    results = run_smoke(
        ScriptedClient([cut_off(""), make_response(None, [tool_call()]), make_response("x")])
    )
    assert "cut off at max_tokens" in results[0].detail

    results = run_smoke(ScriptedClient([make_response("OK"), cut_off("")]))
    assert "cut off at max_tokens" in results[1].detail

    results = run_smoke(
        ScriptedClient([make_response("OK"), make_response(None, [tool_call()]), cut_off("")])
    )
    assert "empty reply (cut off" in results[2].detail


def test_the_smoke_calls_leave_room_for_reasoning() -> None:
    client = ScriptedClient(
        [make_response("OK"), make_response(None, [tool_call()]), make_response("x")]
    )
    run_smoke(client)
    assert all(call["max_tokens"] >= 512 for call in client.calls)


def test_estimated_usage_is_called_out() -> None:
    estimated = make_response("OK")
    object.__setattr__(estimated.usage, "estimated", True)
    results = run_smoke(ScriptedClient([estimated, make_response("no tool")]))
    assert "estimated" in results[0].detail


def test_the_report_is_readable_and_shows_the_shadow_cost() -> None:
    client = ScriptedClient([make_response("OK"), make_response("no tool call here")])
    report = format_report("local_small", run_smoke(client))
    assert report.startswith("Model: local_small")
    assert "[PASS] plain_completion" in report
    assert "[FAIL] tool_call" in report
    assert "[SKIP] tool_round_trip" in report
    assert "shadow cost $0.0000" in report


# ---------------------------------------------------------------------------- CLI


def write_config(tmp_path: Path, base_url: str, *, key_env: str | None = None) -> Path:
    key_line = f"    api_key_env: {key_env}\n" if key_env else ""
    path = tmp_path / "models.yaml"
    path.write_text(
        f"""
providers:
  fake:
    base_url: {base_url}
{key_line}    supports_tool_choice: false
    timeout_s: 5
models:
  fake_small:
    provider: fake
    model: fake-model-id
    price: {{input_per_mtok: 0.05, output_per_mtok: 0.08}}
roles:
  small: [fake_small]
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _hermetic_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("kartly.llm.smoke.load_env", lambda: None)  # never read the real .env
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.delenv("FAKE_KEY", raising=False)


def test_cli_end_to_end_against_a_fake_provider(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, fake_server.base_url)
    assert main(["--config", str(config), "fake_small"]) == 0
    out = capsys.readouterr().out
    assert "[PASS] plain_completion" in out
    assert "[PASS] tool_call" in out
    assert "[PASS] tool_round_trip" in out
    assert "valid call: get_tracking" in out
    assert len(fake_server.requests) == 3


def test_cli_exit_code_1_when_a_step_fails(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = lambda body: (200, completion_payload("I do not use tools."), {})
    config = write_config(tmp_path, fake_server.base_url)
    assert main(["--config", str(config), "fake_small"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] plain_completion" in out


def test_cli_reports_a_provider_error_as_a_failed_step_not_a_traceback(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_server.responder = lambda body: (401, {"error": {"message": "Invalid API Key"}}, {})
    config = write_config(tmp_path, fake_server.base_url)
    assert main(["--config", str(config), "fake_small"]) == 1
    out = capsys.readouterr().out
    assert "AuthenticationError" in out and "[SKIP] tool_call" in out


def test_cli_missing_key_names_the_variable_and_exits_2(
    fake_server: FakeOpenAIServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, fake_server.base_url, key_env="FAKE_KEY")
    assert main(["--config", str(config), "fake_small"]) == 2
    assert "FAKE_KEY" in capsys.readouterr().err
    assert fake_server.requests == []  # failed before any network call


def test_cli_sends_the_key_and_never_prints_it(
    fake_server: FakeOpenAIServer,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_KEY", "gsk-test-key-123456")
    config = write_config(tmp_path, fake_server.base_url, key_env="FAKE_KEY")
    assert main(["--config", str(config), "fake_small"]) == 0
    captured = capsys.readouterr()
    assert fake_server.requests[0]["headers"]["authorization"] == "Bearer gsk-test-key-123456"
    assert "gsk-test-key-123456" not in captured.out + captured.err


def test_cli_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(tmp_path, "http://127.0.0.1:1/v1")
    assert main(["--config", str(config), "--list"]) == 0
    out = capsys.readouterr().out
    assert "fake_small" in out and "fake/fake-model-id" in out and "role small" in out


def test_cli_without_a_model_lists_and_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "http://127.0.0.1:1/v1")
    assert main(["--config", str(config)]) == 2
    assert "fake_small" in capsys.readouterr().out


def test_cli_unknown_model_and_bad_config_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "http://127.0.0.1:1/v1")
    assert main(["--config", str(config), "nope"]) == 2
    assert "Known models" in capsys.readouterr().err
    assert main(["--config", str(tmp_path / "missing.yaml"), "x"]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_cli_with_nothing_listening_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path, "http://127.0.0.1:9/v1")  # port 9 (discard): nothing there
    assert main(["--config", str(config), "fake_small"]) == 1
    assert "is the server running" in capsys.readouterr().out
