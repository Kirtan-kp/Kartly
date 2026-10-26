from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import httpx
import pytest

from kartly.llm import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelNotFoundError,
    ModelPrice,
    OpenAICompatClient,
    ProviderUnavailableError,
    RateLimitError,
    ToolUseFailedError,
)
from kartly.llm.config import ModelConfig, ProviderConfig
from tests.llm_support import FakeOpenAIServer, completion_payload

PRICE = ModelPrice(input_per_mtok=Decimal("0.05"), output_per_mtok=Decimal("0.08"))
KEY = "gsk-super-secret-key"
MESSAGES = [{"role": "user", "content": "hello"}]
TOOLS = [{"type": "function", "function": {"name": "get_order", "parameters": {}}}]

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(
    handler: Handler,
    *,
    key: str | None = None,
    supports_tool_choice: bool = True,
    max_output_tokens: int | None = None,
) -> OpenAICompatClient:
    return OpenAICompatClient(
        "test_model",
        ModelConfig(
            provider="p",
            model="provider-model-id",
            price=PRICE,
            max_output_tokens=max_output_tokens,
        ),
        ProviderConfig(
            base_url="http://provider.test/v1/", supports_tool_choice=supports_tool_choice
        ),
        api_key=key,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def reply(
    payload: dict[str, Any], status: int = 200, headers: dict[str, str] | None = None
) -> Handler:
    return lambda request: httpx.Response(status, json=payload, headers=headers)


def error_body(message: str = "boom", code: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"error": {"message": message, "code": code, **extra}}


# ------------------------------------------------------------------ the request


def test_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_payload("hi"))

    make_client(handler, max_output_tokens=512).chat(MESSAGES, tools=TOOLS, seed=7)
    request = seen[0]
    body = json.loads(request.content)
    assert str(request.url) == "http://provider.test/v1/chat/completions"  # trailing slash handled
    assert body["model"] == "provider-model-id"  # the provider's id, not our logical name
    assert body["messages"] == MESSAGES
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 512  # the configured default
    assert body["tools"] == TOOLS
    assert body["seed"] == 7
    assert "tool_choice" not in body and "response_format" not in body


def test_an_explicit_max_tokens_beats_the_configured_default() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload("hi"))

    make_client(handler, max_output_tokens=512).chat(MESSAGES, max_tokens=64)
    assert seen[0]["max_tokens"] == 64


def test_no_tools_key_when_no_tools_are_given() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload("hi"))

    make_client(handler).chat(MESSAGES, tools=[])
    assert "tools" not in seen[0]


def test_extra_body_is_sent_but_cannot_override_what_the_caller_asked_for() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload("hi"))

    client = OpenAICompatClient(
        "c",
        ModelConfig.model_construct(  # skips validation to prove the client is safe regardless
            provider="p",
            model="m",
            price=PRICE,
            timeout_s=None,
            max_output_tokens=None,
            extra_body={"reasoning_effort": "low", "temperature": 99, "model": "evil"},
        ),
        ProviderConfig(base_url="http://provider.test/v1"),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat(MESSAGES, temperature=0.0)
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[0]["temperature"] == 0.0  # the explicit argument wins
    assert seen[0]["model"] == "m"


def test_bearer_header_only_when_there_is_a_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_payload("hi"))

    make_client(handler, key=KEY).chat(MESSAGES)
    make_client(handler, key=None).chat(MESSAGES)
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    assert "authorization" not in seen[1].headers


def test_tool_choice_is_sent_when_supported() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload("hi"))

    response = make_client(handler).chat(MESSAGES, tools=TOOLS, tool_choice="required")
    assert seen[0]["tool_choice"] == "required"
    assert response.ignored_params == ()


def test_unsupported_tool_choice_is_dropped_and_reported_not_silent() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload("hi"))

    response = make_client(handler, supports_tool_choice=False).chat(
        MESSAGES, tools=TOOLS, tool_choice="required"
    )
    assert "tool_choice" not in seen[0]
    assert response.ignored_params == ("tool_choice",)


# ----------------------------------------------------------------- the response


def test_a_plain_reply_with_usage_and_shadow_cost() -> None:
    payload = completion_payload("Hello!", usage=(1_200, 150), model="llama-3.1-8b-instant-0001")
    response = make_client(reply(payload, headers={"X-RateLimit-Remaining-Requests": "29"})).chat(
        MESSAGES
    )
    assert response.content == "Hello!"
    assert response.model == "test_model"
    assert response.provider_model == "llama-3.1-8b-instant-0001"
    assert response.finish_reason == "stop"
    assert response.tool_calls == ()
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (1_200, 150)
    assert response.usage.estimated is False
    assert response.cost_usd == Decimal("0.000072")  # 1200*0.05/1M + 150*0.08/1M, unrounded
    assert response.latency_ms >= 0
    assert response.rate_limit == {"x-ratelimit-remaining-requests": "29"}


def test_missing_usage_is_estimated_and_flagged() -> None:
    response = make_client(reply(completion_payload("a" * 400, usage=None))).chat(MESSAGES)
    assert response.usage.estimated is True
    assert response.usage.completion_tokens == 100
    assert response.usage.prompt_tokens > 0
    assert response.cost_usd > 0


def test_cached_tokens_are_read_from_prompt_token_details() -> None:
    payload = completion_payload("hi")
    payload["usage"]["prompt_tokens_details"] = {"cached_tokens": 30}
    assert make_client(reply(payload)).chat(MESSAGES).usage.cached_tokens == 30


def test_missing_provider_model_falls_back_to_the_configured_id() -> None:
    payload = completion_payload("hi")
    del payload["model"]
    assert make_client(reply(payload)).chat(MESSAGES).provider_model == "provider-model-id"


def test_null_content_is_none_not_a_crash() -> None:
    assert make_client(reply(completion_payload(None))).chat(MESSAGES).content is None


# -------------------------------------------------------------------- tool calls


def tool_payload(
    arguments: Any, *, call_id: Any = "call_9", name: Any = "get_order"
) -> dict[str, Any]:
    call: dict[str, Any] = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if call_id is not None:
        call["id"] = call_id
    return completion_payload(None, [call])


def test_a_valid_tool_call() -> None:
    response = make_client(reply(tool_payload('{"order_id": "KRT-100001"}'))).chat(MESSAGES)
    [call] = response.tool_calls
    assert (call.id, call.name) == ("call_9", "get_order")
    assert call.arguments == {"order_id": "KRT-100001"}
    assert call.parse_error is None
    assert response.finish_reason == "tool_calls"


def test_broken_json_arguments_become_a_parse_error_not_an_exception() -> None:
    response = make_client(reply(tool_payload('{"order_id": "KRT-1'))).chat(MESSAGES)
    [call] = response.tool_calls
    assert call.arguments is None
    assert call.raw_arguments == '{"order_id": "KRT-1'  # kept verbatim for the repair prompt
    assert call.parse_error and call.parse_error.startswith("invalid JSON")


@pytest.mark.parametrize("raw", ["[1, 2]", '"just a string"', "42", "null"])
def test_non_object_arguments_are_a_parse_error(raw: str) -> None:
    [call] = make_client(reply(tool_payload(raw))).chat(MESSAGES).tool_calls
    assert call.arguments is None
    assert call.parse_error == "arguments must be a JSON object"


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_arguments_mean_no_arguments(raw: str | None) -> None:
    [call] = make_client(reply(tool_payload(raw))).chat(MESSAGES).tool_calls
    assert call.arguments == {} and call.parse_error is None


def test_arguments_sent_as_an_object_are_accepted() -> None:
    [call] = make_client(reply(tool_payload({"order_id": "KRT-100001"}))).chat(MESSAGES).tool_calls
    assert call.arguments == {"order_id": "KRT-100001"}
    assert json.loads(call.raw_arguments) == {"order_id": "KRT-100001"}


def test_a_missing_call_id_is_made_up_so_the_tool_message_can_reference_it() -> None:
    [call] = make_client(reply(tool_payload("{}", call_id=None))).chat(MESSAGES).tool_calls
    assert call.id == "call_0"


def test_several_calls_keep_their_order() -> None:
    calls = [
        {"id": f"c{i}", "type": "function", "function": {"name": "get_order", "arguments": "{}"}}
        for i in range(3)
    ]
    response = make_client(reply(completion_payload(None, calls))).chat(MESSAGES)
    assert [c.id for c in response.tool_calls] == ["c0", "c1", "c2"]


@pytest.mark.parametrize("name", [None, "", 5])
def test_a_tool_call_without_a_name_is_a_malformed_response(name: Any) -> None:
    with pytest.raises(MalformedResponseError, match="no function name"):
        make_client(reply(tool_payload("{}", name=name))).chat(MESSAGES)


def test_assistant_message_round_trips_the_call_exactly() -> None:
    response = make_client(reply(tool_payload('{"order_id":  "KRT-100001"}'))).chat(MESSAGES)
    message = response.assistant_message()
    assert message["role"] == "assistant"
    assert message["content"] == ""  # never null: some providers reject it next to tool_calls
    assert message["tool_calls"] == [
        {
            "id": "call_9",
            "type": "function",
            "function": {"name": "get_order", "arguments": '{"order_id":  "KRT-100001"}'},
        }
    ]


def test_assistant_message_without_calls_has_no_tool_calls_key() -> None:
    message = make_client(reply(completion_payload("hi"))).chat(MESSAGES).assistant_message()
    assert message == {"role": "assistant", "content": "hi"}


# ----------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "error_class", "retryable"),
    [
        (400, BadRequestError, False),
        (401, AuthenticationError, False),
        (403, AuthenticationError, False),
        (404, ModelNotFoundError, False),
        (422, BadRequestError, False),
        (429, RateLimitError, True),
        (500, ProviderUnavailableError, True),
        (502, ProviderUnavailableError, True),
        (503, ProviderUnavailableError, True),
    ],
)
def test_http_errors_are_classified(
    status: int, error_class: type[LLMError], retryable: bool
) -> None:
    with pytest.raises(error_class) as caught:
        make_client(reply(error_body("nope"), status=status)).chat(MESSAGES)
    error = caught.value
    assert type(error) is error_class
    assert error.retryable is retryable
    assert error.status == status
    assert error.model == "test_model"
    assert "nope" in str(error)


def test_rate_limit_carries_the_retry_after_hint() -> None:
    handler = reply(error_body("slow down"), status=429, headers={"Retry-After": "7"})
    with pytest.raises(RateLimitError) as caught:
        make_client(handler).chat(MESSAGES)
    assert caught.value.retry_after == 7.0


@pytest.mark.parametrize("value", [None, "Wed, 21 Oct 2026 07:28:00 GMT", "-5", "soon"])
def test_unusable_retry_after_is_none(value: str | None) -> None:
    headers = {"Retry-After": value} if value is not None else {}
    with pytest.raises(RateLimitError) as caught:
        make_client(reply(error_body(), status=429, headers=headers)).chat(MESSAGES)
    assert caught.value.retry_after is None


def test_groq_style_tool_use_failed_is_repairable_and_keeps_what_the_model_wrote() -> None:
    body = error_body(
        "Failed to call a function.",
        code="tool_use_failed",
        failed_generation='<function=get_order{"order_id": "KRT-100001"}</function>',
    )
    with pytest.raises(ToolUseFailedError) as caught:
        make_client(reply(body, status=400)).chat(MESSAGES, tools=TOOLS)
    assert caught.value.retryable is False
    assert (
        caught.value.failed_generation == '<function=get_order{"order_id": "KRT-100001"}</function>'
    )


def test_a_400_with_another_code_is_a_plain_bad_request() -> None:
    with pytest.raises(BadRequestError):
        make_client(reply(error_body("too long", code="context_length_exceeded"), status=400)).chat(
            MESSAGES
        )


def test_error_bodies_of_any_shape_are_survivable() -> None:
    def text_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>Bad gateway</html>")

    with pytest.raises(ProviderUnavailableError, match="Bad gateway"):
        make_client(text_body).chat(MESSAGES)
    with pytest.raises(BadRequestError, match="plain string"):
        make_client(reply({"error": "plain string"}, status=400)).chat(MESSAGES)
    with pytest.raises(BadRequestError):
        make_client(reply({"detail": "something else"}, status=400)).chat(MESSAGES)
    with pytest.raises(BadRequestError):
        make_client(reply(["not", "an", "object"], status=400)).chat(MESSAGES)  # type: ignore[arg-type]


def test_a_huge_error_body_is_truncated() -> None:
    with pytest.raises(ProviderUnavailableError) as caught:
        make_client(reply(error_body("x" * 50_000), status=500)).chat(MESSAGES)
    assert len(str(caught.value)) < 500


def test_the_api_key_never_appears_in_an_error_even_if_the_provider_echoes_it() -> None:
    body = error_body(f"Invalid API Key: {KEY}", failed_generation=f"leak {KEY}")
    with pytest.raises(AuthenticationError) as caught:
        make_client(reply(body, status=401), key=KEY).chat(MESSAGES)
    assert KEY not in str(caught.value)
    assert "***" in str(caught.value)

    def text_body(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {KEY}")

    with pytest.raises(AuthenticationError) as caught:
        make_client(text_body, key=KEY).chat(MESSAGES)
    assert KEY not in str(caught.value)

    leaky = error_body("tool failed", code="tool_use_failed", failed_generation=f"x {KEY} y")
    with pytest.raises(ToolUseFailedError) as failed:
        make_client(reply(leaky, status=400), key=KEY).chat(MESSAGES)
    assert KEY not in (failed.value.failed_generation or "")


def test_the_key_is_not_in_the_client_repr() -> None:
    client = make_client(reply(completion_payload("hi")), key=KEY)
    assert KEY not in repr(client) and KEY not in str(client)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": "text"}]},
        {"choices": [{"message": {"content": "x", "tool_calls": "oops"}}]},
        [],
    ],
)
def test_a_200_that_is_not_a_chat_completion_is_malformed(payload: Any) -> None:
    with pytest.raises(MalformedResponseError) as caught:
        make_client(reply(payload)).chat(MESSAGES)
    assert caught.value.retryable is True


def test_a_200_with_a_non_json_body_is_malformed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>captive portal</html>")

    with pytest.raises(MalformedResponseError):
        make_client(handler).chat(MESSAGES)


# ---------------------------------------------------------------------- network


def test_a_timeout_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeoutError) as caught:
        make_client(handler).chat(MESSAGES)
    assert caught.value.retryable is True


def test_a_refused_connection_says_the_server_may_not_be_running() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError, match="is the server running"):
        make_client(handler).chat(MESSAGES)


def test_other_network_failures_are_provider_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("server hung up", request=request)

    with pytest.raises(ProviderUnavailableError, match="RemoteProtocolError"):
        make_client(handler).chat(MESSAGES)


# --------------------------------------------- over a real socket (no mock transport)


def real_client(
    base_url: str,
    *,
    key: str | None = None,
    timeout_s: float = 5.0,
    supports_tool_choice: bool = True,
) -> OpenAICompatClient:
    return OpenAICompatClient(
        "real",
        ModelConfig(provider="p", model="m", price=PRICE),
        ProviderConfig(
            base_url=base_url, timeout_s=timeout_s, supports_tool_choice=supports_tool_choice
        ),
        api_key=key,
        http_client=httpx.Client(trust_env=False),  # never route localhost through a proxy
    )


def test_a_full_exchange_with_a_real_http_server(fake_server: FakeOpenAIServer) -> None:
    client = real_client(fake_server.base_url, key=KEY)
    first = client.chat(MESSAGES, tools=TOOLS)
    [call] = first.tool_calls
    assert call.name == "get_tracking" and call.arguments == {"order_id": "KRT-100002"}

    history = [
        *MESSAGES,
        first.assistant_message(),
        {"role": "tool", "tool_call_id": call.id, "content": '{"ok": true}'},
    ]
    final = client.chat(history, tools=TOOLS)
    assert final.content == "Your order KRT-100002 is in transit."

    assert [r["path"] for r in fake_server.requests] == ["/v1/chat/completions"] * 2
    assert fake_server.requests[0]["headers"]["authorization"] == f"Bearer {KEY}"
    sent_tool_message = fake_server.requests[1]["body"]["messages"][-1]
    assert sent_tool_message["tool_call_id"] == "call_abc"  # ids survive the round trip


def test_429_with_retry_after_over_a_real_socket(fake_server: FakeOpenAIServer) -> None:
    fake_server.responder = lambda body: (429, error_body("rate limited"), {"Retry-After": "3"})
    with pytest.raises(RateLimitError) as caught:
        real_client(fake_server.base_url).chat(MESSAGES)
    assert caught.value.retry_after == 3.0


def test_connection_refused_over_a_real_socket() -> None:
    with socket.socket() as sock:  # grab a free port, then close it so nothing listens
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(ProviderUnavailableError, match="cannot connect"):
        real_client(f"http://127.0.0.1:{port}/v1").chat(MESSAGES)


def test_a_slow_server_times_out(fake_server: FakeOpenAIServer) -> None:
    def slow(body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
        time.sleep(1.0)
        return 200, completion_payload("late"), {}

    fake_server.responder = slow
    started = time.perf_counter()
    with pytest.raises(LLMTimeoutError):
        real_client(fake_server.base_url, timeout_s=0.2).chat(MESSAGES)
    assert time.perf_counter() - started < 0.9  # gave up at the timeout, did not wait it out


def test_closing_only_closes_a_client_it_created() -> None:
    shared = httpx.Client(transport=httpx.MockTransport(reply(completion_payload("hi"))))
    client = OpenAICompatClient(
        "c",
        ModelConfig(provider="p", model="m", price=PRICE),
        ProviderConfig(base_url="http://x.test/v1"),
        http_client=shared,
    )
    with client:
        client.chat(MESSAGES)
    assert shared.is_closed is False  # the caller owns it
    shared.close()

    owned = OpenAICompatClient(
        "c",
        ModelConfig(provider="p", model="m", price=PRICE),
        ProviderConfig(base_url="http://x/v1"),
    )
    owned.close()
    assert owned._http.is_closed is True
