"""Test doubles for the LLM layer (import as ``tests.llm_support``).

- ScriptedClient: a ChatClient that returns queued responses and records every call.
- FakeOpenAIServer: a tiny OpenAI-compatible HTTP server, so the real httpx client, headers and
  JSON handling are exercised over a real socket rather than a mock.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from kartly.llm.types import ChatResponse, ToolCall, Usage

# --------------------------------------------------------------------- scripted client


def make_response(
    content: str | None = "ok",
    tool_calls: Sequence[ToolCall] = (),
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cost: str = "0.00001",
    latency_ms: int = 5,
    model: str = "scripted",
) -> ChatResponse:
    return ChatResponse(
        model=model,
        provider_model=model,
        content=content,
        tool_calls=tuple(tool_calls),
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(prompt_tokens, completion_tokens),
        cost_usd=Decimal(cost),
        latency_ms=latency_ms,
    )


def tool_call(
    name: str = "get_tracking",
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str = "call_1",
    raw: str | None = None,
    parse_error: str | None = None,
) -> ToolCall:
    args = {"order_id": "KRT-100002"} if arguments is None and raw is None else arguments
    raw_text = raw if raw is not None else json.dumps(args)
    return ToolCall(call_id, name, args, raw_text, parse_error)


class ScriptedClient:
    """Returns queued responses in order; an Exception in the queue is raised instead."""

    name = "scripted"

    def __init__(self, responses: Iterable[ChatResponse | Exception]) -> None:
        self._queue = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls.append({"messages": [dict(m) for m in messages], **kwargs})
        item = self._queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item


# ------------------------------------------------------------------- fake HTTP server

Responder = Callable[[dict[str, Any]], tuple[int, dict[str, Any], dict[str, str]]]


def completion_payload(
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    *,
    usage: tuple[int, int] | None = (50, 10),
    model: str = "fake-model-id",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }
    if usage is not None:
        payload["usage"] = {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        }
    return payload


def well_behaved_model(body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str]]:
    """Answers like a decent tool-calling model: OK, then a tool call, then a final answer."""
    messages = body.get("messages", [])
    if any(m.get("role") == "tool" for m in messages):
        return 200, completion_payload("Your order KRT-100002 is in transit."), {}
    if body.get("tools"):
        call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_tracking", "arguments": '{"order_id": "KRT-100002"}'},
        }
        return 200, completion_payload(None, [call]), {}
    return 200, completion_payload("OK"), {}


class FakeOpenAIServer:
    def __init__(self, responder: Responder = well_behaved_model) -> None:
        self.responder = responder
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                status, payload, extra = outer.responder(body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in extra.items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def start(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
