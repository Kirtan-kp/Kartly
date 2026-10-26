"""A chat client for any OpenAI-compatible endpoint (Ollama, Groq, ...).

One request per call: no retries, no fallback. Failures are classified into the error types in
kartly.llm.errors so the resilient wrapper (Week 5) can decide. Sync on purpose: the workers
(RQ) are sync, and a thread per run is plenty at this scale.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any

import httpx

from kartly.llm.config import ModelConfig, ProviderConfig
from kartly.llm.errors import (
    AuthenticationError,
    BadRequestError,
    LLMError,
    LLMTimeoutError,
    MalformedResponseError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
    ToolUseFailedError,
)
from kartly.llm.pricing import compute_cost, estimate_prompt_tokens, estimate_tokens
from kartly.llm.types import ChatResponse, ToolCall, Usage
from kartly.sanitize import clean_str

MAX_ERROR_TEXT = 300
MIN_REDACTED_KEY_LENGTH = 6  # shorter "keys" would mangle ordinary words if replaced


class OpenAICompatClient:
    def __init__(
        self,
        name: str,
        model: ModelConfig,
        provider: ProviderConfig,
        *,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.name = name
        self._model = model
        self._provider = provider
        self._api_key = api_key
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client()
        self._url = provider.base_url.rstrip("/") + "/chat/completions"
        seconds = model.timeout_s or provider.timeout_s
        self._timeout = httpx.Timeout(seconds, connect=min(10.0, seconds))

    def __repr__(self) -> str:  # never includes the API key
        return (
            f"OpenAICompatClient(name={self.name!r}, model={self._model.model!r}, "
            f"base_url={self._provider.base_url!r})"
        )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---------------------------------------------------------------------- chat

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> ChatResponse:
        body: dict[str, Any] = {
            **self._model.extra_body,  # cannot clash with the fields below (see config.py)
            "model": self._model.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        ignored: list[str] = []
        limit = max_tokens if max_tokens is not None else self._model.max_output_tokens
        if limit is not None:
            body["max_tokens"] = limit
        if tools:
            body["tools"] = list(tools)
        if tool_choice is not None:
            if self._provider.supports_tool_choice:
                body["tool_choice"] = tool_choice
            else:
                ignored.append("tool_choice")  # reported on the response, never silent
        if seed is not None:
            body["seed"] = seed
        if response_format is not None:
            body["response_format"] = dict(response_format)

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        started = time.perf_counter()
        try:
            response = self._http.post(self._url, json=body, headers=headers, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name}: no response within {self._timeout.read:g}s", model=self.name
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: cannot connect to {self._provider.base_url} "
                "(is the server running?)",
                model=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"{self.name}: network error ({type(exc).__name__})", model=self.name
            ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code != 200:
            raise self._error_for(response)
        return self._parse(response, body, latency_ms, tuple(ignored))

    # -------------------------------------------------------------------- errors

    def _error_for(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        code, text, failed_generation = self._error_fields(response)
        message = f"{self.name}: HTTP {status}" + (f": {text}" if text else "")
        common: dict[str, Any] = {"model": self.name, "status": status}
        if status == 429:
            return RateLimitError(message, retry_after=_retry_after(response), **common)
        if status in (401, 403):
            return AuthenticationError(message, **common)
        if status == 404:
            return ModelNotFoundError(f"{message} (check the model name and base_url)", **common)
        if status == 400 and code == "tool_use_failed":
            return ToolUseFailedError(message, failed_generation=failed_generation, **common)
        if 400 <= status < 500:
            return BadRequestError(message, **common)
        return ProviderUnavailableError(message, **common)

    def _redact(self, text: str) -> str:
        """Providers sometimes echo the bad key back in a 401 body; never let it reach a log."""
        key = self._api_key
        if key and len(key) >= MIN_REDACTED_KEY_LENGTH:
            return text.replace(key, "***")
        return text

    def _error_fields(self, response: httpx.Response) -> tuple[str | None, str, str | None]:
        """(error code, short text, failed_generation) from an error body of any shape."""
        try:
            body = response.json()
        except ValueError:
            return None, self._redact(clean_str(response.text, MAX_ERROR_TEXT)), None
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, str):
            return None, self._redact(clean_str(error, MAX_ERROR_TEXT)), None
        if isinstance(error, dict):
            code = error.get("code")
            failed = error.get("failed_generation")
            return (
                str(code) if code is not None else None,
                self._redact(clean_str(str(error.get("message", "")), MAX_ERROR_TEXT)),
                self._redact(clean_str(failed, MAX_ERROR_TEXT * 4))
                if isinstance(failed, str)
                else None,
            )
        return None, "", None

    # ------------------------------------------------------------------- parsing

    def _parse(
        self,
        response: httpx.Response,
        body: Mapping[str, Any],
        latency_ms: int,
        ignored: tuple[str, ...],
    ) -> ChatResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MalformedResponseError(
                f"{self.name}: response was not a chat completion", model=self.name
            ) from exc
        if not isinstance(message, dict):
            raise MalformedResponseError(f"{self.name}: bad message in response", model=self.name)

        content = message.get("content")
        content = content if isinstance(content, str) else None
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise MalformedResponseError(
                f"{self.name}: bad tool_calls in response", model=self.name
            )
        tool_calls = tuple(self._parse_tool_call(raw, i) for i, raw in enumerate(raw_calls))

        usage = _parse_usage(data.get("usage"))
        if usage is None:
            produced = (content or "") + "".join(c.raw_arguments + c.name for c in tool_calls)
            usage = Usage(
                prompt_tokens=estimate_prompt_tokens(body["messages"], body.get("tools")),
                completion_tokens=estimate_tokens(produced),
                estimated=True,
            )
        reported_model = data.get("model")
        finish = choice.get("finish_reason")
        return ChatResponse(
            model=self.name,
            provider_model=reported_model if isinstance(reported_model, str) else self._model.model,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish if isinstance(finish, str) else None,
            usage=usage,
            cost_usd=compute_cost(self._model.price, usage),
            latency_ms=latency_ms,
            ignored_params=ignored,
            rate_limit={
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower().startswith("x-ratelimit-")
            },
        )

    def _parse_tool_call(self, raw: Any, index: int) -> ToolCall:
        function = raw.get("function") if isinstance(raw, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(function, dict) or not isinstance(name, str) or not name:
            raise MalformedResponseError(
                f"{self.name}: tool call {index} has no function name", model=self.name
            )
        call_id = raw.get("id")
        call_id = call_id if isinstance(call_id, str) and call_id else f"call_{index}"

        arguments = function.get("arguments")
        if isinstance(arguments, dict):  # some servers send an object instead of a JSON string
            return ToolCall(call_id, name, arguments, json.dumps(arguments))
        raw_text = arguments if isinstance(arguments, str) else ""
        if not raw_text.strip():
            return ToolCall(call_id, name, {}, "{}")
        try:
            parsed = json.loads(raw_text)
        except ValueError as exc:
            return ToolCall(call_id, name, None, raw_text, parse_error=f"invalid JSON: {exc}")
        if not isinstance(parsed, dict):
            return ToolCall(
                call_id, name, None, raw_text, parse_error="arguments must be a JSON object"
            )
        return ToolCall(call_id, name, parsed, raw_text)


def _parse_usage(usage: Any) -> Usage | None:
    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else 0
    return Usage(prompt, completion, cached if isinstance(cached, int) else 0)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None  # an HTTP date: not supported, the caller falls back to its own backoff
    return seconds if seconds is not None and seconds >= 0 else None
