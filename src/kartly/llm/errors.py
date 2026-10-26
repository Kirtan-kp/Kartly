"""Errors raised by the LLM layer.

The client classifies every failure but never retries or falls back by itself (ADR-014). The
resilient wrapper added in Week 5 decides what to do from two things: the class of the error and
its ``retryable`` flag.

- retryable, same or another provider:  RateLimitError, ProviderUnavailableError,
                                        LLMTimeoutError, MalformedResponseError
- repairable by re-prompting the model: ToolUseFailedError
- a configuration bug, fail fast:       AuthenticationError, ModelNotFoundError,
                                        BadRequestError, LLMConfigError

Messages never contain API keys or request headers.
"""

from __future__ import annotations


class LLMConfigError(Exception):
    """models.yaml is missing, invalid, or needs something absent (for example an API key)."""


class LLMError(Exception):
    retryable: bool = False

    def __init__(self, message: str, *, model: str | None = None, status: int | None = None):
        super().__init__(message)
        self.model = model
        self.status = status


class RateLimitError(LLMError):
    """HTTP 429. ``retry_after`` is the provider's hint in seconds, when it gave one."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        status: int | None = None,
        retry_after: float | None = None,
    ):
        super().__init__(message, model=model, status=status)
        self.retry_after = retry_after


class ProviderUnavailableError(LLMError):
    """5xx, or the server could not be reached (for example Ollama is not running)."""

    retryable = True


class LLMTimeoutError(LLMError):
    retryable = True


class MalformedResponseError(LLMError):
    """HTTP 200 but the body was not a usable chat completion."""

    retryable = True


class ToolUseFailedError(LLMError):
    """The model produced an invalid tool call and the provider rejected it (Groq: HTTP 400,
    code ``tool_use_failed``). Not a configuration bug: re-prompting often fixes it."""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        status: int | None = None,
        failed_generation: str | None = None,
    ):
        super().__init__(message, model=model, status=status)
        self.failed_generation = failed_generation


class AuthenticationError(LLMError):
    """HTTP 401 or 403: a missing, wrong, or unauthorised API key."""


class ModelNotFoundError(LLMError):
    """HTTP 404: unknown model (for Ollama, usually not pulled or not created yet)."""


class BadRequestError(LLMError):
    """Any other 4xx: the request itself is wrong."""
