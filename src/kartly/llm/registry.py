"""Builds clients from config/models.yaml and the environment."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import httpx

from kartly.llm.client import OpenAICompatClient
from kartly.llm.config import ModelConfig, ModelsFile, load_models_file
from kartly.llm.errors import LLMConfigError
from kartly.llm.resilient import ResilientClient, RetryPolicy


class ModelRegistry:
    def __init__(self, config: ModelsFile, *, environ: Mapping[str, str] | None = None) -> None:
        self._config = config
        self._environ = os.environ if environ is None else environ

    @classmethod
    def from_yaml(
        cls, path: Path | str, *, environ: Mapping[str, str] | None = None
    ) -> ModelRegistry:
        return cls(load_models_file(path), environ=environ)

    @property
    def model_names(self) -> list[str]:
        return list(self._config.models)

    @property
    def roles(self) -> dict[str, list[str]]:
        return {role: list(chain) for role, chain in self._config.roles.items()}

    def model(self, name: str) -> ModelConfig:
        try:
            return self._config.models[name]
        except KeyError:
            known = ", ".join(self._config.models) or "none"
            raise LLMConfigError(f"unknown model '{name}'. Known models: {known}") from None

    def chain(self, role: str) -> list[str]:
        """The primary model followed by its fallbacks, for a role such as 'small'."""
        try:
            return list(self._config.roles[role])
        except KeyError:
            known = ", ".join(self._config.roles) or "none"
            raise LLMConfigError(f"unknown role '{role}'. Known roles: {known}") from None

    def client(self, name: str, *, http_client: httpx.Client | None = None) -> OpenAICompatClient:
        model = self.model(name)
        provider = self._config.providers[model.provider]
        api_key: str | None = None
        if provider.api_key_env:
            api_key = self._environ.get(provider.api_key_env, "").strip()
            if not api_key:
                raise LLMConfigError(
                    f"model '{name}' needs the {provider.api_key_env} environment variable "
                    "(set it in your shell or in .env)"
                )
        return OpenAICompatClient(name, model, provider, api_key=api_key, http_client=http_client)

    def resilient(
        self,
        name: str,
        fallbacks: Sequence[str] = (),
        *,
        policy: RetryPolicy | None = None,
        http_client: httpx.Client | None = None,
    ) -> ResilientClient:
        """``name`` first, then each fallback in order, with retries and a circuit breaker."""
        names = [name, *(f for f in fallbacks if f != name)]
        clients = [self.client(n, http_client=http_client) for n in dict.fromkeys(names)]
        return ResilientClient(clients, policy)
