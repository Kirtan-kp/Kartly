"""The shape of config/models.yaml, validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kartly.llm.errors import LLMConfigError
from kartly.llm.pricing import ModelPrice


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str
    api_key_env: str | None = None  # name of the environment variable holding the key
    supports_tool_choice: bool = True
    timeout_s: float = Field(default=60.0, gt=0)


# Request fields the client owns. A model's ``extra_body`` may not set them, so config can never
# silently override what the agent loop asked for.
RESERVED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "response_format",
        "stream",
    }
)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str  # the model ID the provider expects
    price: ModelPrice
    timeout_s: float | None = Field(default=None, gt=0)  # overrides the provider's
    max_output_tokens: int | None = Field(default=None, gt=0)  # default cap per call
    # Provider-specific fields sent with every request, e.g. ``reasoning_effort: low`` for
    # reasoning models.
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("extra_body")
    @classmethod
    def _no_reserved_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        clash = sorted(RESERVED_REQUEST_FIELDS & value.keys())
        if clash:
            raise ValueError(f"extra_body may not set {', '.join(clash)}")
        return value


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    roles: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_references(self) -> ModelsFile:
        for name, model in self.models.items():
            if model.provider not in self.providers:
                raise ValueError(f"model '{name}' uses unknown provider '{model.provider}'")
        for role, chain in self.roles.items():
            if not chain:
                raise ValueError(f"role '{role}' lists no models")
            for name in chain:
                if name not in self.models:
                    raise ValueError(f"role '{role}' lists unknown model '{name}'")
        return self


def load_models_file(path: Path | str) -> ModelsFile:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMConfigError(f"cannot read model config {path}: {exc.strerror or exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise LLMConfigError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return ModelsFile.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'file'}: {e['msg']}"
            for e in exc.errors(include_url=False, include_input=False)
        )
        raise LLMConfigError(f"{path} is invalid: {problems}") from exc
