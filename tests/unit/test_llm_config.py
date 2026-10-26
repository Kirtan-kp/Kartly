from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from kartly.llm import LLMConfigError, ModelRegistry
from kartly.llm.config import load_models_file
from tests.support import ROOT

SHIPPED = ROOT / "config" / "models.yaml"

VALID = """
providers:
  p:
    base_url: http://localhost:1/v1
models:
  m:
    provider: p
    model: some-model
    price: {input_per_mtok: 1, output_per_mtok: 2}
roles:
  small: [m]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------ the shipped config


def test_the_shipped_config_is_valid_and_complete() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={})
    assert {"local_small", "groq_small", "groq_large"} <= set(registry.model_names)
    for role, chain in registry.roles.items():
        assert chain, role
        for name in chain:
            registry.model(name)  # every role entry resolves


def test_shipped_prices_are_exact_decimals_not_float_noise() -> None:
    price = ModelRegistry.from_yaml(SHIPPED, environ={}).model("groq_small").price
    assert price.input_per_mtok == Decimal("0.075")
    assert price.output_per_mtok == Decimal("0.30")


def test_every_shipped_price_says_where_it_came_from() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={})
    for name in registry.model_names:
        assert registry.model(name).price.note, f"{name} has no price note"


def test_hosted_reasoning_models_get_a_reasoning_effort() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={})
    assert registry.model("groq_small").extra_body == {"reasoning_effort": "low"}
    assert registry.model("local_small").extra_body == {}


def test_extra_body_defaults_to_empty(tmp_path: Path) -> None:
    assert load_models_file(write(tmp_path, VALID)).models["m"].extra_body == {}


def test_the_small_role_prefers_the_local_model_then_falls_back_to_groq() -> None:
    chain = ModelRegistry.from_yaml(SHIPPED, environ={}).chain("small")
    assert chain[0] == "local_small" and "groq_small" in chain[1:]


def test_the_ollama_modelfile_sets_a_context_length() -> None:
    text = (ROOT / "ollama" / "Modelfile.small").read_text()
    assert "PARAMETER num_ctx 8192" in text
    shipped_model = ModelRegistry.from_yaml(SHIPPED, environ={}).model("local_small").model
    assert shipped_model == "kartly-small"  # the name the README tells you to create


# ------------------------------------------------------------------ validation


def test_a_minimal_valid_file_loads(tmp_path: Path) -> None:
    config = load_models_file(write(tmp_path, VALID))
    assert config.models["m"].price.input_per_mtok == 1


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (("provider: p", "provider: nope"), "unknown provider 'nope'"),
        (("[m]", "[ghost]"), "unknown model 'ghost'"),
        (("[m]", "[]"), "lists no models"),
        (("input_per_mtok: 1", "input_per_mtok: -1"), "greater than or equal to 0"),
        (("base_url: http://localhost:1/v1", "base_urll: x"), "base_url"),
        (("some-model", "some-model\n    max_output_tokens: 0"), "greater than 0"),
        (("model: some-model", "modle: some-model"), "Extra inputs are not permitted"),
        (
            ("some-model", "some-model\n    extra_body: {max_tokens: 5}"),
            "extra_body may not set max_tokens",
        ),
        (
            ("some-model", "some-model\n    extra_body: {model: x, tools: []}"),
            "extra_body may not set model, tools",
        ),
    ],
)
def test_invalid_files_are_rejected_with_a_helpful_message(
    tmp_path: Path, edit: tuple[str, str], expected: str
) -> None:
    broken = VALID.replace(*edit)
    with pytest.raises(LLMConfigError, match=expected):
        load_models_file(write(tmp_path, broken))


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(LLMConfigError, match="cannot read"):
        load_models_file(tmp_path / "absent.yaml")
    with pytest.raises(LLMConfigError, match="not valid YAML"):
        load_models_file(write(tmp_path, "providers: [unclosed"))
    with pytest.raises(LLMConfigError, match="invalid"):
        load_models_file(write(tmp_path, "just a string"))


# ---------------------------------------------------------------- the registry


def test_a_model_needing_a_key_fails_clearly_without_one() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={})
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        registry.client("groq_small")


def test_a_blank_key_counts_as_missing() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={"GROQ_API_KEY": "   "})
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        registry.client("groq_small")


def test_a_client_never_shows_its_key() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={"GROQ_API_KEY": "gsk-very-secret"})
    client = registry.client("groq_small")
    assert "gsk-very-secret" not in repr(client)
    assert "gsk-very-secret" not in str(client)
    client.close()


def test_a_local_model_needs_no_key() -> None:
    client = ModelRegistry.from_yaml(SHIPPED, environ={}).client("local_small")
    assert client.name == "local_small"
    client.close()


def test_unknown_names_list_what_exists() -> None:
    registry = ModelRegistry.from_yaml(SHIPPED, environ={})
    with pytest.raises(LLMConfigError, match="Known models: .*groq_small"):
        registry.model("gpt-9")
    with pytest.raises(LLMConfigError, match="Known roles: .*small"):
        registry.chain("medium")
