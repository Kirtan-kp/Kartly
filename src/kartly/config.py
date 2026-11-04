"""Application settings, read from environment variables (prefix KARTLY_) and .env."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KARTLY_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://kartly:kartly@localhost:5432/kartly"
    redis_url: str = "redis://localhost:6379/0"
    models_config_path: str = "config/models.yaml"
    # API (KARTLY_API_KEY, KARTLY_DEFAULT_MODEL, ...). The key has no default on purpose.
    api_key: str | None = None
    default_model: str | None = (
        None  # a name from models.yaml; default: first model of role "small"
    )
    # A model from models.yaml that labels each ticket (intent, tier) before the resolver runs,
    # sending Tier 3 to a human. Empty means no model router (the rules still run).
    router_model: str | None = None
    # Policy search (the search_policy tool). Set kb_dir (for example "kb") to enable it; it needs
    # an Ollama embedding model (ollama pull nomic-embed-text). min_score is a dense cosine
    # cut-off: below it the tool reports "no policy found" (ADR-037).
    kb_dir: str | None = None
    embed_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    policy_min_score: float = 0.67
    max_workers: int = 2  # tickets processed at the same time
    max_pending_runs: int = 20  # running + queued; beyond this the API answers 503


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_env() -> None:
    """Load ``.env`` into the process environment (existing variables win).

    Settings reads .env itself, but provider API keys (GROQ_API_KEY, ...) are looked up in
    ``os.environ`` by name, so entry points call this once at startup.
    """
    load_dotenv(override=False)
