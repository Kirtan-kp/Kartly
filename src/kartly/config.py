"""Application settings, read from environment variables (prefix KARTLY_) and .env."""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KARTLY_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://kartly:kartly@localhost:5432/kartly"
    redis_url: str = "redis://localhost:6379/0"
    models_config_path: str = "config/models.yaml"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_env() -> None:
    """Load ``.env`` into the process environment (existing variables win).

    Settings reads .env itself, but provider API keys (GROQ_API_KEY, ...) are looked up in
    ``os.environ`` by name, so entry points call this once at startup.
    """
    load_dotenv(override=False)
