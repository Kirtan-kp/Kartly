"""Application settings, read from environment variables (prefix KARTLY_) and .env."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KARTLY_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://kartly:kartly@localhost:5432/kartly"
    redis_url: str = "redis://localhost:6379/0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
