"""Application settings, read from environment variables (prefix KARTLY_) and .env."""

from functools import lru_cache
from typing import Literal

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
    # Models to try, in order, when the default model keeps failing (comma separated names from
    # models.yaml, e.g. "groq_large"). The default model is always retried first (ADR-050).
    fallback_models: str = ""
    # Calls to one model (the first and its retries) before moving on to the next one.
    llm_max_attempts: int = 3
    # A model that fails this many calls in a row is skipped for llm_cooldown_s seconds.
    llm_failure_threshold: int = 3
    llm_cooldown_s: float = 30.0
    # A model from models.yaml that labels each ticket (intent, tier) before the resolver runs,
    # sending Tier 3 to a human. Empty means no model router (the rules still run).
    router_model: str | None = None
    # A model that reads each policy reply and holds it if a statement is not in the retrieved
    # text (ADR-045). Empty means no claim check (the citation and number check still runs).
    claim_check_model: str | None = None
    # How much each write tier may do: off (the model never sees the tool), shadow (a call is
    # checked and becomes a proposal a human must approve) or auto (it runs now). Shadow by
    # default, so nothing changes in the store until someone chooses otherwise (ADR-046).
    tier1_mode: Literal["off", "shadow", "auto"] = "shadow"
    tier2_mode: Literal["off", "shadow", "auto"] = "shadow"
    # Money limits for issue_refund (ADR-047): a refund that takes an order's total refunded above
    # the auto limit, or a customer's refunds over 24 hours above the daily cap, needs a human.
    refund_auto_limit_inr: int = 1000
    refund_daily_cap_inr: int = 5000
    # Policy search (the search_policy tool). Set kb_dir (for example "kb") to enable it; it needs
    # an Ollama embedding model (ollama pull nomic-embed-text). min_score is a dense cosine
    # cut-off: below it the tool reports "no policy found" (ADR-037).
    kb_dir: str | None = None
    embed_model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"
    policy_min_score: float = 0.67
    # Token budget for the first prompt of a run (system prompt + tools + thread), estimated.
    context_budget_tokens: int = 6000
    max_workers: int = 2  # tickets processed at the same time
    max_pending_runs: int = 20  # running + queued; beyond this the API answers 503
    # Durable queue (ADR-052). "inprocess": the API runs tickets in its own threads (default).
    # "db": the API writes a job and workers run it; jobs survive a restart and are retried.
    queue_mode: Literal["inprocess", "db"] = "inprocess"
    embedded_workers: int = 2  # worker threads started inside the API in db mode (0 = none)
    job_max_attempts: int = 3
    job_lease_s: float = 300.0
    worker_poll_s: float = 1.0
    # Alerts (ADR-055). Rules run over the metrics every ``alert_interval_s`` seconds.
    alerts_enabled: bool = True
    alert_interval_s: float = 60.0
    alert_window_s: float = 900.0
    # Off by default: when on, a critical alert may freeze writes or switch the agent off.
    alert_auto_actions: bool = False
    alert_webhook_url: str | None = None  # an https URL that accepts {"text": "..."}


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_env() -> None:
    """Load ``.env`` into the process environment (existing variables win).

    Settings reads .env itself, but provider API keys (GROQ_API_KEY, ...) are looked up in
    ``os.environ`` by name, so entry points call this once at startup.
    """
    load_dotenv(override=False)
