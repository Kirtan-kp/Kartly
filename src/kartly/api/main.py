"""Build the real app from settings: database, model, gateway, runner, bounded executor.

python -m kartly.api            # serves on 127.0.0.1:8000, docs at /docs
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from sqlalchemy.orm import Session, sessionmaker

from kartly.alerts import AlertLoop, AlertManager, combined, log_notifier, webhook_notifier
from kartly.api.app import ApiState, create_app
from kartly.api.executor import BoundedExecutor
from kartly.config import Settings, get_settings, load_env
from kartly.controls import Controls
from kartly.gateway import ToolGateway, modes_from_settings
from kartly.guards.claims import ClaimJudge
from kartly.llm import LLMConfigError, ModelRegistry, RetryPolicy
from kartly.llm.config import ModelsFile, load_models_file
from kartly.orchestrator import TicketRunner
from kartly.queue import JobQueue
from kartly.retrieval import EmbeddingError, KnowledgeBaseError
from kartly.retrieval.setup import gateway_tools
from kartly.reviewers import Reviewers
from kartly.store.db import make_engine, make_session_factory
from kartly.worker import Worker, WorkerPool


@dataclass
class Runtime:
    """Everything the API and the workers share, built once from settings."""

    settings: Settings
    factory: sessionmaker[Session]
    runner: TicketRunner
    gateway: ToolGateway
    controls: Controls
    queue: JobQueue
    close: Callable[[], None]


def build_runtime(*, require_api_key: bool = False) -> Runtime:
    """Raises LLMConfigError or ValueError with a plain message when setup is incomplete."""
    load_env()
    settings = get_settings()
    if require_api_key and not settings.api_key:
        raise ValueError(
            "KARTLY_API_KEY is not set. Generate one with "
            'python -c "import secrets; print(secrets.token_urlsafe(32))" and put it in .env'
        )
    registry = ModelRegistry.from_yaml(settings.models_config_path)
    model_name = settings.default_model or registry.chain("small")[0]
    fallbacks = [n.strip() for n in settings.fallback_models.split(",") if n.strip()]
    client = registry.resilient(
        model_name,
        fallbacks,
        policy=RetryPolicy(
            max_attempts=settings.llm_max_attempts,
            failure_threshold=settings.llm_failure_threshold,
            cooldown_s=settings.llm_cooldown_s,
        ),
    )
    router = registry.client(settings.router_model) if settings.router_model else None
    judge_client = (
        registry.client(settings.claim_check_model) if settings.claim_check_model else None
    )
    engine = make_engine()
    factory = make_session_factory(engine)
    controls = Controls(factory)
    gateway = ToolGateway(
        factory,
        gateway_tools(settings),
        modes=modes_from_settings(settings.tier1_mode, settings.tier2_mode),
        controls=controls,
    )
    runner = TicketRunner(
        factory,
        client,
        gateway,
        model_tier=model_name,
        router=router,
        claim_judge=ClaimJudge(judge_client) if judge_client is not None else None,
        controls=controls,
        context_budget=settings.context_budget_tokens,
    )
    queue = JobQueue(factory, max_attempts=settings.job_max_attempts, lease_s=settings.job_lease_s)

    def close() -> None:
        client.close()
        if router is not None:
            router.close()
        if judge_client is not None:
            judge_client.close()
        engine.dispose()

    return Runtime(settings, factory, runner, gateway, controls, queue, close)


def _load_models(path: str) -> ModelsFile | None:
    """The price list for the cost report; the API still starts without it."""
    try:
        return load_models_file(path)
    except LLMConfigError:
        return None


def create_app_from_env() -> FastAPI:
    rt = build_runtime(require_api_key=True)
    settings = rt.settings
    db_mode = settings.queue_mode == "db"
    workers = (
        WorkerPool(
            [Worker(rt.queue, rt.runner.run, rt.factory) for _ in range(settings.embedded_workers)],
            settings.worker_poll_s,
        )
        if db_mode and settings.embedded_workers > 0
        else None
    )
    manager = (
        AlertManager(
            rt.factory,
            controls=rt.controls,
            notifier=combined(
                log_notifier,
                *(
                    [webhook_notifier(settings.alert_webhook_url)]
                    if settings.alert_webhook_url
                    else []
                ),
            ),
            auto_actions=settings.alert_auto_actions,
            window_s=settings.alert_window_s,
        )
        if settings.alerts_enabled
        else None
    )
    state = ApiState(
        session_factory=rt.factory,
        runner=rt.runner,
        executor=BoundedExecutor(settings.max_workers, settings.max_pending_runs),
        api_key=settings.api_key or "",
        gateway=rt.gateway,
        controls=rt.controls,
        queue=rt.queue if db_mode else None,
        max_backlog=settings.max_pending_runs,
        workers=workers,
        alerts=manager,
        reviewers=Reviewers(
            rt.factory,
            session_ttl_s=settings.session_ttl_s,
            session_idle_s=settings.session_idle_s,
        ),
        models=_load_models(settings.models_config_path),
        cookie_secure=settings.cookie_secure,
        docs_enabled=settings.docs_enabled,
        session_ttl_s=settings.session_ttl_s,
        alert_loop=AlertLoop(manager, settings.alert_interval_s) if manager else None,
    )
    return create_app(state, on_shutdown=rt.close)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Kartly API.")
    parser.add_argument("--host", default="127.0.0.1", help="default: this machine only")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        app = create_app_from_env()
    except (LLMConfigError, ValueError, KnowledgeBaseError, EmbeddingError) as exc:
        print(f"error: {exc}")
        return 2
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
