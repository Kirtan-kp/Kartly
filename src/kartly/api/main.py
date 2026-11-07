"""Build the real app from settings: database, model, gateway, runner, bounded executor.

python -m kartly.api            # serves on 127.0.0.1:8000, docs at /docs
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn
from fastapi import FastAPI

from kartly.api.app import ApiState, create_app
from kartly.api.executor import BoundedExecutor
from kartly.config import get_settings, load_env
from kartly.gateway import ToolGateway
from kartly.guards.claims import ClaimJudge
from kartly.llm import LLMConfigError, ModelRegistry
from kartly.orchestrator import TicketRunner
from kartly.retrieval import EmbeddingError, KnowledgeBaseError
from kartly.retrieval.setup import gateway_tools
from kartly.store.db import make_engine, make_session_factory


def create_app_from_env() -> FastAPI:
    """Raises LLMConfigError or ValueError with a plain message when setup is incomplete."""
    load_env()
    settings = get_settings()
    if not settings.api_key:
        raise ValueError(
            "KARTLY_API_KEY is not set. Generate one with "
            'python -c "import secrets; print(secrets.token_urlsafe(32))" and put it in .env'
        )
    registry = ModelRegistry.from_yaml(settings.models_config_path)
    model_name = settings.default_model or registry.chain("small")[0]
    client = registry.client(model_name)
    router = registry.client(settings.router_model) if settings.router_model else None
    judge_client = (
        registry.client(settings.claim_check_model) if settings.claim_check_model else None
    )
    engine = make_engine()
    factory = make_session_factory(engine)
    runner = TicketRunner(
        factory,
        client,
        ToolGateway(factory, gateway_tools(settings)),
        model_tier=model_name,
        router=router,
        claim_judge=ClaimJudge(judge_client) if judge_client is not None else None,
        context_budget=settings.context_budget_tokens,
    )
    state = ApiState(
        session_factory=factory,
        runner=runner,
        executor=BoundedExecutor(settings.max_workers, settings.max_pending_runs),
        api_key=settings.api_key,
    )

    def cleanup() -> None:
        client.close()
        if router is not None:
            router.close()
        if judge_client is not None:
            judge_client.close()
        engine.dispose()

    return create_app(state, on_shutdown=cleanup)


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
