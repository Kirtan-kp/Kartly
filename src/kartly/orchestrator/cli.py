"""Run one real ticket through the agent loop and print the stored trace.

    python -m kartly.orchestrator.cli --model groq_small "Where is my order KRT-100002?"
    python -m kartly.orchestrator.cli --model local_small --customer C-0001 "Where is KRT-100002?"
    python -m kartly.orchestrator.cli --model groq_small --customer C-0002 "Where is KRT-100002?"

The last example asks about an order that belongs to another customer: the answer should be
"not found", and the trace's audit rows record the attempt. Needs the database from the
quickstart (``docker compose up -d``, ``alembic upgrade head``, ``python -m kartly.store.seed``).
Exit codes: 0 drafted, 1 escalated to a human, 2 configuration or database problem.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence

from sqlalchemy.exc import SQLAlchemyError

from kartly.config import get_settings, load_env
from kartly.gateway import ToolGateway
from kartly.llm import LLMConfigError, ModelRegistry
from kartly.orchestrator.service import TicketRunner, UnknownCustomerError
from kartly.orchestrator.trace import format_trace, load_trace
from kartly.retrieval import EmbeddingError, KnowledgeBaseError
from kartly.retrieval.setup import gateway_tools
from kartly.sanitize import clean_str
from kartly.store.db import make_engine, make_session_factory

DEFAULT_CUSTOMER = "C-0001"
DEFAULT_MESSAGE = "Hi, where is my order KRT-100002?"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one ticket through the agent loop.")
    parser.add_argument(
        "message", nargs="?", default=DEFAULT_MESSAGE, help="the customer's message"
    )
    parser.add_argument("--model", required=True, help="model name from config/models.yaml")
    parser.add_argument("--customer", default=DEFAULT_CUSTOMER, help="customer id (default C-0001)")
    parser.add_argument("--router", help="a model that labels the ticket first (config name)")
    parser.add_argument("--config", help="path to models.yaml")
    args = parser.parse_args(argv)

    load_env()
    settings = get_settings()
    try:
        registry = ModelRegistry.from_yaml(args.config or settings.models_config_path)
        engine = make_engine()
        factory = make_session_factory(engine)
        with (
            registry.client(args.model) as client,
            contextlib.ExitStack() as stack,
        ):
            router = stack.enter_context(registry.client(args.router)) if args.router else None
            runner = TicketRunner(
                factory,
                client,
                ToolGateway(factory, gateway_tools(settings)),
                model_tier=args.model,
                router=router,
                context_budget=settings.context_budget_tokens,
            )
            ticket_id = runner.create_ticket(args.customer, args.message)
            print(f"Running {args.model} for customer {args.customer} ...", flush=True)
            summary = runner.run(ticket_id)
            with factory() as session:
                trace = load_trace(session, summary.run_id)
    except (LLMConfigError, KnowledgeBaseError, EmbeddingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except UnknownCustomerError as exc:
        print(
            f"error: {exc}. Seed the database: python -m kartly.store.seed --reset", file=sys.stderr
        )
        return 2
    except SQLAlchemyError as exc:
        print(
            f"error: database problem ({type(exc).__name__}): {clean_str(str(exc), 200)}\n"
            "Is it running and migrated? docker compose up -d; alembic upgrade head; "
            "python -m kartly.store.seed --reset",
            file=sys.stderr,
        )
        return 2

    assert trace is not None
    print(format_trace(trace))
    print()
    if summary.answer is not None:
        print(f"Draft reply:\n{summary.answer}")
        return 0
    print(f"Escalated to a human: {summary.stop_reason} ({summary.detail})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
