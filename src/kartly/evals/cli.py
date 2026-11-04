"""Run the golden cases against a model.

    python -m kartly.evals --model local_small
    python -m kartly.evals --model groq_small --pause 20      # stay under Groq's free-tier limit
    python -m kartly.evals --model groq_small --only idor-friends-order --show-replies
    python -m kartly.evals --list

Needs no database and no Docker: each case runs on its own in-memory copy of the mock store.
Results go to ``evals/runs/`` (one JSON file per run, one row appended to ``experiments.csv``).
Exit codes: 0 every case passed, 1 a case failed or errored, 2 a configuration problem.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path

from kartly.config import get_settings, load_env
from kartly.evals.cases import Case, CaseFileError, load_cases
from kartly.evals.runner import (
    CaseResult,
    format_result,
    format_summary,
    run_suite,
    save_run,
    summarize,
)
from kartly.llm import LLMConfigError, ModelRegistry
from kartly.orchestrator.prompts import DEFAULT_POLICY_PROMPT, POLICY_RULES
from kartly.retrieval import EmbeddingError, KnowledgeBaseError
from kartly.retrieval.setup import build_policy_tools
from kartly.tools import default_tools

DEFAULT_CASES = "evals/golden"
DEFAULT_OUT = "evals/runs"


def _select(cases: list[Case], only: Sequence[str] | None) -> list[Case]:
    if not only:
        return cases
    known = {c.id for c in cases}
    unknown = [i for i in only if i not in known]
    if unknown:
        raise CaseFileError(
            f"unknown case id(s): {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )
    return [c for c in cases if c.id in set(only)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden eval cases against a model.")
    parser.add_argument("--model", help="model name from config/models.yaml")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="a case file or a directory of them")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these case ids")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case, to see variance")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between runs")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where results are written")
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="turn the injection screen off, to measure the model's own resistance",
    )
    parser.add_argument(
        "--router",
        help="also run the routing classifier with this model (a Tier 3 label escalates; "
        "cases that expect a draft for such tickets will then fail)",
    )
    parser.add_argument(
        "--kb",
        help="also give the model the search_policy tool over this knowledge-base directory "
        "(for example kb; needs Ollama with the embedding model)",
    )
    parser.add_argument(
        "--policy-prompt",
        default=DEFAULT_POLICY_PROMPT,
        choices=sorted(POLICY_RULES),
        help="which wording of the policy rules to use with --kb (default p1)",
    )
    parser.add_argument("--no-save", action="store_true", help="print only; write no files")
    parser.add_argument("--note", default="", help="a short label stored in the experiment log")
    parser.add_argument("--show-replies", action="store_true", help="also print passing replies")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument("--config", help="path to models.yaml")
    args = parser.parse_args(argv)

    try:
        cases = _select(load_cases(args.cases), args.only)
    except CaseFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.list:
        for case in cases:
            print(f"{case.id:<28} tier {case.tier}  {case.intent:<13} {case.description}")
        return 0
    if not args.model:
        print("error: --model is required (see --list for cases)", file=sys.stderr)
        return 2
    if args.repeat < 1 or args.pause < 0:
        print("error: --repeat must be at least 1 and --pause cannot be negative", file=sys.stderr)
        return 2

    needs_kb = [c.id for c in cases if any("search_policy" in m.any_of for m in c.expect.calls)]
    if needs_kb and not args.kb:
        print(
            f"error: {len(needs_kb)} case(s) expect the search_policy tool "
            f"({', '.join(needs_kb[:3])}{', ...' if len(needs_kb) > 3 else ''}); add --kb kb "
            "(and run Ollama with the embedding model)",
            file=sys.stderr,
        )
        return 2

    load_env()
    try:
        registry = ModelRegistry.from_yaml(args.config or get_settings().models_config_path)
        client = registry.client(args.model)
        router = registry.client(args.router) if args.router else None
        tools = None
        if args.kb:
            settings = get_settings()
            tools = default_tools() + build_policy_tools(
                args.kb,
                embed_model=settings.embed_model,
                ollama_url=settings.ollama_url,
                min_score=settings.policy_min_score,
                cache_path=Path(args.out) / "embedding_cache.json",
            )
    except (LLMConfigError, KnowledgeBaseError, EmbeddingError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    def show(result: CaseResult) -> None:
        print(format_result(result), flush=True)
        if args.show_replies and result.status == "pass" and result.answer:
            print(f"        reply: {result.answer!r}")

    print(
        f"Running {len(cases)} case(s) x {args.repeat} against {args.model} "
        "(each on a fresh in-memory store) ...",
        flush=True,
    )
    with client, contextlib.ExitStack() as stack:
        if router is not None:
            stack.enter_context(router)
        results = run_suite(
            cases,
            lambda _case: client,
            model_name=args.model,
            repeat=args.repeat,
            pause_s=args.pause,
            screen=not args.no_screen,
            router=router,
            tools=tools,
            policy_prompt=args.policy_prompt,
            on_result=show,
        )
    summary = summarize(
        results, cases, args.model, policy=bool(args.kb), policy_prompt=args.policy_prompt
    )
    print()
    print(format_summary(summary))
    if not args.no_save:
        note = args.note or " ".join(
            part
            for part in (
                f"router={args.router}" if args.router else "",
                f"kb={args.kb}" if args.kb else "",
            )
            if part
        )
        path = save_run(results, summary, Path(args.out), note=note)
        print(f"\nSaved {path} and a row in {Path(args.out) / 'experiments.csv'}")
    return 0 if summary.failed == 0 and summary.errored == 0 else 1
