"""Standalone JSON CLI for the online router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_router.errors import ConfigurationError, RouterError
from llm_router.models import RouteDecision
from llm_router.router import Router


def _load_registry(path: str | None) -> Any:
    """Load whichever registry factory the configured registry exposes."""

    from llm_router.registry import ModelRegistry

    if path is None:
        path = str(Path(__file__).resolve().parent / "data" / "models.yaml")
    candidate = Path(path)
    for factory_name in ("from_yaml", "from_file", "load"):
        factory = getattr(ModelRegistry, factory_name, None)
        if factory is not None:
            return factory(candidate)
    return ModelRegistry(candidate)


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationError(f"invalid CLI arguments: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(description="Route a task to the lowest-cost capable model")
    parser.add_argument("task", nargs="?", help="task text to classify")
    parser.add_argument("--task", dest="task_option", help="task text (alternative to positional task)")
    parser.add_argument("--tier", choices=("efficiency", "balanced", "intelligence"), default="balanced")
    parser.add_argument("--provider")
    parser.add_argument("--providers", nargs="+", help="allowed providers")
    parser.add_argument("--max-cost", type=float)
    parser.add_argument("--max-latency", type=float)
    parser.add_argument("--registry", help="registry YAML path")
    return parser


def main(argv: list[str] | None = None, *, router: Router | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        task = args.task_option if args.task_option is not None else args.task
        if not task:
            raise ConfigurationError("task is required")
        active_router = router
        if active_router is None:
            active_router = Router(_load_registry(args.registry))
        try:
            decision = active_router.route(
                task,
                tier=args.tier,
                provider=args.provider,
                providers=args.providers,
                max_cost=args.max_cost,
                max_latency=args.max_latency,
            )
        finally:
            if router is None:
                active_router.close()
        print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")))
        return 0 if decision.error is None else 1
    except RouterError as exc:
        # Construction/configuration failures happen before Router.route can
        # create its envelope, so the CLI emits the same error object shape.
        decision = RouteDecision.model_validate({"error": exc.as_route_error().model_dump(mode="json")})
        print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    sys.exit(main())
