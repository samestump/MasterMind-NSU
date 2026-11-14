#!/usr/bin/env python3
import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

# Imported scientific dependencies must not reserve one thread per host CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from harness.abstention import run_evidence
from harness.error_diagnostics import DiagnosticBackend, budget_balance
from harness.backend import ALLOWED_MODELS, JiuwenBackend
from harness.local_tokens import LOCAL_USAGE_SOURCE, TOKEN_BUDGET_REFERENCE, LocallyCountedBackend
from harness.runner import Limits


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def budget(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be non-negative and finite")
    return result


# The Docker image supplies the profile for the judge's --tasks-dir invocation.
PROFILE_ENV = "TOKEN_PREDICTION_PROFILE"
PROFILE_KEYS = {"version", "model", "call_delay", "local_token_count", "evidence_verification"}


def load_profile():
    """Load the shipped judge profile for both Docker and direct CLI use."""
    default_path = Path(__file__).resolve().parent / "competition" / "profile.json"
    path = Path(os.environ.get(PROFILE_ENV, default_path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or set(data) != PROFILE_KEYS
            or type(data["version"]) is not int or data["version"] != 2
            or data["model"] not in ALLOWED_MODELS
            or type(data["local_token_count"]) is not bool
            or data["evidence_verification"] != "triggered"):
        raise ValueError("invalid_competition_profile")
    data["call_delay"] = budget(data["call_delay"])
    return data


def parse_args(argv=None, *, profile):
    parser = argparse.ArgumentParser(description="Sequential openJiuwen task harness with triggered verification")
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--model", choices=ALLOWED_MODELS, default=profile["model"])
    parser.add_argument("--call-delay", type=budget, default=profile["call_delay"],
                        help="seconds before each model call")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--task-timeout", type=positive, default=180)
    parser.add_argument("--run-timeout", type=positive, default=3500)
    parser.add_argument("--tool-timeout", type=positive, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--token-budget", type=budget, default=None,
                        help="weighted-token target for reporting (uses the smaller of this value and BUDGET)")
    parser.add_argument("--reasoning-accounting", choices=("included", "separate"), default="included")
    parser.add_argument("--local-token-count", action=argparse.BooleanOptionalAction,
                        default=profile["local_token_count"],
                        help="estimate usage with the bundled tokenizer")
    args = parser.parse_args(argv)
    if args.local_token_count and args.reasoning_accounting != "included":
        parser.error("--local-token-count already includes reasoning in output")
    if "BUDGET" in os.environ:
        try:
            env_budget = budget(os.environ["BUDGET"])
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("BUDGET must be a non-negative finite number")
        args.token_budget = (env_budget if args.token_budget is None
                             else min(env_budget, args.token_budget))
    if not args.tasks_dir.is_dir():
        parser.error("tasks directory does not exist")
    if args.max_steps < 1 or args.max_output_tokens < 64:
        parser.error("--max-steps must be positive and --max-output-tokens must be at least 64")
    if not all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY")):
        parser.error("LLM_BASE_URL and LLM_API_KEY are required")
    return args


def main():
    try:
        profile = load_profile()
    except (OSError, ValueError, TypeError, argparse.ArgumentTypeError) as exc:
        print(json.dumps({"status": "invalid_profile", "error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    args = parse_args(profile=profile)
    try:
        # The judge enforces the budget. The local counter supplies diagnostics.
        limits = Limits(
            steps=args.max_steps, task_seconds=args.task_timeout,
            run_seconds=args.run_timeout, tool_seconds=args.tool_timeout,
            output_tokens=args.max_output_tokens)
        backend_type = LocallyCountedBackend if args.local_token_count else JiuwenBackend
        model_backend = backend_type(args.model, args.reasoning_accounting, call_delay_seconds=args.call_delay)
        backend = DiagnosticBackend(model_backend, args.tasks_dir / "provider_errors.jsonl",
                                    budget_target=args.token_budget)
        summary = asyncio.run(run_evidence(args.tasks_dir.resolve(), backend, limits,
            evidence_verification=profile["evidence_verification"]))
        summary["token_budget"] = args.token_budget
        accounted = summary["weighted_tokens"]
        if args.local_token_count:
            summary["usage_source"] = LOCAL_USAGE_SOURCE
            summary["estimated_input_tokens"] = model_backend.estimated_input_tokens
            summary["estimated_output_tokens"] = model_backend.estimated_output_tokens
            summary["calls_without_reasoning_text"] = model_backend.calls_without_reasoning_text
            summary["output_reserve_tokens"] = model_backend.output_reserve_tokens
            summary["unreconciled_input_tokens"] = model_backend.estimated_input_tokens - sum(
                result["input_tokens"] for result in summary["results"]
            )
            accounted += summary["unreconciled_input_tokens"] / 10
            summary["reference_weighted_tokens"] = TOKEN_BUDGET_REFERENCE
            summary["weighted_tokens_over_reference"] = round(max(0.0, accounted - TOKEN_BUDGET_REFERENCE), 3)
        if args.token_budget is not None:
            summary.update(budget_balance(args.token_budget, accounted))
    except Exception as exc:
        print(json.dumps({"status": "startup_error", "error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0 if summary["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
