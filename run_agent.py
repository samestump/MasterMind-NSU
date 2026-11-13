#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# Imported scientific dependencies must not reserve one thread per host CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from harness.error_diagnostics import DiagnosticBackend, budget_balance
from harness.backend import ALLOWED_MODELS, DEFAULT_MODEL, JiuwenBackend
from harness.local_tokens import LOCAL_USAGE_SOURCE, TOKEN_BUDGET_REFERENCE, LocallyCountedBackend
from harness.runner import Limits, SYSTEM, run
from harness.provenance import IncompatibleWeights, runtime_provenance, validate_weights
from harness.ordering import OrderedTasks, forecast_all


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
PROFILE_KEYS = {"version", "model", "estimator", "estimator_sha256", "train_dataset_sha256",
                "call_delay", "order", "source"}


def load_profile():
    """Return (defaults, problem). A broken profile is reported and ignored, never fatal."""
    location = os.environ.get(PROFILE_ENV)
    if not location:
        return None, None
    try:
        path = Path(location)
        data = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or set(data) - {"local_token_count"} != PROFILE_KEYS
                or data["version"] != 1 or type(data.get("local_token_count", False)) is not bool):
            raise ValueError("profile_fields")
        if data["model"] not in ALLOWED_MODELS or data["order"] not in ("name", "estimator"):
            raise ValueError("profile_values")
        delay = budget(data["call_delay"])
        weights = path.parent / data["estimator"]
        # Hash the parsed JSON, not raw bytes: a Windows checkout turns LF into CRLF.
        content = json.dumps(json.loads(weights.read_text(encoding="utf-8")), sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        if hashlib.sha256(content.encode()).hexdigest() != data["estimator_sha256"]:
            raise ValueError("profile_estimator_sha256")
        return {"model": data["model"], "estimator": weights, "call_delay": delay,
                "train_dataset_sha256": data["train_dataset_sha256"], "order": data["order"],
                "local_token_count": data.get("local_token_count", False)}, None
    except Exception as exc:
        return None, str(exc) if isinstance(exc, ValueError) else type(exc).__name__


def parse_args(argv=None, *, profile=None):
    parser = argparse.ArgumentParser(description="Sequential openJiuwen task harness")
    parser.add_argument("--tasks-dir", required=True, type=Path)
    parser.add_argument("--model", choices=ALLOWED_MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--estimator", type=Path,
                        help="explicit estimator weights; save a forecast before any solver call")
    parser.add_argument("--train-dataset-sha256", help="expected training dataset identity, independent of weights")
    parser.add_argument("--call-delay", type=budget, default=0.0,
                        help="seconds before each model call (part of estimator compatibility)")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--task-timeout", type=positive, default=180)
    parser.add_argument("--run-timeout", type=positive, default=3500)
    parser.add_argument("--tool-timeout", type=positive, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--token-budget", type=budget, default=None,
                        help="weighted-token target for reporting (uses the smaller of this value and BUDGET)")
    parser.add_argument("--reasoning-accounting", choices=("included", "separate"), default="included")
    parser.add_argument("--order", choices=("name", "estimator"),
                        help="estimator: forecast all tasks first, then solve by reward*P(pass)/max(1, cost_q80)")
    parser.add_argument("--local-token-count", action=argparse.BooleanOptionalAction, default=None,
                        help="estimate usage with the bundled tokenizer")
    if profile:
        parser.set_defaults(model=profile["model"], call_delay=profile["call_delay"])
    args = parser.parse_args(argv)
    args.profile_estimator = profile["estimator"] if profile else None
    if args.local_token_count:
        if args.estimator or args.order == "estimator":
            parser.error("--local-token-count cannot use provider estimator weights")
    if profile and not args.local_token_count:
        if args.estimator is None:
            args.estimator = profile["estimator"]
        if args.train_dataset_sha256 is None:
            args.train_dataset_sha256 = profile["train_dataset_sha256"]
        args.order = args.order or profile["order"]
    # The profile's forecast uses public task features before the first request.
    # Local counting changes the source of runtime usage, not those inputs.
    if args.local_token_count is None:
        args.local_token_count = bool(profile and profile.get("local_token_count"))
    if args.local_token_count and args.reasoning_accounting != "included":
        parser.error("--local-token-count already includes reasoning in output")
    args.order = args.order or "name"
    if not math.isfinite(args.call_delay) or args.call_delay < 0:
        parser.error("--call-delay must be non-negative and finite")
    if args.estimator and not args.train_dataset_sha256:
        parser.error("--estimator requires --train-dataset-sha256")
    if args.order == "estimator" and not args.estimator:
        parser.error("--order estimator requires --estimator")
    if "BUDGET" in os.environ:
        try:
            env_budget = budget(os.environ["BUDGET"])
        except (ValueError, argparse.ArgumentTypeError):
            parser.error("BUDGET must be a non-negative finite number")
        args.token_budget = (env_budget if args.token_budget is None
                             else min(env_budget, args.token_budget))
    if not args.tasks_dir.is_dir():
        parser.error("tasks directory does not exist")
    if args.max_steps < 1 or args.max_output_tokens < 1:
        parser.error("step and output limits must be positive")
    if not all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY")):
        parser.error("LLM_BASE_URL and LLM_API_KEY are required")
    return args


def prepare_estimator(args, limits):
    weights = json.loads(args.estimator.read_text(encoding="utf-8"))
    if not isinstance(weights, dict):
        raise ValueError("invalid_estimator_weights")
    current = runtime_provenance(args.model, limits, SYSTEM,
        args.train_dataset_sha256, args.reasoning_accounting, args.call_delay)
    validate_weights(weights, current)
    return weights, current


def main():
    profile, problem = load_profile()
    if problem:
        print(json.dumps({"status": "profile_ignored", "reason": problem}), file=sys.stderr)
    args = parse_args(profile=profile)
    try:
        # Keep the reporting target out of the runner's enforced limits and provenance.
        limits = Limits(
            steps=args.max_steps, task_seconds=args.task_timeout,
            run_seconds=args.run_timeout, tool_seconds=args.tool_timeout,
            output_tokens=args.max_output_tokens)
        weights = current = None
        if args.estimator:
            try:
                weights, current = prepare_estimator(args, limits)
            except Exception as exc:
                if args.estimator != args.profile_estimator:
                    raise  # explicit --estimator stays strict: refuse before any API call
                # The image's own weights do not fit this launch (e.g. another model flag):
                # solve as the plain baseline instead of failing the judge's run.
                print(json.dumps({"status": "estimator_disabled", "reason": str(exc)
                                  if isinstance(exc, IncompatibleWeights) else type(exc).__name__}),
                      file=sys.stderr)
                args.estimator, args.order = None, "name"
        backend_type = LocallyCountedBackend if args.local_token_count else JiuwenBackend
        model_backend = backend_type(args.model, args.reasoning_accounting, call_delay_seconds=args.call_delay)
        backend = DiagnosticBackend(model_backend, args.tasks_dir / "provider_errors.jsonl",
                                    budget_target=args.token_budget)
        tasks, runner_weights = args.tasks_dir.resolve(), weights
        if args.order == "estimator":
            # All forecasts are saved before the first tool or model call; the runner then
            # starts tasks in that order and does not forecast them a second time.
            tasks, runner_weights = OrderedTasks(forecast_all(tasks, weights, current)), None
        summary = asyncio.run(run(tasks, backend, limits, estimator_model=runner_weights,
            train_dataset_sha256=args.train_dataset_sha256, reasoning_accounting=args.reasoning_accounting,
            call_delay_seconds=args.call_delay))
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
    except IncompatibleWeights as exc:
        print(json.dumps({"status": "incompatible_estimator", "reason": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "startup_error", "error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0 if summary["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
