"""Explicit training/runtime compatibility contract, independent of task outcomes."""
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import re
import math
from dataclasses import asdict
from pathlib import Path

from harness.features import SCHEMA_VERSION

COST_UNIT = "output_tokens + input_tokens / 10"
FEATURE_SCHEMA = {
    "version": SCHEMA_VERSION,
    "public_fields": ["task_type", "instruction_characters", "workspace_files", "workspace_bytes"],
    "vector": ["intercept", "log1p_instruction_characters_or_zero", "log1p_workspace_bytes",
               "workspace_files", "is_search", "is_data_analysis", "is_coding"],
}
FIELDS = {"version", "api_model", "sdk_version", "prompt_sha256", "solver_config",
          "limits", "train_dataset_sha256", "feature_schema", "cost_unit"}


class IncompatibleWeights(ValueError):
    pass


def reject(field, reason="mismatch"):
    raise IncompatibleWeights(f"Incompatible estimator weights: {field} {reason}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_provenance(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        reject("provenance", "missing or incomplete")
    if type(value["version"]) is not int or value["version"] != 1:
        reject("provenance.version", "unsupported")
    for key in ("api_model", "sdk_version"):
        if not isinstance(value[key], str) or not value[key].strip():
            reject("provenance." + key, "missing")
    for key in ("prompt_sha256", "train_dataset_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            reject("provenance." + key, "invalid SHA-256")
    if canonical(value["feature_schema"]) != canonical(FEATURE_SCHEMA):
        reject("provenance.feature_schema", "unsupported")
    if value["cost_unit"] != COST_UNIT:
        reject("provenance.cost_unit", "unsupported")
    config = value["solver_config"]
    if not isinstance(config, dict) or set(config) != {
        "policy", "temperature", "reasoning_accounting", "request_timeout_seconds",
        "max_retries", "call_delay_seconds", "implementation_sha256",
    }:
        reject("provenance.solver_config", "incomplete")
    if not isinstance(config["implementation_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", config["implementation_sha256"]):
        reject("provenance.solver_config.implementation_sha256", "invalid SHA-256")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "steps", "task_seconds", "run_seconds", "tool_seconds", "output_tokens", "token_budget",
    }:
        reject("provenance.limits", "incomplete")
    value = copy.deepcopy(value)
    for key in ("steps", "output_tokens"):
        if type(limits[key]) is not int or limits[key] < 1:
            reject("provenance.limits." + key, "invalid")
    for group, keys in (("limits", ("task_seconds", "run_seconds", "tool_seconds", "token_budget")),
                        ("solver_config", ("temperature", "request_timeout_seconds", "call_delay_seconds"))):
        for key in keys:
            number = value[group][key]
            if key == "token_budget" and number is None:
                continue
            minimum_zero = key in {"temperature", "call_delay_seconds", "token_budget"}
            if (type(number) not in (int, float) or not math.isfinite(number) or
                    (number < 0 if minimum_zero else number <= 0)):
                reject("provenance." + group + "." + key, "invalid")
            value[group][key] = float(number)
    if (not isinstance(config["policy"], str) or not config["policy"] or
            not isinstance(config["reasoning_accounting"], str) or
            config["reasoning_accounting"] not in {"included", "separate"} or
            type(config["max_retries"]) is not int or config["max_retries"] < 0):
        reject("provenance.solver_config", "invalid")
    return value


def runtime_provenance(api_model, limits, prompt, train_dataset_sha256,
                       reasoning_accounting="included", call_delay_seconds=0.0, request_timeout_seconds=120):
    try:
        sdk_version = version("openjiuwen")
    except PackageNotFoundError:
        reject("provenance.sdk_version", "cannot determine installed openjiuwen version")
    sources = {name: hashlib.sha256((Path(__file__).parent / name).read_text(encoding="utf-8").encode()).hexdigest()
               for name in ("runner.py", "backend.py", "tools.py", "sandbox_worker.py", "files.py", "features.py")}
    return validate_provenance({
        "version": 1, "api_model": api_model, "sdk_version": sdk_version,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "solver_config": {"policy": "sequential_baseline_v1", "temperature": 0,
                          "reasoning_accounting": reasoning_accounting, "request_timeout_seconds": request_timeout_seconds,
                          "max_retries": 0, "call_delay_seconds": float(call_delay_seconds),
                          "implementation_sha256": hashlib.sha256(canonical(sources).encode()).hexdigest()},
        "limits": asdict(limits), "train_dataset_sha256": train_dataset_sha256,
        "feature_schema": FEATURE_SCHEMA, "cost_unit": COST_UNIT,
    })


def _first_mismatch(expected, actual, keys):
    for key in sorted(keys):
        if canonical(expected[key]) != canonical(actual[key]):
            if isinstance(expected[key], dict) and isinstance(actual[key], dict):
                for child in sorted(actual[key]):
                    if canonical(expected[key].get(child)) != canonical(actual[key][child]):
                        return "provenance." + key + "." + child
            return "provenance." + key
    return None


def provenance_mismatch(expected, actual):
    """Return the first incompatible field; share normalization with runtime checks.

    Exact equality of every field. Used where two data sets must come from the same
    configuration (collection plan drift, train vs validation comparison); never relaxed.
    """
    expected, actual = validate_provenance(expected), validate_provenance(actual)
    return _first_mismatch(expected, actual, FIELDS)


# RUNTIME-COLLECTION-LIMITS. Weights predict one task's cost and P(pass) from public
# features. Only limits that act inside a single task shape that distribution, so they
# must match exactly. The run-level limits decide which tasks get attempted; inside a
# task they only matter when they cut it short, and such cut rows describe the cut, not
# the task. Hence a run-level limit may differ only in the direction in which the
# collection could not have been cut more than the runtime would be.
PER_TASK_LIMITS = ("steps", "task_seconds", "tool_seconds", "output_tokens")
LIMITS_TRANSFER_RULE = "per_task_exact;run_seconds_runtime_le_trained;token_budget_trained_unbounded_or_equal"


def limits_transfer_mismatch(trained, runtime):
    """Return (field, reason) if weights trained under ``trained`` limits may not run under ``runtime``.

    - steps, task_seconds, tool_seconds, output_tokens: exact.
    - run_seconds: runtime <= trained. A shorter runtime only stops the run earlier; a longer
      one would use weights whose late tasks may have been truncated by the collection deadline.
    - token_budget: trained must be unbounded (no row was capped or stopped by a budget) or equal.
      Any runtime budget, including the competition BUDGET, still caps every call in the runner.
    """
    for key in PER_TASK_LIMITS:
        if canonical(float(trained[key])) != canonical(float(runtime[key])):
            return "provenance.limits." + key, "mismatch (per-task limit must equal collection)"
    if float(runtime["run_seconds"]) > float(trained["run_seconds"]):
        return "provenance.limits.run_seconds", "exceeds the collection run limit"
    if trained["token_budget"] is not None and (
            runtime["token_budget"] is None or float(runtime["token_budget"]) != float(trained["token_budget"])):
        return "provenance.limits.token_budget", "differs from a budget-capped collection"
    return None


def weights_runtime_mismatch(trained, runtime):
    """Weights -> runtime check: every non-limit field exact, limits by the transfer rule."""
    trained, runtime = validate_provenance(trained), validate_provenance(runtime)
    field = _first_mismatch(trained, runtime, FIELDS - {"limits"})
    if field:
        return field, "mismatch"
    return limits_transfer_mismatch(trained["limits"], runtime["limits"])


def validate_weights(weights, current):
    if not isinstance(weights, dict):
        reject("weights", "invalid")
    provenance = validate_provenance(weights.get("provenance"))
    if weights.get("version") != 2 or type(weights.get("use_length")) is not bool:
        reject("weights.version/use_length", "unsupported")
    size = len(FEATURE_SCHEMA["vector"])
    for key in ("feature_means", "feature_stds", "cost_weights", "pass_weights"):
        values = weights.get(key)
        if (not isinstance(values, list) or len(values) != size or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            reject("weights." + key, "incompatible feature dimensions or values")
    if any(v <= 0 for v in weights["feature_stds"]):
        reject("weights.feature_stds", "must be positive")
    for key in ("log_residual_q50", "log_residual_q80"):
        if type(weights.get(key)) not in (int, float) or not math.isfinite(weights[key]):
            reject("weights." + key, "invalid")
    mismatch = weights_runtime_mismatch(provenance, current)
    if mismatch:
        reject(*mismatch)
