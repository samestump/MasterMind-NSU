"""Predict weighted token cost and pass probability with frozen linear models."""

import math

TYPES = ("math", "search", "data_analysis", "coding")


def features(row, use_length):
    values = [1.0,
              math.log1p(row["instruction_characters"]) if use_length else 0.0,
              math.log1p(row["workspace_bytes"]),
              float(row["workspace_files"])]
    values += [1.0 if row["task_type"] == kind else 0.0 for kind in TYPES[1:]]
    return values


def predict(model, row):
    raw = features(row, model["use_length"])
    means = model["feature_means"]
    stds = model["feature_stds"]
    scaled = [(value - mean) / std for value, mean, std in zip(raw, means, stds)]
    log_cost = sum(weight * value for weight, value in zip(model["cost_weights"], scaled))
    logit = sum(weight * value for weight, value in zip(model["pass_weights"], scaled))
    logit = max(-30.0, min(30.0, logit))
    return {"cost": math.exp(log_cost + model["log_residual_q50"]),
            "cost_q80": math.exp(log_cost + model["log_residual_q80"]),
            "p_pass": 1 / (1 + math.exp(-logit))}
