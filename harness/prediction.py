"""Internal forecast format; not the unpublished organizer prediction contract."""
import hashlib
import json
import math

from harness.features import SCHEMA_VERSION, public_features
from scripts import estimator
from harness.provenance import LIMITS_TRANSFER_RULE, validate_provenance, validate_weights


def forecast(task, weights, current):
    validate_weights(weights, current)
    snapshot = public_features(task)
    prediction = estimator.predict(weights, dict(snapshot))
    if (set(prediction) != {"cost", "cost_q80", "p_pass"}
            or not all(math.isfinite(v) for v in prediction.values())
            or min(prediction["cost"], prediction["cost_q80"]) <= 0
            or not 0 <= prediction["p_pass"] <= 1):
        raise ValueError("invalid_prediction")
    return {
        "event": "prediction", "format": "internal_pre_execution_v1",
        "task_id": task.task_id, "api_model": current["api_model"],
        "feature_schema_version": SCHEMA_VERSION, "features": snapshot,
        "weights_sha256": hashlib.sha256(json.dumps(
            weights, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "limits": current["limits"], "cost_unit": current["cost_unit"],
        "provenance": current,
        # The weights keep their own collection limits; the forecast is per task under them.
        "estimator_limits": validate_provenance(weights["provenance"])["limits"],
        "limits_transfer_rule": LIMITS_TRANSFER_RULE,
        "prediction": prediction,
    }
