"""Estimator task order for the competition runner, without touching the solver.

VALIDATION-CONTROL confirmed on a separate validation run that starting tasks by
reward * P(pass) / cost_q80 scores more than name order under a binding BUDGET and the
same at 100000; skipping by a threshold was not chosen. Only the start order changes.

The solver core (harness/runner.py and the files hashed into implementation_sha256)
stays byte-identical, so frozen weights and the train/validation evidence remain bound
to the code that produced them. runner.run walks ``sorted(p.parent for p in
tasks_dir.glob("*/task.json"))``; ``OrderedTasks`` returns the same real task
directories (no copies, no symlinks) whose sort key is the forecast rank. A test fails
loudly if the runner stops honouring that order.
"""
import json
from pathlib import Path

from harness.files import Journal, atomic_write, load_task
from harness.prediction import forecast

ORDER_RULE = "reward * p_pass / max(1, cost_q80), descending; ties by task_id; nothing skipped"


def order_value(reward, prediction):
    """Expected reward per conservative (q80) weighted token; the order checked on validation."""
    return float(reward) * prediction["p_pass"] / max(1.0, prediction["cost_q80"])


class _RankedPath(type(Path())):
    """A real task directory that sorts by its precomputed rank instead of its name."""

    def __lt__(self, other):
        return (getattr(self, "rank", 1 << 62), str(self)) < (getattr(other, "rank", 1 << 62), str(other))


class _Hit:
    def __init__(self, parent):
        self.parent = parent


class OrderedTasks:
    """Stands in for tasks_dir: same task.json hits, parents sorted by forecast rank."""

    def __init__(self, directories):
        self.directories = []
        for rank, directory in enumerate(directories):
            ranked = _RankedPath(directory)
            ranked.rank = rank
            self.directories.append(ranked)

    def glob(self, pattern):
        if pattern != "*/task.json":
            raise ValueError("unsupported pattern")
        return [_Hit(directory) for directory in self.directories]

    def resolve(self):
        return self


def forecast_all(tasks_dir, weights, current):
    """Forecast every task before any tool or model call; return directories in run order.

    Predictions are saved to prediction.json and as the first trajectory event, the same
    record the runner writes. A task that cannot be loaded gets no forecast and keeps
    its place after the forecast ones; the runner then fails it exactly as without order.
    """
    directories = sorted(p.parent for p in Path(tasks_dir).glob("*/task.json"))
    values, unloaded = {}, []
    for directory in directories:
        try:
            task = load_task(directory)
        except Exception:
            unloaded.append(directory)
            continue
        prediction = forecast(task, weights, current)
        prediction["order_rule"] = ORDER_RULE
        atomic_write(directory / "prediction.json", json.dumps(prediction, allow_nan=False) + "\n")
        Journal(directory / "trajectory.jsonl").append(prediction)
        values[directory] = (-order_value(task.reward, prediction["prediction"]), task.task_id)
    return sorted(values, key=values.__getitem__) + unloaded
