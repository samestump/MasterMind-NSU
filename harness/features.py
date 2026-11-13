"""Public, pre-execution features shared by collection and forecasting."""
from harness.files import safe_files

SCHEMA_VERSION = 1


def public_features(task):
    files = list(safe_files(task.workspace))
    return {
        "task_type": task.task_type,
        "instruction_characters": len(task.instruction),
        "workspace_files": len(files),
        "workspace_bytes": sum(path.stat().st_size for path, _ in files),
    }
