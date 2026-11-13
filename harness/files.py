import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    directory: Path
    task_id: str
    task_type: str
    instruction: str
    reward: float

    @property
    def workspace(self):
        path = self.directory / "workspace"
        return path if path.is_dir() else self.directory


def load_task(path: Path) -> Task:
    if path.is_symlink() or (path / "task.json").is_symlink():
        raise ValueError("symlink_task")
    data = json.loads((path / "task.json").read_text())
    # Deliberately retain only public fields; never give metadata to tools.
    instruction = data["instruction"]
    if not isinstance(instruction, str) or len(instruction) > 100_000:
        raise ValueError("invalid_instruction")
    return Task(path, str(data.get("task_id", path.name)),
                str(data.get("task_type", "unknown")), instruction, float(data["reward"]))


def safe_files(root: Path):
    if root.is_symlink():
        raise ValueError("symlink_workspace")
    count = size = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(p in {"__pycache__", ".git"} for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError("symlink_file")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("nonregular_file")
        if path.name in {"task.json", "answer.txt", "trajectory.jsonl", "result.json", "prediction.json"}:
            continue
        count += 1
        size += path.stat().st_size
        if count > 2000 or size > 16 * 1024**2:
            raise ValueError("workspace_too_large")
        yield path, rel


def stage_task(task: Task, destination: Path):
    for src, rel in safe_files(task.workspace):
        dst = destination / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def test_hashes(workspace: Path):
    return {str(rel): hashlib.sha256(path.read_bytes()).hexdigest()
            for path, rel in safe_files(workspace)
            if path.name.startswith("test") and path.suffix == ".py"}


def unchanged_tests(workspace: Path, original_tests: dict):
    current = test_hashes(workspace)
    if any(current.get(name) != digest for name, digest in original_tests.items()):
        raise ValueError("tests_modified")
    return current


def atomic_write(path: Path, data: str | bytes, *, mode: int | None = None):
    """Replace and fsync before the next task can start."""
    payload = data.encode() if isinstance(data, str) else data
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".harness-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            if mode is not None:
                os.fchmod(f.fileno(), mode)
            os.fsync(f.fileno())
        os.replace(name, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish(task: Task, workspace: Path, answer: str, original_tests: dict):
    files = list(safe_files(workspace))
    current_tests = unchanged_tests(workspace, original_tests)
    # Validate all paths before copying any changes back to the mounted input.
    for _, rel in files:
        dest = task.workspace / rel
        if dest.is_symlink() or not dest.resolve().is_relative_to(task.workspace.resolve()):
            raise ValueError("unsafe_destination")
    for path, rel in files:
        if str(rel) not in current_tests:
            destination = task.workspace / rel
            try:
                executable = destination.stat().st_mode & 0o111
            except FileNotFoundError:
                executable = 0
            atomic_write(destination, path.read_bytes(), mode=0o644 | executable)
    if task.task_type != "coding":
        atomic_write(task.directory / "answer.txt", answer.strip() + "\n", mode=0o644)


class Journal:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
