"""Original public inputs and budget estimates shared by evidence policies."""
import math

from harness.files import safe_files


MAX_CONTEXT_BYTES = 24000
RESERVED_SOURCES = {'task.json', 'answer.txt', 'trajectory.jsonl', 'result.json', 'prediction.json'}


def public_snapshot(task):
    sources = {'$instruction': task.instruction}
    # In the directory-fallback layout only top-level controller files are metadata.
    workspace = task.workspace
    unavailable = [
        path.relative_to(workspace).as_posix() for path in sorted(workspace.rglob('*'))
        if path.is_file() and path.name in RESERVED_SOURCES
        and (workspace != task.directory or len(path.relative_to(workspace).parts) > 1)
        and not any(part in {'.git', '__pycache__'} for part in path.relative_to(workspace).parts)
    ]
    for path, relative in safe_files(workspace):
        name = relative.as_posix()
        if name == '$instruction':
            unavailable.append(name)
            continue
        try:
            sources[name] = path.read_bytes().decode('utf-8')
        except UnicodeDecodeError:
            unavailable.append(name)
    complete = not unavailable and sum(
        len(content.encode()) for name, content in sources.items() if name != '$instruction'
    ) <= MAX_CONTEXT_BYTES
    return sources, unavailable, complete


def prompt_bytes(messages):
    return sum(len(message['content'].encode()) + 64 for message in messages) + 512


def reserved_output_cap(backend, prompt, max_tokens, remaining_tokens):
    if remaining_tokens is None:
        return max_tokens
    counter = getattr(backend, 'token_counter', None)
    input_tokens = counter.input_tokens(prompt) if counter is not None else prompt_bytes(prompt)
    return min(max_tokens, math.floor(remaining_tokens - input_tokens / 10))
