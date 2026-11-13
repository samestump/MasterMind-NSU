import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


def run_bash(workspace: Path, command: str, timeout: float = 30) -> dict:
    """Run a command with no access to task metadata, credentials or network."""
    with tempfile.TemporaryDirectory(prefix="harness-tool-") as tmp:
        scratch = Path(tmp)
        env = {
            "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(scratch), "TMPDIR": str(scratch), "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        }
        # A file, rather than PIPE, bounds RAM even for noisy or hostile commands.
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen(
                [sys.executable, "-I", str(Path(__file__).with_name("sandbox_worker.py")),
                 str(workspace), str(scratch)], stdin=subprocess.PIPE, stdout=output,
                stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            timed_out = False
            try:
                proc.communicate(command.encode(), timeout=max(0.01, timeout))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Kill background children as well as a timed-out foreground tool.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            output.seek(0)
            data = output.read(16001)
        text = data[:16000].decode("utf-8", errors="replace")
        if len(data) > 16000:
            text += "\n[output truncated]"
        if proc.returncode == 125 and "SANDBOX_SETUP_FAILED:" in text:
            raise RuntimeError("sandbox_unavailable")
        return {"exit_code": proc.returncode, "timed_out": timed_out, "output": text}
