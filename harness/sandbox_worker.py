"""Linux tool subprocess: Landlock filesystem rules and a seccomp deny filter.

Fails closed if the kernel cannot enforce these restrictions. The controller
starts this file with a clean environment, a fresh process group and a timeout.
No task metadata or API credentials are present in this process.
"""

import ctypes
import errno
import os
import platform
import resource
import sys
from pathlib import Path


def restrict(workspace: Path, scratch: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)

    def check(value):
        if value < 0:
            raise OSError(ctypes.get_errno(), "sandbox setup failed")
        return value

    # Landlock syscall numbers are shared by x86_64 and aarch64.
    machine = platform.machine()
    if machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("unsupported sandbox architecture")
    abi = check(libc.syscall(444, 0, 0, 1))
    if abi < 3:
        raise RuntimeError("Landlock ABI >= 3 is required")
    handled = (1 << 15) - 1  # all filesystem rights up to TRUNCATE
    readonly = (1 << 0) | (1 << 2) | (1 << 3)  # execute, read file/dir
    writable = handled
    ruleset = ctypes.c_uint64(handled)
    fd = check(libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0))

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    def allow(path: Path, rights: int):
        if not path.exists():
            return
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            if not path.is_dir():
                rights &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
            rule = PathRule(rights, pfd)
            check(libc.syscall(445, fd, 1, ctypes.byref(rule), 0))
        finally:
            os.close(pfd)

    try:
        for path in {Path(p).resolve() for p in (
            "/usr", "/bin", "/lib", "/lib64", sys.base_prefix, sys.prefix,
            "/etc/ld.so.cache", "/etc/localtime",
        )}:
            allow(path, readonly)
        allow(Path("/dev/null"), (1 << 1) | (1 << 2))
        allow(Path("/dev/urandom"), 1 << 2)
        allow(workspace, writable)
        allow(scratch, writable)
        check(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        check(libc.syscall(446, fd, 0))
    finally:
        os.close(fd)

    # Deny networking, parent process inspection and escaping our process group.
    # Check audit arch first; reject x32 syscall aliases on x86_64 as well.
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    if machine == "x86_64":
        arch = 0xC000003E
        denied = [41, 42, 53, 62, 90, 91, 92, 93, 94, 101, 109, 112, 132, 133,
                  155, 161, 165, 166, 188, 189, 190, 197, 198, 199, 200, 234, 235,
                  259, 260, 261, 268, 280, 452,
                  250, 272, 298, 303, 304, 308, 310, 311, 321, 424, 425, 426, 427, 438, 440]
    else:
        arch = 0xC00000B7
        denied = [5, 6, 7, 14, 15, 16, 33, 39, 40, 41, 51, 52, 53, 54, 55, 88,
                  97, 117, 129, 130, 131, 154, 157, 198, 199, 203, 452,
                  219, 241, 264, 265, 268, 270, 271, 280, 424, 425, 426, 427, 438, 440]
    instructions = [Filter(0x20, 0, 0, 4), Filter(0x15, 1, 0, arch),
                    Filter(0x06, 0, 0, 0x80000000), Filter(0x20, 0, 0, 0)]
    if machine == "x86_64":
        instructions += [Filter(0x35, 0, 1, 0x40000000), Filter(0x06, 0, 0, 0x80000000)]
    for nr in denied:
        instructions += [Filter(0x15, 0, 1, nr), Filter(0x06, 0, 0, 0x00050000 | errno.EPERM)]
    instructions.append(Filter(0x06, 0, 0, 0x7FFF0000))
    filters = (Filter * len(instructions))(*instructions)
    program = Program(len(instructions), filters)
    check(libc.prctl(22, 2, ctypes.byref(program), 0, 0))  # SECCOMP_MODE_FILTER


def limit_processes():
    # In a PID namespace /proc cannot show other processes for the same real
    # host UID. Lowering RLIMIT_NPROC from that incomplete count can prohibit
    # every fork for --user 1000. Prefer the actual per-container cgroup cap.
    for path in ("/sys/fs/cgroup/pids.max", "/sys/fs/cgroup/pids/pids.max"):
        try:
            if 0 < int(Path(path).read_text().strip()) <= 512:
                return
        except (OSError, ValueError):
            pass
    # Local runs without a tight cgroup cap keep a per-UID fallback.
    threads = 0
    for process in Path("/proc").iterdir():
        try:
            if process.name.isdigit() and process.stat().st_uid == os.getuid():
                threads += len(list((process / "task").iterdir()))
        except (FileNotFoundError, PermissionError):
            continue
    ceiling = max(64, threads + 64)
    _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if hard != resource.RLIM_INFINITY:
        ceiling = min(ceiling, hard)
    resource.setrlimit(resource.RLIMIT_NPROC, (ceiling, ceiling))


def main():
    workspace, scratch = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    command = sys.stdin.read(65537)
    if len(command) > 65536:
        raise ValueError("command too large")
    os.chdir(workspace)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (25, 26))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    limit_processes()
    restrict(workspace, scratch)
    os.execve("/bin/bash", ["bash", "--noprofile", "--norc", "-c", command], dict(os.environ))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("SANDBOX_SETUP_FAILED:" + type(exc).__name__, file=sys.stderr)
        sys.exit(125)
