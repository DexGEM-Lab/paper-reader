#!/usr/bin/env python3
"""Run one Pi turn in the shared container; reap even detached tool children."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def pid_path(turn_id: str) -> Path:
    if uuid.UUID(turn_id).hex != turn_id:
        raise ValueError("invalid turn id")
    return Path("/tmp") / f"paper-reader-turn-{turn_id}.pid"


def descendants(parent: int) -> set[int]:
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm may contain spaces or parentheses: fields after its final ')'.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {parent}
    while True:
        children = {pid for pid, ppid in parents.items() if ppid in found}
        if children <= found:
            return found - {parent}
        found |= children


def cleanup() -> None:
    # Subreaper adoption catches grandchildren even when a tool called setsid().
    for _ in range(100):
        children = descendants(os.getpid())
        if not children:
            return
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        time.sleep(0.01)
    raise RuntimeError("could not reap all turn children")


def cancel(turn_id: str) -> None:
    path = pid_path(turn_id)
    try:
        pid = int(path.read_text())
        # A stale PID file must not terminate a later unrelated process.
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if turn_id.encode() in argv and b"run" in argv:
            os.kill(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        pass


def main() -> int:
    action, turn_id = sys.argv[1:3]
    path = pid_path(turn_id)
    if action == "cancel":
        cancel(turn_id)
        return 0
    if action != "run":
        raise ValueError("unknown action")
    timeout = float(sys.argv[3])
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    # PR_SET_CHILD_SUBREAPER: do not leak detached bash/python grandchildren.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable child subreaper")

    def terminated(_signum, _frame):
        raise InterruptedError("turn cancelled")

    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    path.write_text(str(os.getpid()))
    cwd = Path.cwd()
    for name in (".home", ".tmp"):
        (cwd / name).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(HOME=str(cwd / ".home"), TMPDIR=str(cwd / ".tmp"), PI_CODING_AGENT_DIR="/pi-home")
    try:
        process = subprocess.Popen(sys.argv[4:], env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Pi agent timed out after {timeout:g} seconds", file=sys.stderr)
            return 124
        except InterruptedError:
            return 130
    finally:
        cleanup()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
