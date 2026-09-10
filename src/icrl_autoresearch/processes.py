"""Process ownership and locks for the single-GPU campaign supervisor.

This module deliberately has no torch import. The supervisor must not retain
a CUDA context while experiment workers come and go.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable, Iterator


class LockBusy(RuntimeError):
    pass


class FileLock:
    """Advisory process lock; never unlink the file or race on stale PIDs."""

    def __init__(self, path: Path, owner: dict[str, Any]) -> None:
        self.path = path
        self.owner = owner
        self.handle: Any = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            if os.name == "nt":
                import msvcrt
                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write("\n")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise LockBusy(f"Campaign lock is already held: {self.path}") from exc
        self.handle = handle
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({**self.owner, "pid": os.getpid()}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def fileno(self) -> int:
        return int(self.handle.fileno())

    def __exit__(self, *args: Any) -> None:
        if self.handle is not None:
            # Closing our reference releases the lock once any inheriting
            # worker references have also closed. Explicit LOCK_UN would
            # release it too early after a supervisor failure.
            self.handle.close()
            self.handle = None


@contextmanager
def interrupt_on_signal() -> Iterator[None]:
    """Make SIGTERM take the same cleanup path as notebook Ctrl+C."""
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def terminate_tree(process: subprocess.Popen[Any], grace_seconds: float = 3.0) -> None:
    """Terminate only the process group this supervisor created."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False,
            )
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()  # Reap the root even if descendants remain.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=3)


@dataclass(frozen=True)
class ProcessResult:
    pid: int
    returncode: int
    tail: str
    elapsed_seconds: float
    timed_out: bool


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: float,
    emit: Callable[[str], None] = lambda line: print(line, end="", flush=True),
    started: Callable[[int], None] | None = None,
    inherit_fds: tuple[int, ...] = (),
) -> ProcessResult:
    """Stream durable output, bound silent hangs, and reap the entire worker.

    File-backed stdout avoids pipe-buffer deadlocks and an inherited pipe
    keeping the supervisor blocked after the worker has exited.
    """
    if timeout_seconds <= 0:
        raise ValueError("Worker timeout must be positive")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: deque[str] = deque(maxlen=200)
    begin = time.monotonic()
    timed_out = False
    with log_path.open("x", encoding="utf-8") as sink:
        options: dict[str, Any] = {}
        if os.name == "posix":
            options.update(start_new_session=True, pass_fds=inherit_fds)
        elif os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, shell=False,
            stdout=sink, stderr=subprocess.STDOUT, **options,
        )
        try:
            if started:
                started(process.pid)
            with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                while True:
                    for line in reader:
                        tail.append(line)
                        emit(line)
                    code = process.poll()
                    if code is not None:
                        # The last write can land between EOF and poll().
                        for line in reader:
                            tail.append(line)
                            emit(line)
                        break
                    if time.monotonic() - begin >= timeout_seconds:
                        timed_out = True
                        break
                    time.sleep(0.05)
        finally:
            # Covers success (including stray compiler descendants), failure,
            # timeout, a failed callback, SIGTERM, and KeyboardInterrupt.
            terminate_tree(process)
        sink.flush()
        os.fsync(sink.fileno())
    return ProcessResult(
        process.pid, int(process.returncode), "".join(tail),
        time.monotonic() - begin, timed_out,
    )
