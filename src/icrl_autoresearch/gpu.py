"""Torch-free NVIDIA device discovery and ownership gates.

The campaign supervisor must not import torch: importing CUDA in the parent
would keep an allocator/context alive between candidate processes.  Everything
in this module uses ``nvidia-smi`` and an advisory lock keyed by the physical
GPU UUID.  The candidate worker performs the authoritative torch/SM preflight.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any


EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
MIN_VRAM_MIB = 90 * 1024
FREE_MEMORY_TOLERANCE_MIB = 256


class NvidiaSmiError(RuntimeError):
    """The host cannot provide a trustworthy NVIDIA inventory."""


@dataclass(frozen=True)
class GpuSnapshot:
    uuid: str
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    driver: str

    @property
    def vram_gib(self) -> float:
        return self.total_mib / 1024.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "device": self.name,
            "name": self.name,
            "vram_GiB": self.vram_gib,
            "memory_total_MiB": self.total_mib,
            "memory_used_MiB": self.used_mib,
            "memory_free_MiB": self.free_mib,
            "driver": self.driver,
            "driver_version": self.driver,
        }


@dataclass(frozen=True)
class ComputeProcess:
    gpu_uuid: str
    pid: int
    used_mib: int | None

    def as_dict(self) -> dict[str, Any]:
        return {"gpu_uuid": self.gpu_uuid, "pid": self.pid, "used_gpu_memory_MiB": self.used_mib}


def _smi_path() -> str:
    path = shutil.which("nvidia-smi")
    if not path:
        raise NvidiaSmiError("nvidia-smi is not available on this host")
    return path


def _run_smi(arguments: list[str], timeout: float = 10.0) -> str:
    command = [_smi_path(), *arguments]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NvidiaSmiError(f"nvidia-smi query failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise NvidiaSmiError(f"nvidia-smi exited {completed.returncode}: {detail}")
    return completed.stdout


def _csv_rows(output: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        stripped = [cell.strip() for cell in row]
        if stripped and any(stripped):
            rows.append(stripped)
    return rows


def _integer(value: str, field: str) -> int:
    try:
        return int(value.split()[0].replace(",", ""))
    except (ValueError, IndexError) as exc:
        raise NvidiaSmiError(f"nvidia-smi returned an invalid {field}: {value!r}") from exc


def list_gpus() -> list[GpuSnapshot]:
    output = _run_smi([
        "--query-gpu=uuid,name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    snapshots: list[GpuSnapshot] = []
    for row in _csv_rows(output):
        if len(row) != 6:
            raise NvidiaSmiError(f"nvidia-smi returned an unexpected GPU row: {row!r}")
        snapshots.append(GpuSnapshot(
            uuid=row[0],
            name=row[1],
            total_mib=_integer(row[2], "total memory"),
            used_mib=_integer(row[3], "used memory"),
            free_mib=_integer(row[4], "free memory"),
            driver=row[5],
        ))
    if not snapshots:
        raise NvidiaSmiError("nvidia-smi returned no GPUs")
    return snapshots


def list_compute_processes(gpu_uuid: str | None = None) -> list[ComputeProcess]:
    arguments = [
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    if gpu_uuid:
        arguments[0:0] = ["-i", gpu_uuid]
    try:
        output = _run_smi(arguments)
    except NvidiaSmiError as exc:
        # Some driver versions return a nonzero status for an empty process
        # table.  Re-querying the GPU inventory is the safe distinction from
        # a genuinely unavailable nvidia-smi command.
        if "No running processes" in str(exc):
            return []
        raise
    processes: list[ComputeProcess] = []
    for row in _csv_rows(output):
        if len(row) < 2 or row[0].lower().startswith("no running processes"):
            continue
        if len(row) != 3:
            raise NvidiaSmiError(f"nvidia-smi returned an unexpected compute row: {row!r}")
        used = None if row[2].upper() in {"N/A", "[N/A]"} else _integer(row[2], "process memory")
        processes.append(ComputeProcess(row[0], _integer(row[1], "process pid"), used))
    return processes


def select_target_gpu(*, gpu_uuid: str | None = None, gpu_index: int | None = None) -> GpuSnapshot:
    snapshots = list_gpus()
    if gpu_uuid:
        selected = [gpu for gpu in snapshots if gpu.uuid == gpu_uuid]
        if len(selected) != 1:
            raise NvidiaSmiError(f"requested GPU UUID was not found: {gpu_uuid}")
        gpu = selected[0]
    elif gpu_index is not None:
        if gpu_index < 0 or gpu_index >= len(snapshots):
            raise NvidiaSmiError(f"requested GPU index is out of range: {gpu_index}")
        gpu = snapshots[gpu_index]
    else:
        candidates = [gpu for gpu in snapshots if gpu.name == EXPECTED_GPU_NAME]
        if len(candidates) != 1:
            raise NvidiaSmiError(
                "target GPU selection is ambiguous; pass --gpu-uuid or --gpu-index "
                f"(matching devices: {[gpu.uuid for gpu in candidates]})"
            )
        gpu = candidates[0]
    if gpu.name != EXPECTED_GPU_NAME:
        raise NvidiaSmiError(f"target execution requires {EXPECTED_GPU_NAME}; found {gpu.name}")
    if gpu.total_mib < MIN_VRAM_MIB:
        raise NvidiaSmiError(f"target GPU has less than 90 GiB VRAM: {gpu.total_mib / 1024:.2f} GiB")
    return gpu


def assert_idle_gpu(gpu: GpuSnapshot) -> None:
    processes = list_compute_processes(gpu.uuid)
    if processes:
        raise NvidiaSmiError(
            f"target GPU {gpu.uuid} is already in use by compute PIDs "
            f"{[process.pid for process in processes]}"
        )


def wait_gpu_released(
    gpu: GpuSnapshot,
    *,
    baseline_free_mib: int,
    timeout_seconds: float = 15.0,
    tolerance_mib: int = FREE_MEMORY_TOLERANCE_MIB,
) -> GpuSnapshot:
    """Require worker PIDs gone and memory returned near the pre-run baseline."""
    deadline = time.monotonic() + timeout_seconds
    last: GpuSnapshot = gpu
    while time.monotonic() <= deadline:
        current = next((item for item in list_gpus() if item.uuid == gpu.uuid), None)
        if current is None:
            raise NvidiaSmiError(f"target GPU disappeared during cleanup: {gpu.uuid}")
        last = current
        if not list_compute_processes(gpu.uuid) and current.free_mib >= baseline_free_mib - tolerance_mib:
            return current
        time.sleep(0.25)
    processes = list_compute_processes(gpu.uuid)
    raise NvidiaSmiError(
        f"GPU was not released after {timeout_seconds:.1f}s: "
        f"compute_pids={[process.pid for process in processes]}, "
        f"free_MiB={last.free_mib}, baseline_free_MiB={baseline_free_mib}"
    )


def gpu_lock_path(gpu_uuid: str) -> Path:
    key = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest()[:32]
    return Path(tempfile.gettempdir()) / "icrl-autoresearch-gpu-locks" / f"{key}.lock"


def worker_environment(gpu: GpuSnapshot, *, corpus_root: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    # UUID selection remains physical even when CUDA exposes it as logical 0.
    environment["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    environment["ICRL_G0_GPU_UUID"] = gpu.uuid
    if corpus_root is not None:
        environment["ICRL_CORPUS_ROOT"] = str(corpus_root.resolve())
    return environment
