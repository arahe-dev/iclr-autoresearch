from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
import re
import statistics
from typing import Any

from .contract import load_contract
from .generation0 import PLAN_ID, build_plan
from .gpu import EXPECTED_GPU_NAME
from .processes import FileLock
from .provenance import TIMING_BOUNDARIES, frozen_identity, require_hex, validate_code_identity, validate_corpus


RESULT_REQUIRED_FIELDS = (
    "schema_version",
    "plan_id",
    "source_commit",
    "harness_commit",
    "run_id",
    "arm_id",
    "treatment_id",
    "attempt_id",
    "block_id",
    "within_block_position",
    "selected_levels",
    "oracle",
    "latency_ms",
    "real_input_tok_s",
    "decision",
    "peak_GiB", "samples_ms", "warmups", "timed_repetitions", "timing",
    "memory", "provenance", "validation", "admission", "execution",
)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timing boundary must be an ISO-8601 timestamp")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timing boundary must carry a timezone")
    return parsed


def validate_gpu(gpu: Any, locked: dict[str, Any] | None = None) -> None:
    if not isinstance(gpu, dict) or gpu.get("device") != EXPECTED_GPU_NAME:
        raise ValueError("result must report actual target GPU identity")
    if gpu.get("sm") != "sm_120" or re.fullmatch(r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", str(gpu.get("gpu_uuid", ""))) is None:
        raise ValueError("result must report actual CUDA GPU UUID and SM120")
    if _finite_positive(gpu.get("vram_GiB"), "GPU VRAM") < 90:
        raise ValueError("result GPU capacity is below the frozen target")
    if any(not isinstance(gpu.get(k), str) or not gpu[k] for k in ("torch", "cuda")):
        raise ValueError("result must identify PyTorch and CUDA versions")
    if locked is not None:
        if gpu["gpu_uuid"] != locked.get("uuid") or gpu["device"] != locked.get("name", locked.get("device")):
            raise ValueError("worker GPU identity disagrees with the locked physical GPU")
        # The supervisor uses idle nvidia-smi memory.free as its torch-free
        # capacity source; nvidia-smi reports MiB while CUDA reports bytes.
        if abs(gpu["vram_GiB"] - _finite_positive(locked.get("vram_GiB"), "locked GPU VRAM")) > 1 / 1024:
            raise ValueError("worker GPU capacity disagrees with the locked physical GPU")


def _validate_timing(record: dict[str, Any]) -> None:
    if type(record["warmups"]) is not int or record["warmups"] != 2 or type(record["timed_repetitions"]) is not int or record["timed_repetitions"] != 5:
        raise ValueError("result must preserve two warmups and five timed repetitions")
    samples = record["samples_ms"]
    if not isinstance(samples, list) or len(samples) != 5:
        raise ValueError("result must contain all five timing samples")
    for sample in samples:
        _finite_positive(sample, "timing sample")
    if not math.isclose(statistics.median(samples), record["latency_ms"], rel_tol=1e-12):
        raise ValueError("latency_ms must equal the median of the timing samples")
    if not math.isclose(record["real_input_tok_s"], 65536 * 1000 / record["latency_ms"], rel_tol=1e-12):
        raise ValueError("throughput must use the frozen 65536 real input tokens")
    timing = record["timing"]
    if not isinstance(timing, dict) or timing.get("boundaries") != TIMING_BOUNDARIES or timing.get("real_input_tokens") != 65536:
        raise ValueError("timing boundaries differ from the exact B32 F+B protocol")
    if type(timing.get("valid_targets")) is not int or not 0 < timing["valid_targets"] <= 65536:
        raise ValueError("timing must report the actual valid-target denominator")
    intervals = timing.get("intervals")
    if not isinstance(intervals, list) or len(intervals) != 7:
        raise ValueError("timing must record boundaries and peaks for both warmups and all timed repetitions")
    previous_ns = 0
    previous_end = _timestamp(record["validation"].get("completed_before_timing_at"))
    peaks = []
    for index, interval in enumerate(intervals):
        if not isinstance(interval, dict) or interval.get("phase") != (f"warmup_{index + 1}" if index < 2 else f"timing_{index - 1}"):
            raise ValueError("timing intervals are not in the frozen measurement order")
        start, end = (_timestamp(interval.get(key)) for key in ("started_at", "ended_at"))
        if start < previous_end or end < start:
            raise ValueError("timing must follow validation and have ordered boundaries")
        start_ns, end_ns = (interval.get(key) for key in ("start_monotonic_ns", "end_monotonic_ns"))
        if type(start_ns) is not int or type(end_ns) is not int or not previous_ns < start_ns < end_ns:
            raise ValueError("timing monotonic boundaries are invalid")
        elapsed = _finite_positive(interval.get("elapsed_ms"), "interval elapsed_ms")
        if index >= 2 and elapsed != samples[index - 2]:
            raise ValueError("interval elapsed_ms disagrees with timing samples")
        peaks.append(_finite_positive(interval.get("peak_GiB"), "interval peak_GiB"))
        previous_ns, previous_end = end_ns, end
    if max(peaks) != record["peak_GiB"]:
        raise ValueError("peak memory must be the maximum over all measurement intervals")


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result contains a nonfinite number")
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item)
    elif isinstance(value, list):
        for item in value:
            _reject_nonfinite(item)


def validate_result(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("result record must be a JSON object")
    missing = [key for key in RESULT_REQUIRED_FIELDS if key not in record]
    if missing:
        raise ValueError(f"missing result fields: {missing}")
    _reject_nonfinite(record)
    if type(record["schema_version"]) is not int or record["schema_version"] != 2 or record["plan_id"] != PLAN_ID:
        raise ValueError("result does not belong to Generation-0 plan")
    require_hex(record["source_commit"], 40, "source_commit")
    require_hex(record["harness_commit"], 40, "harness_commit")
    require_hex(record["attempt_id"], 32, "attempt_id")
    for field in ("run_id", "arm_id", "decision"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if isinstance(record["block_id"], bool) or not isinstance(record["block_id"], int) or record["block_id"] < 1:
        raise ValueError("block_id must be a positive integer")
    if isinstance(record["within_block_position"], bool) or not isinstance(record["within_block_position"], int) or record["within_block_position"] < 1:
        raise ValueError("within_block_position must be a positive integer")
    if not isinstance(record["selected_levels"], dict):
        raise ValueError("selected_levels must be an object")
    if not isinstance(record["oracle"], dict) or record["oracle"].get("status") != "PASS":
        raise ValueError("oracle correctness must pass before a result can be appended")
    _finite_positive(record["latency_ms"], "latency_ms")
    _finite_positive(record["real_input_tok_s"], "real_input_tok_s")
    _finite_positive(record["peak_GiB"], "peak_GiB")
    plan = build_plan()
    slot = next((slot for slot in plan["run_order"] if slot["run_id"] == record["run_id"]), None)
    if slot is None or any(record[key] != slot[key] for key in ("arm_id", "block_id", "within_block_position")):
        raise ValueError("result is not a slot in the constrained frozen run order")
    treatment_id = "G0-00" if slot["arm_id"] == "CONTROL" else slot["arm_id"]
    arm = next(arm for arm in plan["arms"] if arm["arm_id"] == treatment_id)
    if record["treatment_id"] != treatment_id or record["selected_levels"] != arm["patch_spec"]["selected_levels"]:
        raise ValueError("result treatment differs from its frozen factor assignment")
    if record["decision"] != "MEASURED_G0_B32_FB":
        raise ValueError("Generation-0 measurements do not constitute promotion")
    oracle = record["oracle"]
    for key in ("same_document_reset", "tied_qk_backward", "initial_loss_gate"):
        if oracle.get(key) != "PASS":
            raise ValueError(f"oracle {key} must pass")
    for key in ("forward_rel_l2", "dq_rel_l2", "dv_rel_l2", "projection_rel_l2", "projection_grad_rel_l2", "forward_max_abs"):
        value = oracle.get(key)
        maximum = 3e-3 if key == "forward_max_abs" else 3e-4
        if type(value) not in (float, int) or not 0 <= value <= maximum:
            raise ValueError(f"oracle {key} failed its exact FP32 tolerance")
    initial_loss = _finite_positive(oracle.get("initial_loss"), "initial_loss")
    if abs(initial_loss - load_contract()["training"]["initial_loss_gate"]) > 1e-2:
        raise ValueError("initial loss failed the frozen tolerance")
    validation = record["validation"]
    if not isinstance(validation, dict) or any(validation.get(key) != "PASS" for key in (
        "status", "frozen_corpus", "frozen_contract", "selected_treatment", "initial_loss")) or validation.get("oracle_before_timing") is not True:
        raise ValueError("worker validation gates must explicitly pass before timing")
    _validate_timing(record)
    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("result provenance must be an object")
    for key, commit_field in (("candidate", "source_commit"), ("harness", "harness_commit")):
        identity = provenance.get(key)
        validate_code_identity(identity)
        if identity["commit"] != record[commit_field]:
            raise ValueError(f"{key} commit differs from result provenance")
    frozen = frozen_identity(plan)
    if provenance.get("frozen") != frozen:
        raise ValueError("result frozen champion/contract/config/plan identity drifted")
    for identity in (provenance["candidate"], provenance["harness"]):
        files = identity["files"]
        for name, key in (("configs/generation_0_screening.json", "config_sha256"),
                          ("vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py", "canonical_source_sha256")):
            if files.get(name) != frozen[key]:
                raise ValueError(f"source file identity disagrees with frozen {name}")
    validate_corpus(provenance.get("corpus"))
    execution = record["execution"]
    if not isinstance(execution, dict) or execution.get("candidate_commit") != record["source_commit"] or execution.get("attempt_id") != record["attempt_id"] or execution.get("champion_immutable") is not True:
        raise ValueError("execution provenance is missing or inconsistent")
    if type(execution.get("pid")) is not int or execution["pid"] <= 0:
        raise ValueError("execution must identify its actual worker PID")
    gpu = execution.get("gpu")
    if not isinstance(execution.get("supervisor_gpu"), dict) or not re.fullmatch(r"codex/(generation0|experiment)/.+", str(execution.get("candidate_branch", ""))):
        raise ValueError("execution must identify the supervisor GPU and named candidate branch")
    validate_gpu(gpu, execution.get("supervisor_gpu"))
    memory = record["memory"]
    if not isinstance(memory, dict) or memory.get("status") != "PASS" or memory.get("peak_GiB") != record["peak_GiB"] or memory.get("capacity_GiB") != gpu["vram_GiB"] or record["peak_GiB"] >= gpu["vram_GiB"]:
        raise ValueError("result memory gate is missing or inconsistent")
    admission = record["admission"]
    if not isinstance(admission, dict) or admission.get("status") != "PASS" or admission.get("worker_exited") is not True or admission.get("gpu_released") is not True or type(admission.get("returncode")) is not int or admission["returncode"] != 0:
        raise ValueError("supervisor admission requires successful exit and GPU release")
    require_hex(admission.get("manifest_identity_sha256"), 64, "manifest identity SHA256")
    if _timestamp(admission.get("accepted_at")) < _timestamp(record["timing"]["intervals"][-1]["ended_at"]):
        raise ValueError("admission predates timing completion")


def strict_json_loads(text: str) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique_object)
    _reject_nonfinite(value)
    return value


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Read and validate every JSONL record; a partial/corrupt ledger is fatal."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"ledger contains an unterminated record at line {line_number}")
            if not line.strip():
                raise ValueError(f"ledger contains a blank line at line {line_number}")
            try:
                value = strict_json_loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ledger line {line_number} is not valid JSON: {exc.msg}") from exc
            validate_result(value)
            run_id = value["run_id"]
            if run_id in seen:
                raise ValueError(f"ledger contains duplicate run_id: {run_id}")
            seen.add(run_id)
            records.append(value)
    return records


def existing_run_ids(path: Path) -> set[str]:
    return {record["run_id"] for record in read_ledger(path)}


def append_result(path: Path, record: dict[str, Any]) -> None:
    """Append one validated record; duplicate run IDs and rewrites are rejected."""
    validate_result(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with FileLock(lock_path, {"kind": "result-ledger", "ledger": str(path.resolve())}):
        if record["run_id"] in existing_run_ids(path):
            raise ValueError(f"run_id already exists; append-only ledger refuses overwrite: {record['run_id']}")
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
