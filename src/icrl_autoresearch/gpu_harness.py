"""Fail-closed, single-GPU supervisor for the Generation-0 campaign.

The supervisor owns ordering, provenance, process lifetime, GPU ownership,
resume state, and append-only result admission.  It intentionally never
imports torch.  CUDA and the exact model/treatment gates live in the child
candidate runner, which is launched as a fresh process for every slot.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import traceback
import uuid
from typing import Any, Callable, Iterable

from .contract import REPO_ROOT, canonical_json, contract_sha256, source_sha256
from .generation0 import (
    BASE_COMMIT,
    PLAN_ID,
    CANONICAL_SOURCE,
    CANONICAL_SOURCE_SHA256,
    CHAMPION_BRANCH,
    assert_control_treatment,
    build_plan,
)
from .gpu import (
    EXPECTED_GPU_NAME,
    GpuSnapshot,
    assert_idle_gpu,
    gpu_lock_path,
    select_target_gpu,
    wait_gpu_released,
    worker_environment,
)
from .processes import FileLock, ProcessResult, interrupt_on_signal, run_process
from .results import append_result, read_ledger, validate_result, validate_gpu, strict_json_loads
from .provenance import code_identity, corpus_fingerprint, frozen_identity, json_sha256, validate_corpus


CHAMPION_COMMIT_FULL = BASE_COMMIT
MANIFEST_SCHEMA_VERSION = 3
STATE_SCHEMA_VERSION = 1
WARMUPS = 2
REPETITIONS = 5
DEFAULT_TIMEOUT_SECONDS = 60 * 60
SHELL_METACHARS = set(";&|<>`$\n\r")
FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(candidate_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(candidate_root), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def assert_candidate_worktree(candidate_root: Path) -> dict[str, str]:
    """Verify an immutable, clean, named candidate worktree."""
    candidate = candidate_root.resolve()
    if not candidate.is_dir():
        raise ValueError(f"candidate worktree does not exist: {candidate}")
    if candidate == REPO_ROOT.resolve():
        raise ValueError("GPU execution refuses to run in the planning/champion workspace")

    top = Path(_git(candidate, "rev-parse", "--show-toplevel")).resolve()
    if top != candidate:
        raise ValueError(f"candidate root is not the selected Git worktree: {candidate}")
    branch = _git(candidate, "symbolic-ref", "--quiet", "--short", "HEAD")
    if not branch:
        raise ValueError("GPU execution requires a named candidate branch; detached HEAD is refused")
    if branch == CHAMPION_BRANCH or branch.startswith("codex/champion/"):
        raise ValueError(f"champion branch is immutable and cannot be an execution target: {branch}")
    if not branch.startswith(("codex/experiment/", "codex/generation0/")):
        raise ValueError("candidate branch must use codex/experiment/ or codex/generation0/ prefix")
    dirty = _git(candidate, "status", "--porcelain")
    if dirty:
        raise ValueError("candidate worktree must be clean and committed before execution")
    try:
        champion_commit = _git(candidate, "rev-parse", "--verify", CHAMPION_BRANCH)
    except RuntimeError as exc:
        raise ValueError("candidate repository does not contain the immutable champion ref") from exc
    if champion_commit != CHAMPION_COMMIT_FULL:
        raise ValueError("immutable champion ref does not resolve to the certified 40-character commit")

    source = candidate / CANONICAL_SOURCE
    if not source.exists() or source_sha256(source) != CANONICAL_SOURCE_SHA256:
        raise ValueError("candidate canonical source is not byte-equivalent to experiment 0000")
    commit = _git(candidate, "rev-parse", "HEAD")
    if not FULL_SHA_RE.fullmatch(commit):
        raise ValueError(f"candidate HEAD is not a full Git SHA: {commit!r}")
    return {
        "candidate_root": str(candidate),
        "candidate_branch": branch,
        "candidate_commit": commit,
        "champion_commit": champion_commit,
        "base_commit": BASE_COMMIT,
        "canonical_source_sha256": CANONICAL_SOURCE_SHA256,
    }


def assert_target_gpu() -> dict[str, Any]:
    """Legacy-compatible torch-free target GPU check."""
    return select_target_gpu().as_dict()


def _arm_by_id(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {arm["arm_id"]: arm for arm in plan["arms"]}


def run_context(plan: dict[str, Any], slot: dict[str, Any]) -> dict[str, Any]:
    arms = _arm_by_id(plan)
    arm_id = slot["arm_id"]
    if arm_id == "CONTROL":
        selected_levels = dict(plan["control_reference"]["factor_levels"])
    else:
        if arm_id not in arms:
            raise ValueError(f"run order references unknown arm: {arm_id}")
        selected_levels = dict(arms[arm_id]["patch_spec"]["selected_levels"])
    return {
        "plan_id": plan["plan_id"],
        "run_id": slot["run_id"],
        "arm_id": arm_id,
        "treatment_id": "G0-00" if arm_id == "CONTROL" else arm_id,
        "kind": slot["kind"],
        "block_id": slot["block_id"],
        "within_block_position": slot["within_block_position"],
        "selected_levels": selected_levels,
        "oracle_required": True,
        "champion_immutable": True,
    }


def _format_command(template: str, context: dict[str, Any], candidate: dict[str, str], attempt_id: str | None = None) -> str:
    values = {
        "plan_id": context["plan_id"],
        "run_id": context["run_id"],
        "arm_id": context["arm_id"],
        "block_id": context["block_id"],
        "within_block_position": context["within_block_position"],
        # Forward slashes keep the argv template portable to Windows while
        # remaining valid to Python and Git there.
        "candidate_root": candidate["candidate_root"].replace("\\", "/"),
        "selected_levels_json": json.dumps(context["selected_levels"], sort_keys=True),
        "attempt_id": attempt_id or "{attempt_id}",
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"unknown command-template placeholder: {exc.args[0]}") from exc


def _command_argv(command: str) -> list[str]:
    """Parse an argv-only command and reject shell composition."""
    quote: str | None = None
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if quote is None and char in SHELL_METACHARS:
            raise ValueError("command-template may contain argv quoting only; shell operators are refused")
    if quote is not None:
        raise ValueError("command-template contains an unterminated quote")
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid argv command-template: {exc}") from exc
    if not argv:
        raise ValueError("command-template produced an empty argv")
    return argv


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("GPU command emitted no JSON result")
    try:
        value = strict_json_loads(text)
    except json.JSONDecodeError:
        value = None
        for line in reversed(text.splitlines()):
            try:
                value = strict_json_loads(line)
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise ValueError("GPU command stdout did not contain a JSON result")
    if not isinstance(value, dict):
        raise ValueError("GPU command result must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_fingerprint(corpus_root: Path | None) -> dict[str, Any]:
    return corpus_fingerprint(corpus_root)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"attempt log contains a blank line at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"attempt log line {line_number} is invalid JSON") from exc
            if not isinstance(value, dict) or value.get("event") not in {"begin", "started", "end", "recovered"}:
                raise ValueError(f"attempt log line {line_number} has an invalid event")
            events.append(value)
    return events


def _active_attempts(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    active: dict[str, dict[str, Any]] = {}
    for event in events:
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise ValueError("attempt event is missing attempt_id")
        if event["event"] == "begin":
            if attempt_id in active:
                raise ValueError(f"attempt began twice without completion: {attempt_id}")
            active[attempt_id] = event
        elif event["event"] == "started":
            if attempt_id not in active:
                raise ValueError(f"attempt started without a begin event: {attempt_id}")
            active[attempt_id] = {**active[attempt_id], "pid": event.get("pid")}
        elif event["event"] in {"end", "recovered"}:
            if attempt_id not in active:
                raise ValueError(f"attempt ended without a begin event: {attempt_id}")
            active.pop(attempt_id, None)
    return active


def _manifest_identity(
    plan: dict[str, Any],
    candidate: dict[str, str],
    gpu: dict[str, Any],
    command_template: str,
    corpus: dict[str, Any],
    timeout_seconds: float,
    preflight: dict[str, Any] | None,
    ledger: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "plan_id": plan["plan_id"],
        "plan_sha256": hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest(),
        "contract_sha256": contract_sha256(),
        "source_commit": candidate["candidate_commit"],
        "harness_commit": provenance["harness"]["commit"],
        "provenance": provenance,
        "candidate_branch": candidate["candidate_branch"],
        "canonical_source_sha256": candidate["canonical_source_sha256"],
        "champion_commit": provenance["frozen"]["champion_commit"],
        "champion_branch": provenance["frozen"]["champion_branch"],
        "gpu": {
            "uuid": gpu.get("uuid"),
            "name": gpu.get("name", gpu.get("device")),
            "driver_version": gpu.get("driver_version", gpu.get("driver")),
        },
        "corpus": corpus,
        "command_template": command_template,
        "ledger_path": str(ledger.resolve()),
        "benchmark_limits": {
            "microbatch": 32,
            "warmups": WARMUPS,
            "timed_repetitions": REPETITIONS,
            "timeout_seconds": timeout_seconds,
        },
    }
    if preflight is not None:
        identity["runtime"] = {
            "torch": preflight.get("torch"),
            "cuda": preflight.get("cuda"),
            "sm": preflight.get("sm"),
            "device": preflight.get("device"),
        }
    return identity


def _load_json_object(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _validate_resume(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
    candidate: dict[str, str],
    state: dict[str, Any] | None,
) -> None:
    expected_slots = plan["run_order"]
    if len(records) > len(expected_slots):
        raise ValueError("ledger contains more records than the frozen run order")
    for index, record in enumerate(records):
        expected = expected_slots[index]
        context = run_context(plan, expected)
        if record.get("run_id") != context["run_id"]:
            raise ValueError(
                f"ledger is not an ordered prefix: line {index + 1} has {record.get('run_id')!r}, "
                f"expected {context['run_id']!r}"
            )
        for key in ("plan_id", "arm_id", "treatment_id", "block_id", "within_block_position", "selected_levels"):
            if record.get(key) != context[key]:
                raise ValueError(f"ledger context drift for {context['run_id']}: {key}")
        if record.get("source_commit") != candidate["candidate_commit"]:
            raise ValueError(f"ledger source commit drift for {context['run_id']}")
        execution = record.get("execution")
        if not isinstance(execution, dict) or execution.get("candidate_commit") != candidate["candidate_commit"]:
            raise ValueError(f"ledger execution provenance is missing for {context['run_id']}")
    if state is not None:
        if state.get("schema_version") != STATE_SCHEMA_VERSION or state.get("plan_id") != plan["plan_id"]:
            raise ValueError("campaign state does not belong to this plan")
        completed = state.get("completed_run_ids")
        ledger_completed = [record["run_id"] for record in records]
        if not isinstance(completed, list) or completed != ledger_completed[:len(completed)]:
            raise ValueError("campaign state and result ledger disagree")
        if len(completed) < len(ledger_completed):
            # A process can append a fully validated result and be killed
            # before the following state.json atomic replace.  The ordered
            # ledger is authoritative in that one recoverable direction.
            state["completed_run_ids"] = ledger_completed
            state["failed_run_id"] = None


def _require_oracle_and_memory(payload: dict[str, Any], gpu: dict[str, Any], run_id: str) -> None:
    oracle = payload.get("oracle")
    if not isinstance(oracle, dict) or oracle.get("status") != "PASS":
        raise ValueError(f"{run_id}: oracle.status must be PASS before timing is accepted")
    for gate in ("same_document_reset", "tied_qk_backward"):
        if oracle.get(gate) != "PASS":
            raise ValueError(f"{run_id}: {gate} gate did not pass")
    memory = payload.get("memory")
    if isinstance(memory, dict) and memory.get("status") not in (None, "PASS"):
        raise ValueError(f"{run_id}: memory gate did not pass: {memory.get('status')!r}")
    if payload.get("memory_ok") is False or payload.get("memory_status") not in (None, "PASS"):
        raise ValueError(f"{run_id}: runner reported a memory violation")
    peak = payload.get("peak_GiB")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(float(peak)) or float(peak) <= 0:
        raise ValueError(f"{run_id}: result must include a positive finite peak_GiB")
    capacity = gpu.get("vram_GiB")
    if capacity is None or float(peak) >= float(capacity):
        raise ValueError(f"{run_id}: peak_GiB={peak} exceeds target device capacity")


def _validate_worker_payload(
    payload: dict[str, Any],
    context: dict[str, Any],
    candidate: dict[str, str],
    gpu: dict[str, Any],
    attempt_id: str,
    manifest_identity: dict[str, Any],
    process_result: ProcessResult,
) -> dict[str, Any]:
    for key in ("plan_id", "run_id", "arm_id", "treatment_id", "block_id", "within_block_position", "selected_levels", "source_commit", "attempt_id", "harness_commit", "provenance", "execution"):
        if key not in payload:
            raise ValueError(f"{context['run_id']}: worker result is missing {key}")
    for key in ("plan_id", "run_id", "arm_id", "treatment_id", "block_id", "within_block_position", "selected_levels"):
        if payload[key] != context[key]:
            raise ValueError(f"{context['run_id']}: worker {key} disagrees with run order")
    if payload["source_commit"] != candidate["candidate_commit"]:
        raise ValueError(f"{context['run_id']}: worker source_commit is not the candidate commit")
    if payload["attempt_id"] != attempt_id:
        raise ValueError(f"{context['run_id']}: worker attempt_id does not match the supervisor attempt")
    if payload.get("schema_version") != 2:
        raise ValueError("worker must emit the complete version-2 provenance protocol")
    if payload["provenance"] != manifest_identity["provenance"] or payload["harness_commit"] != manifest_identity["harness_commit"]:
        raise ValueError("worker source/harness/corpus/frozen provenance disagrees with the campaign")
    execution = payload["execution"]
    if not isinstance(execution, dict):
        raise ValueError("worker must report execution identity")
    validate_gpu(execution.get("gpu"), gpu)
    if execution.get("pid") != process_result.pid or process_result.returncode != 0 or process_result.timed_out:
        raise ValueError("worker process identity or exit status disagrees with the supervisor")
    if execution.get("candidate_commit") != candidate["candidate_commit"] or execution.get("attempt_id") != attempt_id:
        raise ValueError("worker execution commit or attempt disagrees with its context")
    _require_oracle_and_memory(payload, gpu, context["run_id"])
    record = {
        **payload,
        "admission": {
            "status": "PASS", "worker_exited": True, "gpu_released": True,
            "returncode": process_result.returncode, "accepted_at": _utcnow(),
            "manifest_identity_sha256": json_sha256(manifest_identity),
        },
        "execution": {
            **execution,
            "candidate_branch": candidate["candidate_branch"],
            "supervisor_gpu": gpu,
        },
    }
    validate_result(record)
    return record


def _failure_status(message: str, process_result: ProcessResult | None, stage: str) -> str:
    text = message.lower()
    if process_result is not None and process_result.timed_out:
        return "TIMEOUT"
    if "out of memory" in text or "cuda out of memory" in text or "out of memory" in (process_result.tail.lower() if process_result else ""):
        return "OOM"
    if "oracle" in text:
        return "ORACLE"
    if "memory" in text or "peak_gib" in text:
        return "MEMORY"
    if stage == "preflight":
        return "ENVIRONMENT"
    if "json" in text or "result" in text or "protocol" in text:
        return "PROTOCOL"
    if "gpu" in text or "nvidia" in text:
        return "GPU"
    return "PROCESS"


def _write_failure(
    artifact_root: Path,
    run_id: str,
    attempt_id: str,
    *,
    stage: str,
    message: str,
    command: list[str],
    context: dict[str, Any] | None,
    gpu: dict[str, Any] | None,
    log_path: Path,
    process_result: ProcessResult | None = None,
    error_type: str | None = None,
) -> Path:
    destination = artifact_root / run_id / "attempts" / attempt_id
    payload = {
        "schema_version": 1,
        "status": _failure_status(message, process_result, stage),
        "stage": stage,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "error_type": error_type or "RuntimeError",
        "message": message,
        "command": command,
        "pid": process_result.pid if process_result else None,
        "returncode": process_result.returncode if process_result else None,
        "timed_out": process_result.timed_out if process_result else False,
        "tail": process_result.tail if process_result else "",
        "elapsed_seconds": process_result.elapsed_seconds if process_result else None,
        "log_path": str(log_path),
        "context": context,
        "gpu": gpu,
        "created_at": _utcnow(),
    }
    path = destination / "failure.json"
    _atomic_json(path, payload)
    _atomic_json(artifact_root / run_id / "failure.json", payload)
    return path


def _emit_to_log(console: Any, line: str) -> None:
    print(line, end="", flush=True)
    console.write(line)
    console.flush()


@contextmanager
def _release_after_attempt(await_release: Callable[[], None]):
    """Release the physical GPU on every exit, retaining the original failure.

    run_process first terminates the worker tree. This guard then waits for
    physical release while the supervisor still owns the GPU lock. A cleanup
    failure is fatal even after successful output; with two failures the worker
    exception remains primary and the release exception is retained as its cause.
    """
    failure: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            await_release()
        except BaseException as release_error:
            if failure is not None:
                raise failure from release_error
            raise


def _run_attempt(
    *,
    stage: str,
    run_id: str,
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    console: Any,
    attempts_path: Path,
    artifact_root: Path,
    context: dict[str, Any] | None,
    gpu: dict[str, Any] | None,
    timeout_seconds: float,
    inherit_fds: tuple[int, ...],
    await_release: Callable[[], None],
    attempt_id: str | None = None,
) -> tuple[str, ProcessResult, dict[str, Any] | None]:
    attempt_id = attempt_id or uuid.uuid4().hex
    begin = _utcnow()
    _append_event(attempts_path, {
        "event": "begin",
        "attempt_id": attempt_id,
        "stage": stage,
        "run_id": run_id,
        "command": command,
        "started_at": begin,
        "pid": None,
    })
    process_result: ProcessResult | None = None
    try:
        with _release_after_attempt(await_release):
            process_result = run_process(
                command,
                cwd=cwd,
                env=env,
                log_path=log_path,
                timeout_seconds=timeout_seconds,
                emit=lambda line: _emit_to_log(console, line),
                started=lambda pid: _append_event(attempts_path, {
                    "event": "started",
                    "attempt_id": attempt_id,
                    "stage": stage,
                    "run_id": run_id,
                    "pid": pid,
                    "started_at": begin,
                }),
                inherit_fds=inherit_fds,
            )
            if process_result.timed_out:
                raise TimeoutError(f"{stage} worker timed out after {timeout_seconds:.1f}s")
            if process_result.returncode != 0:
                raise RuntimeError(f"{stage} worker exited with code {process_result.returncode}")
            payload = _json_from_stdout(Path(log_path).read_text(encoding="utf-8", errors="replace"))
        _append_event(attempts_path, {
            "event": "end",
            "attempt_id": attempt_id,
            "stage": stage,
            "run_id": run_id,
            "status": "PROCESS_OK",
            "returncode": process_result.returncode,
            "ended_at": _utcnow(),
        })
        return attempt_id, process_result, payload
    except BaseException as exc:
        # Include the chained release failure, if any, in durable diagnostics.
        message = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        failure_path = _write_failure(
            artifact_root,
            run_id,
            attempt_id,
            stage=stage,
            message=message,
            command=command,
            context=context,
            gpu=gpu,
            log_path=log_path,
            process_result=process_result,
            error_type=type(exc).__name__,
        )
        _append_event(attempts_path, {
            "event": "end",
            "attempt_id": attempt_id,
            "stage": stage,
            "run_id": run_id,
            "status": "FAILED",
            "failure_path": str(failure_path),
            "returncode": process_result.returncode if process_result else None,
            "ended_at": _utcnow(),
        })
        raise


def _append_preflight_flag(command: list[str]) -> list[str]:
    if "--preflight" in command:
        return command
    return [*command, "--preflight"]


def _validate_preflight(
    payload: dict[str, Any],
    gpu: dict[str, Any],
    corpus: dict[str, Any],
    candidate_commit: str,
) -> dict[str, Any]:
    if payload.get("status") != "PREFLIGHT_PASS":
        raise ValueError("candidate preflight did not return PREFLIGHT_PASS")
    if payload.get("plan_id") != PLAN_ID:
        raise ValueError("candidate preflight belongs to a different plan")
    if payload.get("source_commit") != candidate_commit:
        raise ValueError("candidate preflight source commit disagrees with the selected worktree")
    if payload.get("sm") != "sm_120":
        raise ValueError(f"candidate preflight reported the wrong SM: {payload.get('sm')!r}")
    if payload.get("device") != EXPECTED_GPU_NAME:
        raise ValueError(f"candidate preflight reported the wrong GPU: {payload.get('device')!r}")
    validate_gpu(payload, gpu)
    if payload.get("torch") is None or payload.get("cuda") is None:
        raise ValueError("candidate preflight did not report the torch/CUDA runtime")
    limits = payload.get("benchmark_limits")
    if not isinstance(limits, dict) or limits.get("microbatch") != 32 or limits.get("warmups") != WARMUPS or limits.get("timed_repetitions") != REPETITIONS:
        raise ValueError("candidate preflight benchmark limits disagree with the frozen protocol")
    validate_corpus(corpus)
    if payload.get("corpus") != corpus:
        raise ValueError("candidate preflight corpus fingerprint disagrees with the campaign manifest")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("frozen") != frozen_identity(build_plan()):
        raise ValueError("candidate preflight frozen champion/contract/config/plan identity drifted")
    return payload


def _default_campaign_paths(ledger: Path, campaign_dir: Path | None) -> dict[str, Path]:
    directory = (campaign_dir or ledger.parent).resolve()
    return {
        "directory": directory,
        "manifest": directory / "generation0_manifest.json",
        "state": directory / "generation0_state.json",
        "attempts": directory / "generation0_attempts.jsonl",
        "console": directory / "generation0_console.log",
        "artifact_root": directory / "generation0_artifacts",
        "campaign_lock": directory / "generation0_campaign.lock",
    }


def assert_campaign_paths_writable(paths: Iterable[Path]) -> None:
    """The observed F05 campaign is retained as evidence, never resumed by C15."""
    for path in paths:
        if "iclr-g0-0ce16e9922b2" in [part.lower() for part in path.resolve().parts]:
            raise ValueError("the original iclr-g0-0ce16e9922b2 evidence campaign is immutable; select a new C15 campaign")


def _fake_or_real_gpu(
    gpu_info: dict[str, Any] | None,
    gpu_uuid: str | None,
    gpu_index: int | None,
) -> tuple[dict[str, Any], Callable[[], None], Callable[[], None], Callable[[], None], Path]:
    """Return GPU operations; ``gpu_info`` is test-only dependency injection."""
    if gpu_info is not None:
        info = dict(gpu_info)
        info.setdefault("uuid", gpu_uuid or "test-gpu")
        info.setdefault("device", EXPECTED_GPU_NAME)
        info.setdefault("name", info["device"])
        info.setdefault("vram_GiB", 94.0)
        info.setdefault("driver_version", "test")
        return (
            info,
            lambda: None,
            lambda: None,
            lambda: None,
            gpu_lock_path(str(info["uuid"])),
        )
    selected = select_target_gpu(gpu_uuid=gpu_uuid, gpu_index=gpu_index)
    info = selected.as_dict()
    return (
        info,
        lambda: assert_idle_gpu(selected),
        lambda: None,
        lambda: wait_gpu_released(selected, baseline_free_mib=selected.free_mib),
        gpu_lock_path(selected.uuid),
    )


def execute_plan(
    command_template: str,
    candidate_root: Path,
    ledger: Path,
    *,
    plan: dict[str, Any] | None = None,
    limit: int | None = None,
    campaign_dir: Path | None = None,
    console_log: Path | None = None,
    artifact_root: Path | None = None,
    corpus_root: Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    retry_failed: bool = False,
    preflight_only: bool = False,
    gpu_uuid: str | None = None,
    gpu_index: int | None = None,
    preflight_required: bool = True,
    # These two hooks make process/ledger behavior unit-testable without
    # pretending that a CPU is evidence for a target-GPU measurement.
    gpu_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the next ordered prefix of slots and stop on every violation."""
    if preflight_only and retry_failed:
        raise ValueError("preflight_only cannot be combined with retry_failed")
    if preflight_only and not preflight_required:
        raise ValueError("preflight_only requires candidate preflight")
    selected_plan = plan or build_plan()
    assert_control_treatment(selected_plan)
    if selected_plan.get("execution_allowed") is not True or selected_plan.get("execution_requires_explicit_flag") is not True:
        raise ValueError("Generation-0 execution is not explicitly enabled in the frozen plan")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
        raise ValueError("timeout_seconds must be finite and positive")
    if not preflight_required and gpu_info is None:
        raise ValueError("real execution requires candidate preflight")

    ledger = ledger.resolve()
    paths = _default_campaign_paths(ledger, campaign_dir)
    if console_log is not None:
        paths["console"] = console_log.resolve()
    if artifact_root is not None:
        paths["artifact_root"] = artifact_root.resolve()
    assert_campaign_paths_writable([ledger, *paths.values()])
    candidate = assert_candidate_worktree(candidate_root)
    corpus = _corpus_fingerprint(corpus_root)
    validate_corpus(corpus)
    provenance = {
        "candidate": code_identity(Path(candidate["candidate_root"])),
        "harness": code_identity(REPO_ROOT),
        "frozen": frozen_identity(selected_plan),
        "corpus": corpus,
    }
    if provenance["candidate"]["commit"] != candidate["candidate_commit"]:
        raise ValueError("candidate changed while collecting source provenance")
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["artifact_root"].mkdir(parents=True, exist_ok=True)
    gpu, assert_idle, _unused, await_release, lock_path = _fake_or_real_gpu(gpu_info, gpu_uuid, gpu_index)
    # The campaign lock is acquired before the physical-GPU lock so two
    # invocations of the same campaign cannot deadlock while competing for it.
    with interrupt_on_signal(), FileLock(paths["campaign_lock"], {"kind": "generation0-campaign"}), FileLock(
        lock_path, {"kind": "generation0-gpu", "gpu_uuid": gpu["uuid"]}
    ) as gpu_lock:
        # Read mutable campaign state only after taking the campaign lock. A
        # second notebook invocation may have been waiting here while the
        # first one appended a result; using pre-lock snapshots would rerun
        # that slot and turn a safe resume into an avoidable failure.
        records = read_ledger(ledger)
        manifest = _load_json_object(paths["manifest"], "campaign manifest")
        state = _load_json_object(paths["state"], "campaign state")
        if records and manifest is None:
            raise ValueError("non-empty ledger has no campaign manifest; refusing to mix historical results")
        if state is not None and manifest is None and not (
            state.get("status") == "FAILED" and state.get("failed_run_id") == "PREFLIGHT"
        ):
            raise ValueError("campaign state exists without its manifest")
        static_identity = _manifest_identity(selected_plan, candidate, gpu, command_template, corpus, timeout_seconds, None, ledger, provenance)
        if manifest is not None:
            actual_identity = manifest.get("identity")
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or not isinstance(actual_identity, dict) or {k: v for k, v in actual_identity.items() if k != "runtime"} != static_identity:
                raise ValueError("campaign manifest identity drifted; use a new campaign directory")
            for record in records:
                if record["provenance"] != provenance or record["admission"]["manifest_identity_sha256"] != json_sha256(actual_identity):
                    raise ValueError("ledger provenance disagrees with the campaign manifest")
        _validate_resume(selected_plan, records, candidate, state)
        events = _read_events(paths["attempts"])
        if manifest is None and events and not (
            state is not None and state.get("status") == "FAILED" and state.get("failed_run_id") == "PREFLIGHT"
        ):
            raise ValueError("historical attempt log has no compatible campaign manifest")
        active = _active_attempts(events)
        if active and not retry_failed:
            raise RuntimeError(
                "an earlier attempt has no durable end event; pass --retry-failed to explicitly recover it: "
                f"{sorted(active)}"
            )
        if active and retry_failed:
            for attempt_id, begin in active.items():
                _append_event(paths["attempts"], {
                    "event": "recovered",
                    "attempt_id": attempt_id,
                    "run_id": begin.get("run_id"),
                    "stage": begin.get("stage"),
                    "status": "RECOVERED_INTERRUPTED",
                    "ended_at": _utcnow(),
                })

        if state is not None and state.get("status") == "FAILED" and not retry_failed:
            raise RuntimeError(
                f"campaign is sticky-failed at {state.get('failed_run_id')}; "
                "pass --retry-failed or start a new campaign directory"
            )
        if state is not None and state.get("status") == "FAILED" and state.get("failed_run_id") != "PREFLIGHT":
            next_run_id = selected_plan["run_order"][len(records)]["run_id"] if len(records) < len(selected_plan["run_order"]) else None
            if state.get("failed_run_id") != next_run_id:
                raise ValueError("campaign state names a failed run that is not the next ordered slot")

        assert_idle()
        environment = worker_environment(
            GpuSnapshot(
                uuid=str(gpu["uuid"]),
                name=str(gpu.get("name", gpu.get("device", EXPECTED_GPU_NAME))),
                total_mib=int(float(gpu.get("vram_GiB", 94.0)) * 1024),
                used_mib=int(gpu.get("memory_used_MiB", 0)),
                free_mib=int(gpu.get("memory_free_MiB", int(float(gpu.get("vram_GiB", 94.0)) * 1024))),
                driver=str(gpu.get("driver_version", gpu.get("driver", "unknown"))),
            ),
            corpus_root=corpus_root,
        )
        environment.update({
            "ICRL_G0_PLAN_ID": selected_plan["plan_id"],
            "ICRL_G0_CHAMPION_IMMUTABLE": "1",
            "ICRL_G0_ARTIFACT_ROOT": str(paths["artifact_root"]),
            "ICRL_G0_PROVENANCE_JSON": json.dumps(provenance, sort_keys=True),
        })
        paths["console"].parent.mkdir(parents=True, exist_ok=True)
        with paths["console"].open("a", encoding="utf-8", buffering=1) as console:
            console.write(f"\n=== Generation-0 supervisor start {_utcnow()} ===\n")
            console.flush()
            preflight_payload: dict[str, Any] | None = None
            preflight_command: list[str] = []
            preflight_log = paths["directory"] / "logs" / "preflight-unstarted.log"
            preflight_attempt_tag = "preflight-unstarted"
            try:
                slots = selected_plan["run_order"]
                if limit is not None and limit < len(records):
                    raise ValueError(f"--limit={limit} is smaller than the existing ledger prefix ({len(records)})")
                target_count = min(limit, len(slots)) if limit is not None else len(slots)
                if preflight_required:
                    preflight_context = {
                        "plan_id": selected_plan["plan_id"],
                        "run_id": "PREFLIGHT",
                        "arm_id": "PREFLIGHT",
                        "block_id": 0,
                        "within_block_position": 0,
                        "selected_levels": selected_plan["control_reference"]["factor_levels"],
                    }
                    preflight_attempt_tag = uuid.uuid4().hex
                    command = _append_preflight_flag(_command_argv(_format_command(command_template, preflight_context, candidate, preflight_attempt_tag)))
                    preflight_command = command
                    environment.update({
                        "ICRL_G0_RUN_ID": "PREFLIGHT",
                        "ICRL_G0_ARM_ID": "PREFLIGHT",
                        "ICRL_G0_BLOCK_ID": "0",
                        "ICRL_G0_WITHIN_BLOCK_POSITION": "0",
                        "ICRL_G0_SELECTED_LEVELS_JSON": json.dumps(preflight_context["selected_levels"], sort_keys=True),
                        "ICRL_G0_ATTEMPT_ID": preflight_attempt_tag,
                        "ICRL_G0_COMMAND": json.dumps(command),
                    })
                    preflight_log = paths["directory"] / "logs" / f"preflight-{preflight_attempt_tag}.log"
                    actual_attempt, process_result, payload = _run_attempt(
                        stage="preflight",
                        run_id="PREFLIGHT",
                        command=command,
                        env=environment,
                        cwd=Path(candidate["candidate_root"]),
                        log_path=preflight_log,
                        console=console,
                        attempts_path=paths["attempts"],
                        artifact_root=paths["artifact_root"],
                        context=preflight_context,
                        gpu=gpu,
                        timeout_seconds=timeout_seconds,
                        inherit_fds=(gpu_lock.fileno(),),
                        await_release=await_release,
                        attempt_id=preflight_attempt_tag,
                    )
                    if payload is None:
                        raise ValueError("preflight worker emitted no payload")
                    _validate_preflight(payload, gpu, corpus, candidate["candidate_commit"])
                    if payload.get("attempt_id") != actual_attempt or payload.get("provenance") != provenance or payload.get("harness_commit") != provenance["harness"]["commit"]:
                        raise ValueError("preflight attempt or source provenance disagrees with the supervisor")
                    preflight_payload = payload
                    console.write(f"preflight: PASS {json.dumps(preflight_payload, sort_keys=True)}\n")
                    console.flush()
                    if state is not None:
                        state["failed_run_id"] = None

                expected_identity = _manifest_identity(
                    selected_plan,
                    candidate,
                    gpu,
                    command_template,
                    corpus,
                    timeout_seconds,
                    preflight_payload,
                    ledger,
                    provenance,
                )
                if manifest is not None:
                    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION or manifest.get("identity") != expected_identity:
                        raise ValueError("campaign manifest identity drifted; use a new campaign directory")
                else:
                    manifest = {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "created_at": _utcnow(),
                        "identity": expected_identity,
                        "status": "READY",
                        "champion_immutable": True,
                    }
                    _atomic_json(paths["manifest"], manifest)

                if state is None:
                    state = {
                        "schema_version": STATE_SCHEMA_VERSION,
                        "plan_id": selected_plan["plan_id"],
                        "status": "READY",
                        "completed_run_ids": [record["run_id"] for record in records],
                        "failed_run_id": None,
                        "updated_at": _utcnow(),
                    }
                    _atomic_json(paths["state"], state)

                if preflight_only:
                    state["completed_run_ids"] = [record["run_id"] for record in records]
                    state["failed_run_id"] = None
                    state["status"] = "COMPLETE" if len(records) == len(slots) else "PARTIAL" if records else "READY"
                    state["updated_at"] = _utcnow()
                    _atomic_json(paths["state"], state)
                    summary = {
                        "status": "PREFLIGHT_ONLY",
                        "preflight_only": True,
                        "campaign_status": state["status"],
                        "plan_id": selected_plan["plan_id"],
                        "executed_run_ids": [],
                        "skipped_existing_run_ids": [],
                        "completed_count": len(records),
                        "total_slots": len(slots),
                        "ledger": str(ledger),
                        "manifest": str(paths["manifest"]),
                        "state": str(paths["state"]),
                        "console_log": str(paths["console"]),
                        "champion_immutable": True,
                    }
                    console.write(json.dumps(summary, sort_keys=True) + "\n")
                    console.flush()
                    return summary

                executed: list[str] = []
                skipped: list[str] = []
                for index in range(len(records), target_count):
                    slot = slots[index]
                    context = run_context(selected_plan, slot)
                    attempt_tag = uuid.uuid4().hex
                    command = _command_argv(_format_command(command_template, context, candidate, attempt_tag))
                    environment.update({
                        "ICRL_G0_RUN_ID": context["run_id"],
                        "ICRL_G0_ARM_ID": context["arm_id"],
                        "ICRL_G0_BLOCK_ID": str(context["block_id"]),
                        "ICRL_G0_WITHIN_BLOCK_POSITION": str(context["within_block_position"]),
                        "ICRL_G0_SELECTED_LEVELS_JSON": json.dumps(context["selected_levels"], sort_keys=True),
                        "ICRL_G0_ATTEMPT_ID": attempt_tag,
                        "ICRL_G0_COMMAND": json.dumps(command),
                    })
                    console.write(
                        f"\n=== {context['run_id']} {context['arm_id']} block={context['block_id']} "
                        f"position={context['within_block_position']} ===\ncommand: {command}\n"
                    )
                    console.flush()
                    worker_log = paths["directory"] / "logs" / f"{context['run_id']}-{attempt_tag}.log"
                    process_result: ProcessResult | None = None
                    actual_attempt = attempt_tag
                    try:
                        actual_attempt, process_result, payload = _run_attempt(
                            stage="benchmark",
                            run_id=context["run_id"],
                            command=command,
                            env=environment,
                            cwd=Path(candidate["candidate_root"]),
                            log_path=worker_log,
                            console=console,
                            attempts_path=paths["attempts"],
                            artifact_root=paths["artifact_root"],
                            context=context,
                            gpu=gpu,
                            timeout_seconds=timeout_seconds,
                            inherit_fds=(gpu_lock.fileno(),),
                            await_release=await_release,
                            attempt_id=attempt_tag,
                        )
                        if payload is None:
                            raise ValueError("benchmark worker emitted no payload")
                        if code_identity(REPO_ROOT) != provenance["harness"] or code_identity(Path(candidate["candidate_root"])) != provenance["candidate"]:
                            raise ValueError("source changed during the worker attempt")
                        record = _validate_worker_payload(payload, context, candidate, gpu, actual_attempt, expected_identity, process_result)
                        # Ledger admission happens only after the process has
                        # exited and the physical GPU has returned to baseline.
                        append_result(ledger, record)
                        records.append(record)
                        executed.append(context["run_id"])
                        state["completed_run_ids"] = [item["run_id"] for item in records]
                        state["failed_run_id"] = None
                        state["status"] = "COMPLETE" if len(records) == len(slots) else "PARTIAL"
                        state["updated_at"] = _utcnow()
                        _atomic_json(paths["state"], state)
                        console.write(f"accepted append-only result: {context['run_id']}\n")
                        console.flush()
                    except BaseException as exc:
                        failure_path = paths["artifact_root"] / context["run_id"] / "failure.json"
                        if not failure_path.exists():
                            _write_failure(
                                paths["artifact_root"],
                                context["run_id"],
                                actual_attempt,
                                stage="benchmark",
                                message=f"{type(exc).__name__}: {exc}",
                                command=command,
                                context=context,
                                gpu=gpu,
                                log_path=worker_log,
                                process_result=process_result,
                                error_type=type(exc).__name__,
                            )
                        state["completed_run_ids"] = [item["run_id"] for item in records]
                        state["failed_run_id"] = context["run_id"]
                        state["status"] = "FAILED"
                        state["updated_at"] = _utcnow()
                        _atomic_json(paths["state"], state)
                        console.write(f"FAIL_CLOSED: {type(exc).__name__}: {exc}\n")
                        console.flush()
                        raise

                state["status"] = "COMPLETE" if len(records) == len(slots) else "PARTIAL"
                state["updated_at"] = _utcnow()
                _atomic_json(paths["state"], state)
                summary = {
                    "status": state["status"],
                    "plan_id": selected_plan["plan_id"],
                    "executed_run_ids": executed,
                    "skipped_existing_run_ids": skipped,
                    "completed_count": len(records),
                    "total_slots": len(slots),
                    "ledger": str(ledger),
                    "manifest": str(paths["manifest"]),
                    "state": str(paths["state"]),
                    "console_log": str(paths["console"]),
                    "champion_immutable": True,
                }
                console.write(json.dumps(summary, sort_keys=True) + "\n")
                console.flush()
                return summary
            except BaseException as exc:
                if preflight_required and preflight_payload is None:
                    failure_path = paths["artifact_root"] / "PREFLIGHT" / "failure.json"
                    if not failure_path.exists():
                        _write_failure(
                            paths["artifact_root"],
                            "PREFLIGHT",
                            preflight_attempt_tag,
                            stage="preflight",
                            message=f"{type(exc).__name__}: {exc}",
                            command=preflight_command,
                            context={"plan_id": selected_plan["plan_id"], "run_id": "PREFLIGHT"},
                            gpu=gpu,
                            log_path=preflight_log,
                            error_type=type(exc).__name__,
                        )
                    if state is None:
                        state = {
                            "schema_version": STATE_SCHEMA_VERSION,
                            "plan_id": selected_plan["plan_id"],
                            "status": "FAILED",
                            "completed_run_ids": [item["run_id"] for item in records],
                            "failed_run_id": "PREFLIGHT",
                            "updated_at": _utcnow(),
                        }
                    else:
                        state["status"] = "FAILED"
                        state["failed_run_id"] = "PREFLIGHT"
                        state["updated_at"] = _utcnow()
                    _atomic_json(paths["state"], state)
                console.write(traceback.format_exc())
                console.flush()
                raise
