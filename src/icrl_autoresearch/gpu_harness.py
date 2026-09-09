"""Explicit GPU execution adapter for the Generation-0 treatment plan.

The harness never edits source or creates/promotes champion branches. It runs a
user-supplied command in a separate candidate worktree, requires the target
SM120 GPU, requires an oracle-passing JSON result, and appends only validated
records to the Generation-0 ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .contract import REPO_ROOT, source_sha256
from .generation0 import (
    BASE_COMMIT,
    CANONICAL_SOURCE,
    CANONICAL_SOURCE_SHA256,
    CHAMPION_BRANCH,
    assert_control_treatment,
    build_plan,
)
from .results import append_result, existing_run_ids


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
    """Verify that execution is isolated from the immutable champion checkout."""
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

    source = candidate / CANONICAL_SOURCE
    if not source.exists() or source_sha256(source) != CANONICAL_SOURCE_SHA256:
        raise ValueError("candidate canonical source is not byte-equivalent to experiment 0000")
    return {
        "candidate_root": str(candidate),
        "candidate_branch": branch,
        "candidate_commit": _git(candidate, "rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "canonical_source_sha256": CANONICAL_SOURCE_SHA256,
    }


def assert_target_gpu() -> dict[str, Any]:
    """Require an available SM120 GPU only on the explicit execution path."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on execution host
        raise RuntimeError("GPU execution requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires torch.cuda.is_available()")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability != (12, 0):
        raise RuntimeError(f"GPU execution requires sm_120; found sm_{capability[0]}{capability[1]}")
    return {
        "device": torch.cuda.get_device_name(0),
        "sm": f"sm_{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


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
        "kind": slot["kind"],
        "block_id": slot["block_id"],
        "within_block_position": slot["within_block_position"],
        "selected_levels": selected_levels,
        "oracle_required": True,
        "champion_immutable": True,
    }


def _format_command(template: str, context: dict[str, Any], candidate: dict[str, str]) -> str:
    values = {
        "plan_id": context["plan_id"],
        "run_id": context["run_id"],
        "arm_id": context["arm_id"],
        "block_id": context["block_id"],
        "within_block_position": context["within_block_position"],
        "candidate_root": candidate["candidate_root"],
        "selected_levels_json": json.dumps(context["selected_levels"], sort_keys=True),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"unknown command-template placeholder: {exc.args[0]}") from exc


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("GPU command emitted no JSON result")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
        for line in reversed(text.splitlines()):
            try:
                value = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if value is None:
            raise ValueError("GPU command stdout did not contain a JSON result")
    if not isinstance(value, dict):
        raise ValueError("GPU command result must be a JSON object")
    return value


def _run_slot(
    command_template: str,
    context: dict[str, Any],
    candidate: dict[str, str],
    gpu: dict[str, Any],
    ledger: Path,
) -> dict[str, Any]:
    command = _format_command(command_template, context, candidate)
    env = os.environ.copy()
    env.update({
        "ICRL_G0_PLAN_ID": context["plan_id"],
        "ICRL_G0_RUN_ID": context["run_id"],
        "ICRL_G0_ARM_ID": context["arm_id"],
        "ICRL_G0_BLOCK_ID": str(context["block_id"]),
        "ICRL_G0_WITHIN_BLOCK_POSITION": str(context["within_block_position"]),
        "ICRL_G0_SELECTED_LEVELS_JSON": json.dumps(context["selected_levels"], sort_keys=True),
        "ICRL_G0_CHAMPION_IMMUTABLE": "1",
    })
    completed = subprocess.run(
        command,
        cwd=candidate["candidate_root"],
        env=env,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"GPU command failed for {context['run_id']} ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    payload = _json_from_stdout(completed.stdout)
    for key in ("run_id", "arm_id", "block_id"):
        if key in payload and payload[key] != context[key]:
            raise ValueError(f"GPU result {key} disagrees with run order for {context['run_id']}")
    if payload.get("source_commit") not in (None, candidate["candidate_commit"]):
        raise ValueError(f"GPU result source_commit is not the candidate commit for {context['run_id']}")

    record = {
        **payload,
        "schema_version": 1,
        "plan_id": context["plan_id"],
        "source_commit": candidate["candidate_commit"],
        "run_id": context["run_id"],
        "arm_id": context["arm_id"],
        "block_id": context["block_id"],
        "within_block_position": context["within_block_position"],
        "selected_levels": context["selected_levels"],
        "execution": {
            "candidate_branch": candidate["candidate_branch"],
            "candidate_commit": candidate["candidate_commit"],
            "gpu": gpu,
            "champion_immutable": True,
        },
    }
    append_result(ledger, record)
    return record


def execute_plan(
    command_template: str,
    candidate_root: Path,
    ledger: Path,
    *,
    plan: dict[str, Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Execute missing slots only after all candidate/GPU/control gates pass."""
    selected_plan = plan or build_plan()
    assert_control_treatment(selected_plan)
    if selected_plan.get("execution_allowed") is not True or selected_plan.get("execution_requires_explicit_flag") is not True:
        raise ValueError("Generation-0 execution is not explicitly enabled in the frozen plan")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")

    candidate = assert_candidate_worktree(candidate_root)
    gpu = assert_target_gpu()
    existing = existing_run_ids(ledger)
    slots = selected_plan["run_order"][:limit] if limit is not None else selected_plan["run_order"]
    executed = []
    skipped = []
    for slot in slots:
        context = run_context(selected_plan, slot)
        if context["run_id"] in existing:
            skipped.append(context["run_id"])
            continue
        executed.append(_run_slot(command_template, context, candidate, gpu, ledger)["run_id"])
    return {
        "status": "EXECUTED",
        "plan_id": selected_plan["plan_id"],
        "candidate": candidate,
        "gpu": gpu,
        "executed_run_ids": executed,
        "skipped_existing_run_ids": skipped,
        "ledger": str(ledger),
        "champion_immutable": True,
    }
