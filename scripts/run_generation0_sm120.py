"""Strict launcher for the Generation-0 campaign on the target SM120 host.

This wrapper does not implement model factors.  The supplied runner command
must apply the selected exact-semantics treatment in the candidate worktree and
emit one JSON result object as its final stdout line.  The wrapper refuses to
run on the wrong GPU, refuses the champion workspace/branch, tees a complete
console transcript, appends only oracle- and memory-gated records, and stops at
the first failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import traceback
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.generation0 import build_plan
from icrl_autoresearch.gpu_harness import (
    _format_command,
    _json_from_stdout,
    assert_candidate_worktree,
    run_context,
)
from icrl_autoresearch.results import append_result, existing_run_ids


EXPECTED_GPU_NAME = "RTX PRO 6000"
EXPECTED_SM = (12, 0)
MIN_VRAM_GIB = 90.0
PLAN_ID = "G0-L8-SM120-EXACT-B32"


def exact_target_gpu() -> dict[str, Any]:
    """Require the declared RTX PRO 6000 Blackwell / SM120 environment."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - target-host dependent
        raise RuntimeError("target execution requires PyTorch with CUDA support") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("target execution requires torch.cuda.is_available()")
    device = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    properties = torch.cuda.get_device_properties(0)
    vram_gib = float(properties.total_memory / 2**30)
    if EXPECTED_GPU_NAME not in device:
        raise RuntimeError(f"target execution requires a {EXPECTED_GPU_NAME} GPU; found {device}")
    if capability != EXPECTED_SM:
        raise RuntimeError(f"target execution requires sm_120; found sm_{capability[0]}{capability[1]}")
    if vram_gib < MIN_VRAM_GIB:
        raise RuntimeError(f"target execution requires at least {MIN_VRAM_GIB:.0f} GiB VRAM; found {vram_gib:.2f} GiB")
    return {
        "device": device,
        "sm": f"sm_{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vram_GiB": vram_gib,
    }


def _write(log: Any, message: str = "") -> None:
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def _require_oracle_and_memory(payload: dict[str, Any], gpu: dict[str, Any], run_id: str) -> None:
    oracle = payload.get("oracle")
    if not isinstance(oracle, dict) or oracle.get("status") != "PASS":
        raise ValueError(f"{run_id}: oracle.status must be PASS before timing is accepted")
    if oracle.get("same_document_reset") != "PASS":
        raise ValueError(f"{run_id}: same-document reset gate did not pass")
    if oracle.get("tied_qk_backward") != "PASS":
        raise ValueError(f"{run_id}: tied-QK backward gate did not pass")

    memory = payload.get("memory")
    if isinstance(memory, dict) and memory.get("status") not in (None, "PASS"):
        raise ValueError(f"{run_id}: memory gate did not pass: {memory.get('status')!r}")
    if payload.get("memory_ok") is False or payload.get("memory_status") not in (None, "PASS"):
        raise ValueError(f"{run_id}: runner reported a memory violation")
    peak = payload.get("peak_GiB")
    if not isinstance(peak, (int, float)) or not math.isfinite(float(peak)) or float(peak) <= 0:
        raise ValueError(f"{run_id}: result must include a positive finite peak_GiB")
    if float(peak) >= float(gpu["vram_GiB"]):
        raise ValueError(f"{run_id}: peak_GiB={peak} exceeds target device capacity")


def _run_slot(
    command_template: str,
    context: dict[str, Any],
    candidate: dict[str, str],
    gpu: dict[str, Any],
    ledger: Path,
    log: Any,
) -> str:
    command = _format_command(command_template, context, candidate)
    env = os.environ.copy()
    env.update(
        {
            "ICRL_G0_PLAN_ID": context["plan_id"],
            "ICRL_G0_RUN_ID": context["run_id"],
            "ICRL_G0_ARM_ID": context["arm_id"],
            "ICRL_G0_BLOCK_ID": str(context["block_id"]),
            "ICRL_G0_WITHIN_BLOCK_POSITION": str(context["within_block_position"]),
            "ICRL_G0_SELECTED_LEVELS_JSON": json.dumps(context["selected_levels"], sort_keys=True),
            "ICRL_G0_CHAMPION_IMMUTABLE": "1",
        }
    )
    _write(log, f"\n=== {context['run_id']} {context['arm_id']} block={context['block_id']} position={context['within_block_position']} ===")
    _write(log, f"command: {command}")
    process = subprocess.Popen(
        command,
        cwd=candidate["candidate_root"],
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{context['run_id']}: runner exited with code {return_code}")

    payload = _json_from_stdout("".join(output))
    for key in ("run_id", "arm_id", "block_id", "within_block_position"):
        if key in payload and payload[key] != context[key]:
            raise ValueError(f"{context['run_id']}: runner {key} disagrees with run order")
    if payload.get("source_commit") not in (None, candidate["candidate_commit"]):
        raise ValueError(f"{context['run_id']}: runner source_commit is not the candidate commit")
    if payload.get("selected_levels") not in (None, context["selected_levels"]):
        raise ValueError(f"{context['run_id']}: runner selected_levels disagree with run order")

    _require_oracle_and_memory(payload, gpu, context["run_id"])
    record = {
        **payload,
        "schema_version": 1,
        "plan_id": PLAN_ID,
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
    _write(log, f"accepted append-only result: {context['run_id']}")
    return context["run_id"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Generation-0 only on the exact RTX PRO 6000 SM120 host")
    parser.add_argument("--execute", action="store_true", help="required to launch any candidate command")
    parser.add_argument("--candidate-root", type=Path, help="separate codex/experiment/* or codex/generation0/* worktree")
    parser.add_argument("--command-template", help="runner command; supports the placeholders used by run_generation0.py")
    parser.add_argument("--ledger", type=Path, default=Path("results/generation0.jsonl"))
    parser.add_argument("--console-log", type=Path, default=Path("results/generation0_console.log"))
    parser.add_argument("--limit", type=int, help="optional prefix of the 52-slot order for a deliberate smoke run")
    args = parser.parse_args()

    plan = build_plan()
    if not args.execute:
        print(json.dumps({
            "status": "READY_NO_EXECUTION",
            "plan_id": PLAN_ID,
            "arms": len(plan["arms"]),
            "run_slots": len(plan["run_order"]),
            "requires": "RTX PRO 6000 sm_120, PyTorch CUDA, candidate runner, --execute",
        }, indent=2, sort_keys=True))
        return 0
    if not args.candidate_root or not args.command_template:
        raise SystemExit("--execute requires both --candidate-root and --command-template")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    args.ledger = args.ledger.resolve()
    args.console_log = args.console_log.resolve()
    args.console_log.parent.mkdir(parents=True, exist_ok=True)
    with args.console_log.open("a", encoding="utf-8", buffering=1) as log:
        _write(log, "\n" + "=" * 80)
        _write(log, f"Generation-0 launcher start: {datetime.now(timezone.utc).isoformat()}")
        _write(log, f"plan={PLAN_ID} execute=True ledger={args.ledger}")
        try:
            candidate = assert_candidate_worktree(args.candidate_root)
            gpu = exact_target_gpu()
            _write(log, f"candidate={json.dumps(candidate, sort_keys=True)}")
            _write(log, f"gpu={json.dumps(gpu, sort_keys=True)}")
            existing = existing_run_ids(args.ledger)
            slots = plan["run_order"][: args.limit] if args.limit is not None else plan["run_order"]
            executed: list[str] = []
            skipped: list[str] = []
            for slot in slots:
                context = run_context(plan, slot)
                if context["run_id"] in existing:
                    skipped.append(context["run_id"])
                    _write(log, f"skip existing result: {context['run_id']}")
                    continue
                executed.append(_run_slot(args.command_template, context, candidate, gpu, args.ledger, log))
            summary = {
                "status": "EXECUTED",
                "plan_id": PLAN_ID,
                "executed_run_ids": executed,
                "skipped_existing_run_ids": skipped,
                "ledger": str(args.ledger),
                "console_log": str(args.console_log),
                "champion_immutable": True,
            }
            _write(log, json.dumps(summary, sort_keys=True))
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            _write(log, f"FAIL_CLOSED: {type(exc).__name__}: {exc}")
            trace = traceback.format_exc()
            print(trace, end="", flush=True)
            log.write(trace)
            log.flush()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
