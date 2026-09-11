"""Target-host entrypoint for the repaired Generation-0 supervisor.

This file remains as the stable SM120-facing name used by the Colab cell and
older probes.  All execution now goes through ``gpu_harness.execute_plan`` so
there is one ordering, provenance, lock, timeout, and fail-closed path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.generation0 import PLAN_ID, build_plan
from icrl_autoresearch.gpu import EXPECTED_GPU_NAME, select_target_gpu
from icrl_autoresearch.gpu_harness import (
    _command_argv,
    _format_command,
    _json_from_stdout,
    execute_plan,
)


def exact_target_gpu() -> dict[str, Any]:
    """Retained compatibility probe; it is deliberately torch-free."""
    return select_target_gpu().as_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Run C15 Generation-0 through the strict {EXPECTED_GPU_NAME}/SM120 supervisor")
    parser.add_argument("--execute", action="store_true", help="required to launch any candidate command")
    parser.add_argument("--candidate-root", type=Path, help="separate codex/experiment/* or codex/generation0/* worktree")
    parser.add_argument("--command-template", help="argv-only runner command; supports the documented placeholders")
    parser.add_argument("--ledger", type=Path, default=Path("results/generation0.jsonl"))
    parser.add_argument("--campaign-dir", type=Path, help="stable directory containing manifest/state/attempts/logs")
    parser.add_argument("--console-log", type=Path, help="append-only supervisor console transcript")
    parser.add_argument("--artifact-root", type=Path, help="durable per-attempt result/failure artifacts")
    parser.add_argument("--corpus-root", type=Path, help="frozen corpus root passed to the candidate worker")
    parser.add_argument("--gpu-uuid", help="physical target GPU UUID")
    parser.add_argument("--gpu-index", type=int, help="physical nvidia-smi GPU index")
    parser.add_argument("--timeout-seconds", type=float, default=60 * 60, help="per-worker wall-clock timeout")
    retry_mode = parser.add_mutually_exclusive_group()
    retry_mode.add_argument("--retry-failed", action="store_true", help="explicitly retry the sticky failed next slot")
    retry_mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="run and validate candidate preflight, persist campaign setup, then stop before benchmarks",
    )
    parser.add_argument("--limit", type=int, help="optional prefix of the frozen 50-slot C15 order (original slot IDs retained)")
    args = parser.parse_args()

    plan = build_plan()
    if not args.execute:
        print(json.dumps({
            "status": "READY_NO_EXECUTION",
            "plan_id": PLAN_ID,
            "arms": len(plan["arms"]),
            "run_slots": len(plan["run_order"]),
            "requires": f"nvidia-smi {EXPECTED_GPU_NAME} sm_120, >=90 GiB, candidate preflight, --execute",
        }, indent=2, sort_keys=True))
        return 0

    if not args.candidate_root or not args.command_template:
        raise SystemExit("--execute requires both --candidate-root and --command-template")
    result = execute_plan(
        args.command_template,
        args.candidate_root,
        args.ledger,
        plan=plan,
        limit=args.limit,
        campaign_dir=args.campaign_dir,
        console_log=args.console_log,
        artifact_root=args.artifact_root,
        corpus_root=args.corpus_root,
        gpu_uuid=args.gpu_uuid,
        gpu_index=args.gpu_index,
        timeout_seconds=args.timeout_seconds,
        retry_failed=args.retry_failed,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
