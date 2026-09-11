from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.generation0 import build_plan
from icrl_autoresearch.gpu_harness import execute_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Generation-0 in an explicitly supplied SM120 candidate worktree"
    )
    parser.add_argument(
        "--command-template",
        help=(
            "candidate command; placeholders: {plan_id}, {run_id}, {arm_id}, "
            "{block_id}, {within_block_position}, {candidate_root}, {selected_levels_json}, {attempt_id}"
        ),
    )
    parser.add_argument("--candidate-root", type=Path, help="separate codex/experiment or codex/generation0 worktree")
    parser.add_argument("--ledger", type=Path, default=Path("results/generation0.jsonl"))
    parser.add_argument("--campaign-dir", type=Path, help="stable directory containing manifest, state, attempts, and logs")
    parser.add_argument("--console-log", type=Path, help="append-only supervisor console transcript")
    parser.add_argument("--artifact-root", type=Path, help="durable per-attempt result/failure artifacts")
    parser.add_argument("--corpus-root", type=Path, help="frozen corpus root passed to the candidate worker")
    parser.add_argument("--gpu-uuid", help="physical target GPU UUID; required when target selection is ambiguous")
    parser.add_argument("--gpu-index", type=int, help="physical nvidia-smi GPU index")
    parser.add_argument("--timeout-seconds", type=float, default=60 * 60, help="per-worker wall-clock timeout")
    retry_mode = parser.add_mutually_exclusive_group()
    retry_mode.add_argument("--retry-failed", action="store_true", help="explicitly retry the sticky failed next slot")
    retry_mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="run and validate candidate preflight, persist campaign setup, then stop before benchmarks",
    )
    parser.add_argument("--limit", type=int, help="optional number of run-order slots to execute")
    parser.add_argument("--execute", action="store_true", help="perform GPU execution; omitted means preview only")
    args = parser.parse_args()

    plan = build_plan()
    if not args.execute:
        print(json.dumps({
            "status": "READY_NO_EXECUTION",
            "plan_id": plan["plan_id"],
            "arms": len(plan["arms"]),
            "run_slots": len(plan["run_order"]),
            "execution_requires_explicit_flag": plan["execution_requires_explicit_flag"],
            "candidate_worktree_required": True,
            "champion_immutable": True,
        }, indent=2, sort_keys=True))
        return

    if not args.command_template:
        raise SystemExit("--execute requires --command-template")
    if not args.candidate_root:
        raise SystemExit("--execute requires --candidate-root")
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


if __name__ == "__main__":
    main()
