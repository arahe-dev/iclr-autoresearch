from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def champion_branch(experiment_id: str, name: str) -> str:
    return f"codex/champion/{experiment_id}-{slug(name)}"


def candidate_branch(experiment_id: str, name: str) -> str:
    return f"codex/experiment/{experiment_id}-{slug(name)}"


def branch_plan(experiment_id: str, name: str, decision: str) -> dict[str, Any]:
    keep = decision.startswith("KEEP") or "__KEEP" in decision
    return {
        "experiment_id": experiment_id,
        "decision": decision,
        "action": "CREATE_CHAMPION_BRANCH" if keep else "REVERT_AND_DO_NOT_PROMOTE",
        "champion_branch": champion_branch(experiment_id, name) if keep else None,
        "candidate_branch": candidate_branch(experiment_id, name),
        "status": "PLANNED_NO_EXECUTION",
    }


def apply_champion_branch(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("action") != "CREATE_CHAMPION_BRANCH":
        return {**plan, "status": "REVERTED_NO_BRANCH_CREATED"}
    branch = plan["champion_branch"]
    subprocess.run(["git", "branch", branch], cwd=ROOT, check=True)
    return {**plan, "status": "CHAMPION_BRANCH_CREATED"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan Git-tracked champion promotion")
    parser.add_argument("experiment_id")
    parser.add_argument("name")
    parser.add_argument("decision")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    plan = branch_plan(args.experiment_id, args.name, args.decision)
    if args.apply:
        plan = apply_champion_branch(plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

