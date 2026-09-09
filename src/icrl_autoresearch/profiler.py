from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProfilerPlan:
    name: str
    command: str
    output_dir: str
    activities: tuple[str, ...] = ("CPU", "CUDA")
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    status: str = "PLANNED_NO_EXECUTION"


def make_plan(name: str, command: str, output_dir: Path) -> dict[str, Any]:
    return asdict(ProfilerPlan(name=name, command=command, output_dir=str(output_dir)))


def run_external_plan(plan: dict[str, Any], *, execute: bool = False) -> dict[str, Any]:
    """Run only an explicitly supplied external command; default is plan-only."""
    if not execute:
        return {**plan, "status": "PLANNED_NO_EXECUTION"}
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(plan["command"], shell=True, text=True, capture_output=True, check=False)
    result = {**plan, "status": "COMPLETED" if completed.returncode == 0 else "FAILED", "returncode": completed.returncode}
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (output_dir / "profile_plan.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a profiler plan; execute only with explicit --execute")
    parser.add_argument("--name", default="unnamed")
    parser.add_argument("--command", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = make_plan(args.name, args.command, args.output_dir)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(json.dumps(run_external_plan(plan, execute=True), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

