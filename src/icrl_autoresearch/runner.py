from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"


def next_experiment_id(root: Path = EXPERIMENTS) -> str:
    ids = []
    for path in root.iterdir():
        if path.is_dir() and re.match(r"^\d{4}_", path.name):
            ids.append(int(path.name[:4]))
    return f"{(max(ids) + 1) if ids else 0:04d}"


def plan_experiment(name: str, parent: str = "0000", root: Path = EXPERIMENTS) -> dict[str, Any]:
    experiment_id = next_experiment_id(root)
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "name": name,
        "parent": parent,
        "status": "PLANNED_NO_EXECUTION",
        "science_delta": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(root / f"{experiment_id}_{name}"),
        "execution": "disabled by default; create records only after explicit authorization",
    }


def create_experiment_folder(plan: dict[str, Any], root: Path = EXPERIMENTS) -> Path:
    folder = root / f"{plan['experiment_id']}_{plan['name']}"
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {k: v for k, v in plan.items() if k != "path"}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or create a tracked experiment folder")
    parser.add_argument("name")
    parser.add_argument("--parent", default="0000")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    plan = plan_experiment(args.name, args.parent)
    if args.create:
        path = create_experiment_folder(plan)
        plan["path"] = str(path)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

