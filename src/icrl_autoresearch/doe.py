from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "generation_0_screening.json"


def load_generation0(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def screening_plan(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_generation0()
    if cfg.get("science_delta_allowed") is not False:
        raise ValueError("Generation-0 plan must remain science-frozen")
    if cfg.get("execution_allowed") is not True or cfg.get("execution_requires_explicit_flag") is not True:
        raise ValueError("Generation-0 execution must be explicitly enabled and explicitly gated")
    rows = []
    for row in cfg.get("screening_rows", []):
        item = dict(row)
        item["status"] = "READY_FOR_EXPLICIT_EXECUTION"
        item["science_delta"] = False
        item["requires_exact_oracle"] = True
        rows.append(item)
    return {
        "schema_version": 1,
        "generation": 0,
        "status": "READY_FOR_EXPLICIT_EXECUTION",
        "baseline_experiment": cfg["baseline_experiment"],
        "design": cfg["design"],
        "science_delta_allowed": False,
        "execution_allowed": True,
        "execution_requires_explicit_flag": True,
        "factors": cfg["factors"],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the compatibility Generation-0 screening summary")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true", help="rejected here; use the GPU harness for explicit candidate execution")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("Generation-0 execution is disabled: this scaffold does not run new model changes.")
    plan = screening_plan(json.loads(args.config.read_text(encoding="utf-8")))
    payload = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
