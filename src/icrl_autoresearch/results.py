from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PLAN_ID = "G0-L8-SM120-EXACT-B32"


def validate_result(record: dict[str, Any]) -> None:
    required = ("schema_version", "plan_id", "source_commit", "run_id", "arm_id", "block_id", "oracle", "latency_ms", "real_input_tok_s", "decision")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"missing result fields: {missing}")
    if record["schema_version"] != 1 or record["plan_id"] != PLAN_ID:
        raise ValueError("result does not belong to Generation-0 plan")
    if record["oracle"].get("status") != "PASS":
        raise ValueError("oracle correctness must pass before a result can be appended")
    if float(record["latency_ms"]) <= 0 or float(record["real_input_tok_s"]) <= 0:
        raise ValueError("latency and real-token throughput must be positive")


def existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["run_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_result(path: Path, record: dict[str, Any]) -> None:
    """Append one validated record; duplicate run IDs and rewrites are rejected."""
    validate_result(record)
    if record["run_id"] in existing_run_ids(path):
        raise ValueError(f"run_id already exists; append-only ledger refuses overwrite: {record['run_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
