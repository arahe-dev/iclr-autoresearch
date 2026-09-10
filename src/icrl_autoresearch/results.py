from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .processes import FileLock


PLAN_ID = "G0-L8-SM120-EXACT-B32"

RESULT_REQUIRED_FIELDS = (
    "schema_version",
    "plan_id",
    "source_commit",
    "run_id",
    "arm_id",
    "block_id",
    "within_block_position",
    "selected_levels",
    "oracle",
    "latency_ms",
    "real_input_tok_s",
    "decision",
)


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _validate_commit(value: Any) -> None:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError("source_commit must be a full 40-character Git SHA")
    if any(char not in "0123456789abcdefABCDEF" for char in value):
        raise ValueError("source_commit must be a hexadecimal Git SHA")


def validate_result(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("result record must be a JSON object")
    missing = [key for key in RESULT_REQUIRED_FIELDS if key not in record]
    if missing:
        raise ValueError(f"missing result fields: {missing}")
    if record["schema_version"] != 1 or record["plan_id"] != PLAN_ID:
        raise ValueError("result does not belong to Generation-0 plan")
    _validate_commit(record["source_commit"])
    for field in ("run_id", "arm_id", "decision"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{field} must be a non-empty string")
    if isinstance(record["block_id"], bool) or not isinstance(record["block_id"], int) or record["block_id"] < 1:
        raise ValueError("block_id must be a positive integer")
    if isinstance(record["within_block_position"], bool) or not isinstance(record["within_block_position"], int) or record["within_block_position"] < 1:
        raise ValueError("within_block_position must be a positive integer")
    if not isinstance(record["selected_levels"], dict):
        raise ValueError("selected_levels must be an object")
    if not isinstance(record["oracle"], dict) or record["oracle"].get("status") != "PASS":
        raise ValueError("oracle correctness must pass before a result can be appended")
    _finite_positive(record["latency_ms"], "latency_ms")
    _finite_positive(record["real_input_tok_s"], "real_input_tok_s")
    if "peak_GiB" in record:
        _finite_positive(record["peak_GiB"], "peak_GiB")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """Read and validate every JSONL record; a partial/corrupt ledger is fatal."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"ledger contains a blank line at line {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"ledger line {line_number} is not valid JSON: {exc.msg}") from exc
            validate_result(value)
            run_id = value["run_id"]
            if run_id in seen:
                raise ValueError(f"ledger contains duplicate run_id: {run_id}")
            seen.add(run_id)
            records.append(value)
    return records


def existing_run_ids(path: Path) -> set[str]:
    return {record["run_id"] for record in read_ledger(path)}


def append_result(path: Path, record: dict[str, Any]) -> None:
    """Append one validated record; duplicate run IDs and rewrites are rejected."""
    validate_result(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with FileLock(lock_path, {"kind": "result-ledger", "ledger": str(path.resolve())}):
        if record["run_id"] in existing_run_ids(path):
            raise ValueError(f"run_id already exists; append-only ledger refuses overwrite: {record['run_id']}")
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())
