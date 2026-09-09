from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contract import REPO_ROOT, assert_science_unchanged, contract_sha256, load_contract, source_sha256
from .decisions import evaluate_report


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    return rows


def validate_repo(root: Path = REPO_ROOT) -> dict[str, Any]:
    contract = load_contract(root / "configs" / "frozen_contract.json")
    source = root / "vendor" / "canonical" / "icrl_reintegrated_bdh_baseline_v1.py"
    if not source.exists():
        raise ValueError("canonical source is missing")
    actual_source_sha = source_sha256(source)
    if actual_source_sha != contract["canonical_source"]["sha256"]:
        raise ValueError(f"canonical source hash mismatch: {actual_source_sha}")

    baseline = read_json(root / "experiments/0000_validated_bdh_baseline/report.json")
    reverted = read_json(root / "experiments/0001_native_read_reverted/report.json")
    assert_science_unchanged(baseline, contract)
    assert_science_unchanged(reverted, contract)
    baseline_decision = evaluate_report(baseline)
    reverted_decision = evaluate_report(reverted)
    if baseline_decision.decision != "KEEP_BASELINE":
        raise ValueError(f"unexpected baseline decision: {baseline_decision}")
    if reverted_decision.decision != "REVERT_READ_REPRESENTATION":
        raise ValueError(f"unexpected 0001 decision: {reverted_decision}")

    doe = read_json(root / "configs/generation_0_screening.json")
    if doe.get("generation") != 0 or doe.get("science_delta_allowed") is not False or doe.get("execution_allowed") is not False:
        raise ValueError("Generation-0 screening is not fail-closed")
    evidence = validate_jsonl(root / "evidence/history.jsonl")
    experiments = validate_jsonl(root / "experiments/history.jsonl")
    if not any(row.get("evidence_id") == "0001-native-read-reverted-20260909" for row in evidence):
        raise ValueError("new native-read negative result is missing from evidence history")
    if not any(row.get("experiment_id") == "0000" for row in experiments):
        raise ValueError("experiment 0000 is missing from history")

    return {
        "status": "PASS",
        "contract_sha256": contract_sha256(contract),
        "canonical_source_sha256": actual_source_sha,
        "baseline_experiment": baseline["experiment_id"],
        "negative_experiment": reverted["experiment_id"],
        "evidence_records": len(evidence),
        "experiment_records": len(experiments),
        "generation0_execution": doe["execution_allowed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the imported exact-BDH autoresearch scaffold")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate_repo(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

