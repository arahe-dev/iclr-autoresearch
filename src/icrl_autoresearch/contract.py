from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs" / "frozen_contract.json"


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def contract_sha256(contract: Mapping[str, Any] | None = None) -> str:
    payload = contract if contract is not None else load_contract()
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def source_sha256(path: Path | None = None) -> str:
    target = path or (REPO_ROOT / "vendor" / "canonical" / "icrl_reintegrated_bdh_baseline_v1.py")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def assert_science_unchanged(report: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> None:
    """Fail closed if a report claims a change to the frozen Arm-A science."""
    frozen = contract or load_contract()
    science = report.get("science", {})
    if science.get("model_changed") is not False:
        raise ValueError("report must explicitly set science.model_changed=false")
    required = {
        "arm": frozen["science"]["arm"],
        "q_equals_k": frozen["science"]["q_equals_k"],
        "strict_exclusive_same_document_attention": frozen["science"]["strict_exclusive_causality"],
        "softmax": frozen["science"]["softmax"],
        "attention_scale": frozen["science"]["attention_scale"],
    }
    for key, expected in required.items():
        if science.get(key) != expected:
            raise ValueError(f"science invariant changed: {key}={science.get(key)!r}; expected {expected!r}")


def assert_target_hardware(report: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> None:
    frozen = contract or load_contract()
    hardware = report.get("hardware", {})
    if hardware.get("sm") not in (None, frozen["target"]["sm"]):
        raise ValueError(f"report hardware is not the target {frozen['target']['sm']}: {hardware.get('sm')}")

