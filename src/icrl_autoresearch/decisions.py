from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


READ_PROMOTE_SPEEDUP = 1.10
FULL_REVERT_BELOW = 1.05
FULL_KEEP_AT = 1.10
TARGET_REAL_TOK_S = 90000.0
TARGET_B64_MS = 131072.0 / TARGET_REAL_TOK_S * 1000.0


@dataclass(frozen=True)
class Decision:
    decision: str
    keep: bool
    revert: bool
    speedup: float | None
    target_pass: bool | None
    reason: str


def read_gate(
    control_fb_ms: float,
    candidate_fb_ms: float,
    *,
    correctness_pass: bool,
    promote_speedup: float = READ_PROMOTE_SPEEDUP,
) -> Decision:
    if candidate_fb_ms <= 0 or control_fb_ms <= 0:
        raise ValueError("read-gate timings must be positive")
    speedup = control_fb_ms / candidate_fb_ms
    if not correctness_pass:
        return Decision("REVERT_CORRECTNESS_GATE", False, True, speedup, False, "exact correctness gate failed")
    if speedup < promote_speedup:
        return Decision(
            "REVERT_READ_REPRESENTATION",
            False,
            True,
            speedup,
            False,
            f"read speedup {speedup:.3f}x is below the {promote_speedup:.2f}x promote gate",
        )
    return Decision("PROMOTE_TO_FULL_MODEL", False, False, speedup, None, "read gate passed")


def full_gate(
    baseline_b64_ms: float,
    candidate_b64_ms: float,
    *,
    correctness_pass: bool,
    memory_regression: bool = False,
    target_tok_s: float = TARGET_REAL_TOK_S,
) -> Decision:
    if baseline_b64_ms <= 0 or candidate_b64_ms <= 0:
        raise ValueError("full-gate timings must be positive")
    speedup = baseline_b64_ms / candidate_b64_ms
    candidate_tok_s = 131072.0 / (candidate_b64_ms / 1000.0)
    target_pass = candidate_tok_s >= target_tok_s
    if not correctness_pass:
        return Decision("REVERT_CORRECTNESS_GATE", False, True, speedup, target_pass, "exact correctness gate failed")
    if memory_regression:
        return Decision("REVERT_MEMORY_REGRESSION", False, True, speedup, target_pass, "candidate regressed the declared memory gate")
    if candidate_b64_ms <= baseline_b64_ms / 2.0:
        return Decision("TWO_X_GATE_PASSED__KEEP", True, False, speedup, target_pass, "candidate passed the early two-times gate")
    if speedup >= FULL_KEEP_AT:
        return Decision("KEEP_EXACT_EXECUTION_REPRESENTATION", True, False, speedup, target_pass, "full B64 speedup passed the keep gate")
    if speedup >= FULL_REVERT_BELOW:
        return Decision("MARGINAL__PROFILE_BEFORE_MORE_ENGINEERING", False, False, speedup, target_pass, "candidate is marginal; no champion promotion")
    return Decision("REVERT_FULL_CANDIDATE", False, True, speedup, target_pass, "full B64 speedup is below the revert threshold")


def evaluate_report(report: Mapping[str, Any]) -> Decision:
    """Evaluate a report already containing measurements; never executes a run."""
    if report.get("decision") == "KEEP_BASELINE" or report.get("status") == "BASELINE_REESTABLISHED":
        return Decision("KEEP_BASELINE", True, False, None, False, "validated imported control")
    benches = report.get("benchmarks", {})
    read = benches.get("B32_read_gate") or benches.get("B32_read_micro")
    full = benches.get("B64", {})
    if read and read.get("candidate") and read.get("control") and full.get("status") == "NOT_RUN":
        corr = report.get("correctness", {}).get("fp32_gradient_gate") == "PASS"
        return read_gate(read["control"]["fb_ms"], read["candidate"]["fb_ms"], correctness_pass=corr)
    baseline = report.get("parent_baseline", {})
    if full.get("median_update_ms") and baseline.get("median_B64_ms"):
        corr = report.get("correctness", {}).get("status") == "PASS"
        return full_gate(
            baseline["median_B64_ms"],
            full["median_update_ms"],
            correctness_pass=corr,
            memory_regression=report.get("gates", {}).get("memory_regression") == "FAIL",
        )
    raise ValueError("report does not contain a recognized read or full B64 gate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a recorded exact-BDH report without running it")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(json.dumps(asdict(evaluate_report(report)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
