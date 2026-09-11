"""Generate the specification-only Generation-0 L8 screening design.

The module emits plans and schemas; it has no model runner and cannot modify a
champion checkout. Every arm is an exact execution representation with the
frozen Arm-A semantic contract attached.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .contract import REPO_ROOT, load_contract, source_sha256
from .doe import load_generation0


PLAN_ID = "G0-L8-SM120-EXACT-B32-C15"
BASE_COMMIT = "908b0b1438ba038d319adf97787aac08f213b590"
EXCLUDED_CELLS = [{
    "arm_id": "G0-F05",
    "level_string": "0100101",
    "status": "STRUCTURALLY_INFEASIBLE",
    "reason": "Observed B32 warmup OOM for the joint frozen treatment on the target SM120 GPU",
    "evidence_campaign": "iclr-g0-0ce16e9922b2",
    "result_policy": "excluded, never imputed or admitted as a measured result",
}]
MISSING_CELL_CAVEAT = (
    "G0-F05 is structurally infeasible. The intercept plus seven main-effect columns "
    "retain rank 8, but the constrained design is not exactly orthogonal and main "
    "effects are not dealiased from two-factor interactions. Estimates are conditional "
    "on the feasible cells and the regression model; do not impute the missing cell."
)
CHAMPION_BRANCH = "codex/champion/0000-validated-bdh-baseline"
CANONICAL_SOURCE = "vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py"
CANONICAL_SOURCE_SHA256 = "947f8b33e740adede3e13382f8cb9e8e374d845de75e282bda5feb05c582f624"
CONTROL_CHECKPOINT_POLICY = "gram_only_sac"
FACTOR_ORDER = (
    "full_symmetric_qk_sm120",
    "qk_backward",
    "dx_gemm_backend",
    "dy_gemm_backend",
    "e_gemm",
    "dy_relu_epilogue_layout",
    "checkpoint_policy",
)

# L8(2^7): A=x, B=y, C=z, D=xy, E=xz, F=yz, G=xyz in +/-1 coding.
# These are historical alias groups for the original complete design. Removing
# F05 loses main-effect/interaction orthogonality; these are not a de-alias claim.
INTERACTION_ALIAS_GROUPS = {
    "A": ["qk_backward*dy_gemm_backend", "dx_gemm_backend*e_gemm", "dy_relu_epilogue_layout*checkpoint_policy"],
    "B": ["full_symmetric_qk_sm120*dy_gemm_backend", "dx_gemm_backend*dy_relu_epilogue_layout", "e_gemm*checkpoint_policy"],
    "C": ["full_symmetric_qk_sm120*e_gemm", "qk_backward*dy_relu_epilogue_layout", "dy_gemm_backend*checkpoint_policy"],
    "D": ["full_symmetric_qk_sm120*qk_backward", "e_gemm*dy_relu_epilogue_layout", "dx_gemm_backend*checkpoint_policy"],
    "E": ["full_symmetric_qk_sm120*dx_gemm_backend", "dy_gemm_backend*dy_relu_epilogue_layout", "qk_backward*checkpoint_policy"],
    "F": ["qk_backward*dx_gemm_backend", "dy_gemm_backend*e_gemm", "full_symmetric_qk_sm120*checkpoint_policy"],
    "G": ["full_symmetric_qk_sm120*dy_relu_epilogue_layout", "qk_backward*e_gemm", "dx_gemm_backend*dy_gemm_backend"],
}

MAIN_EFFECT_NAMES = dict(zip("ABCDEFG", FACTOR_ORDER))


def _interaction_groups() -> dict[str, list[str]]:
    return {
        main: [f"{interaction}" for interaction in interactions]
        for main, interactions in INTERACTION_ALIAS_GROUPS.items()
    }

PREDICTED_EFFECTS = {
    "full_symmetric_qk_sm120": {
        "high_level": "native_sm120_full_symmetric_qk",
        "direction": "negative_latency",
        "magnitude_prior": "large",
        "confidence": "medium",
        "evidence_basis": "attention is a measured dominant component; exact attention probes passed",
        "caution": "full strict-lower same-document Q=K only; blocked-prefix read is excluded"
    },
    "qk_backward": {
        "high_level": "native_tied_qk_vjp",
        "direction": "negative_latency",
        "magnitude_prior": "medium",
        "confidence": "medium",
        "evidence_basis": "tied-QK gradient is an invariant and attention backward is a measured cost center",
        "caution": "must accumulate query and key roles into one exact Q gradient"
    },
    "dx_gemm_backend": {
        "high_level": "native_sm120_dx_gemm",
        "direction": "negative_latency",
        "magnitude_prior": "medium",
        "confidence": "medium",
        "evidence_basis": "full-shape projection/GEMM traffic and exact N64 fused tile evidence",
        "caution": "same operands, shapes, layouts, and accumulation contract"
    },
    "dy_gemm_backend": {
        "high_level": "native_sm120_dy_gemm",
        "direction": "negative_latency",
        "magnitude_prior": "medium",
        "confidence": "medium",
        "evidence_basis": "projection and pointwise materialization are measured costs",
        "caution": "Dy is exact; only backend/launch representation may change"
    },
    "e_gemm": {
        "high_level": "native_sm120_e_gemm",
        "direction": "negative_latency",
        "magnitude_prior": "medium",
        "confidence": "low_to_medium",
        "evidence_basis": "native SM120 GEMM is an explicit hardware hypothesis; no clean isolated E result yet",
        "caution": "do not use FP4, quantization, or an approximate E operator"
    },
    "dy_relu_epilogue_layout": {
        "high_level": "exact_dy_relu_epilogue_layout",
        "direction": "negative_latency",
        "magnitude_prior": "large",
        "confidence": "medium_to_high",
        "evidence_basis": "component diagnostic measured materialized neuron 53.947 ms, fused pointwise 25.119 ms, compiled 15.805 ms",
        "caution": "component evidence is not an end-to-end TPS claim"
    },
    "checkpoint_policy": {
        "low_level": "gram_only_sac",
        "high_level": "selective_exact_projection_only",
        "direction": "ambiguous_latency__positive_memory_headroom",
        "magnitude_prior": "variable",
        "confidence": "low",
        "evidence_basis": "experiment 0000 used exact Gram-only SAC; selective projection-only recomputation is a separate memory/runtime hypothesis; prior chunked-state failures are not evidence for this isolated policy",
        "caution": "selective candidate-QK-block checkpoint from 0001 is excluded"
    },
}

PATCH_INTENT = {
    "full_symmetric_qk_sm120": "Change only the attention backend dispatch to a native SM120 full/symmetric QK implementation; retain full T×T logical scores, strict-exclusive same-document masking, Q=K, shared V, no softmax, and no scale.",
    "qk_backward": "Change only the exact tied-QK VJP dispatch; accumulate the query-role and key-role contributions into the single Q gradient without materializing an independent K or changing the VJP.",
    "dx_gemm_backend": "Change only the Dx GEMM backend/launch; preserve dimensions, operand values, logical layout, BF16 operands, and FP32 accumulation/master-weight semantics.",
    "dy_gemm_backend": "Change only the Dy GEMM backend/launch; preserve exact Dy values and all downstream writer semantics.",
    "e_gemm": "Change only the E GEMM backend/launch; preserve exact E values, dimensions, layout contract, and FP32 accumulation/master weights.",
    "dy_relu_epilogue_layout": "Fuse or change only the Dy→ReLU epilogue/layout while preserving exact ReLU semantics, output layout contract, and the independent oracle tolerance.",
    "checkpoint_policy": "Change only the checkpoint boundary from the exact experiment-0000 Gram-only SAC control to selective exact projection-only recomputation; do not checkpoint the killed candidate QK-block read path and do not change the forward graph.",
}


def l8_rows() -> list[dict[str, Any]]:
    rows = []
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                levels = [x, y, z, x ^ y, x ^ z, y ^ z, x ^ y ^ z]
                index = len(rows)
                rows.append({
                    "arm_id": f"G0-{index:02d}",
                    "row_index": index,
                    "generator_bits": {"x": x, "y": y, "z": z},
                    "levels": dict(zip(FACTOR_ORDER, levels)),
                    "level_string": "".join(str(v) for v in levels),
                    "role": "baseline_control" if index == 0 else "L8_screening_arm",
                    "foldover": False,
                })
    return rows


def foldover_rows(base_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return the eight full-foldover rows, complementing every factor bit."""
    source_rows = base_rows if base_rows is not None else l8_rows()
    rows = []
    for index, base in enumerate(source_rows):
        levels = {name: 1 - int(value) for name, value in base["levels"].items()}
        rows.append({
            "arm_id": f"G0-F{index:02d}",
            "row_index": len(source_rows) + index,
            "source_arm_id": base["arm_id"],
            "levels": levels,
            "level_string": "".join(str(levels[name]) for name in FACTOR_ORDER),
            "role": "full_foldover_arm",
            "foldover": True,
        })
    return rows


def all_rows() -> list[dict[str, Any]]:
    base = l8_rows()
    return [row for row in base + foldover_rows(base) if row["arm_id"] != "G0-F05"]


def _factor_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in config["factors"]}


def patch_spec(arm: dict[str, Any], factor_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    levels = arm["levels"]
    operations = []
    active_high = []
    selected_levels = {}
    for name in FACTOR_ORDER:
        entry = factor_map[name]
        selected = entry["levels"][levels[name]]
        selected_levels[name] = selected
        if levels[name]:
            active_high.append(name)
            operations.append({"factor": name, "operation": PATCH_INTENT[name], "status": "READY_FOR_EXPLICIT_EXECUTION"})
    return {
        "patch_id": arm["arm_id"],
        "parent_experiment": "0000",
        "status": "READY_FOR_EXPLICIT_EXECUTION",
        "control_reference": {
            "experiment_id": "0000",
            "champion_commit": BASE_COMMIT,
            "champion_branch": CHAMPION_BRANCH,
            "canonical_source": CANONICAL_SOURCE,
            "canonical_source_sha256": CANONICAL_SOURCE_SHA256,
            "checkpoint_policy": CONTROL_CHECKPOINT_POLICY,
        },
        "active_high_factors": active_high,
        "selected_levels": selected_levels,
        "operations": operations,
        "allowed_change_class": "execution_representation_only",
        "champion_mutation": "forbidden",
        "required_invariants": [
            "Arm-A science unchanged",
            "exact full/symmetric Q=K attention",
            "strict-exclusive same-document masking and resets",
            "tied-QK backward",
            "no softmax and no attention scale",
            "real frozen corpus and valid-target accounting",
            "FP32 oracle and gradient gates pass before timing",
        ],
        "forbidden_operations": [
            "architecture or equation change",
            "approximate/quantized operator",
            "blocked-prefix read or [B,T,H,K] native-read branch",
            "chunked-state/SplitScan branch",
            "N128 rejected tile configurations",
            "optimizer, loss, corpus, or checkpoint-file semantics change",
        ],
        "measurement": {
            "benchmark": "B32",
            "objective": "exact forward+backward latency_ms",
            "control_latency_ms_approx": 1655.0,
            "target_latency_ms_approx": 728.0,
            "oracle_before_timing": True,
        },
    }


def original_run_order(seed: int = 20260909, blocks: int = 2) -> list[dict[str, Any]]:
    """Reconstruct the original order, including F05, solely for traceability."""
    rows = [row["arm_id"] for row in l8_rows() + foldover_rows()]
    rng = random.Random(seed)
    order = []
    run_index = 0
    for block in range(1, blocks + 1):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        sequence: list[tuple[str, str]] = [("control_anchor", "CONTROL")]
        for pos, arm_id in enumerate(shuffled, 1):
            sequence.append(("arm", arm_id))
            if pos in (2, 4, 6, 8, 10, 12, 14, 16):
                sequence.append(("control_interleave", "CONTROL"))
        sequence.append(("control_anchor", "CONTROL"))
        for within, (kind, arm_id) in enumerate(sequence, 1):
            run_index += 1
            order.append({
                "run_id": f"G0-R{run_index:02d}",
                "block_id": block,
                "within_block_position": within,
                "kind": kind,
                "arm_id": arm_id,
                "randomization_seed": seed,
                "oracle_required": True,
                "control_interleaved": kind != "arm" or arm_id == "G0-00",
            })
    return order


def randomized_run_order(seed: int = 20260909, blocks: int = 2) -> list[dict[str, Any]]:
    # Filter only after shuffling and allocating every original control slot.
    # Preserve run IDs and position covariates (gaps are deliberate).
    return [slot for slot in original_run_order(seed, blocks) if slot["arm_id"] != "G0-F05"]


def build_plan() -> dict[str, Any]:
    config = load_generation0()
    contract = load_contract()
    factor_map = _factor_map(config)
    if tuple(config["factor_order"]) != FACTOR_ORDER:
        raise ValueError("Generation-0 factor order drifted from the L8 generator")
    if config.get("plan_id") != PLAN_ID or config.get("excluded_cells") != EXCLUDED_CELLS:
        raise ValueError("Generation-0 constrained plan identity or exclusion drifted")
    if config["randomization"]["seed"] != 20260909 or config["randomization"]["blocks"] != 2:
        raise ValueError("Generation-0 must preserve the original randomized replicated order")
    arms = all_rows()
    control_levels = {factor["name"]: factor["levels"][0] for factor in config["factors"]}
    configured_foldover = {
        row["row_id"]: row["levels"] for row in config.get("foldover_rows", [])
    }
    generated_foldover = {
        row["arm_id"]: row["level_string"] for row in arms if row.get("foldover")
    }
    if configured_foldover != generated_foldover:
        raise ValueError("configured foldover rows drifted from the L8 bitwise complement")
    config_control = config.get("control_reference", {})
    if control_levels.get("checkpoint_policy") != CONTROL_CHECKPOINT_POLICY:
        raise ValueError("checkpoint_policy low level is not the exact experiment-0000 Gram-only SAC policy")
    if (
        config_control.get("champion_commit") != BASE_COMMIT
        or config_control.get("canonical_source_sha256") != CANONICAL_SOURCE_SHA256
        or config_control.get("factor_levels") != control_levels
    ):
        raise ValueError("configured control reference drifted from the immutable experiment-0000 champion")
    plan = {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "base_commit": BASE_COMMIT,
        "generation": 0,
        "status": "READY_FOR_EXPLICIT_EXECUTION",
        "control_experiment": "0000",
        "science_delta_allowed": False,
        "execution_allowed": True,
        "execution_requires_explicit_flag": True,
        "execution_mode": "candidate_worktree_only_gpu_harness",
        "foldover": config["foldover"],
        "control_reference": {
            "experiment_id": "0000",
            "champion_commit": BASE_COMMIT,
            "champion_branch": CHAMPION_BRANCH,
            "canonical_source": CANONICAL_SOURCE,
            "canonical_source_sha256": CANONICAL_SOURCE_SHA256,
            "checkpoint_policy": CONTROL_CHECKPOINT_POLICY,
            "factor_levels": control_levels,
            "control_equivalence": "byte-equivalent canonical source and behavior-equivalent exact Arm-A treatment before any factor is enabled",
        },
        "objective": config["objective"],
        "randomization": config["randomization"],
        "design": {
            "type": "constrained rank-complete main-effect screening design",
            "array": "original L8(2^7) + foldover, with G0-F05 excluded",
            "treatment_arms": 15,
            "control_slots": 20,
            "run_slots": 50,
            "main_effect_rank_with_intercept": 8,
            "exact_orthogonality": False,
            "missing_cell_caveat": MISSING_CELL_CAVEAT,
            "coding": "0=control, 1=exact execution candidate",
            "generators": {"A": "x", "B": "y", "C": "z", "D": "x xor y", "E": "x xor z", "F": "y xor z", "G": "x xor y xor z"},
            "factor_order": list(FACTOR_ORDER),
            "foldover": "original bitwise complements retained except structurally infeasible G0-F05",
        },
        "factors": config["factors"],
        "predicted_effects": PREDICTED_EFFECTS,
        "interaction_aliases": {
            "column_assignment": MAIN_EFFECT_NAMES,
            "main_effects_dealiased_from_two_factor_interactions": False,
            "missing_cell_caveat": MISSING_CELL_CAVEAT,
            "remaining_two_factor_interaction_groups": _interaction_groups(),
        },
        "excluded_killed_branches": config["excluded_killed_branches"],
        "excluded_cells": EXCLUDED_CELLS,
        "arms": [{**arm, "patch_spec": patch_spec(arm, factor_map)} for arm in arms],
        "run_order": randomized_run_order(config["randomization"]["seed"], config["randomization"]["blocks"]),
        "result_contract": {
            "append_only": True,
            "format": "JSONL",
            "schema": "schemas/generation0_result.schema.json",
            "one_record_per_run_id": True,
            "required_fields": ["schema_version", "plan_id", "source_commit", "harness_commit", "run_id", "arm_id", "treatment_id", "attempt_id", "block_id", "within_block_position", "selected_levels", "oracle", "latency_ms", "real_input_tok_s", "peak_GiB", "samples_ms", "warmups", "timed_repetitions", "timing", "memory", "provenance", "validation", "admission", "execution", "decision"],
        },
        "next_round_handoff": {
            "status": "WAITING_FOR_SCREENING_RESULTS",
            "method": "constrained regression of main effects with original block/position covariates; retain missing-cell and interaction uncertainty",
            "promotion_rule": "only exact-oracle-passing, memory-safe factors with stable sign across blocks",
            "interaction_round": "construct a full 2^k factorial over the selected factors; do not infer aliased interactions from L8",
            "source_commit_for_next_round": "the screening commit plus append-only result ledger",
        },
        "contract_sha256": __import__("hashlib").sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    assert_control_treatment(plan)
    plan["control_verification"] = {
        "status": "PASS_STATIC_CONTROL_GATES",
        "byte_equivalent_canonical_source": True,
        "canonical_source_sha256": CANONICAL_SOURCE_SHA256,
        "behavior_equivalent_frozen_arm_a_contract": True,
        "g0_00_has_no_factor_operations": True,
        "all_factor_low_levels_match_experiment_0000": True,
        "runtime_oracle_required_before_timing": True,
        "champion_mutation": "forbidden_until_results_analysis",
    }
    return plan


def assert_control_treatment(plan: dict[str, Any]) -> None:
    """Fail closed unless G0-00 is the exact imported 0000 control treatment."""
    if plan.get("plan_id") != PLAN_ID or plan.get("excluded_cells") != EXCLUDED_CELLS:
        raise ValueError("Generation-0 constrained plan identity or exclusion drifted")
    if plan.get("base_commit") != BASE_COMMIT:
        raise ValueError("control base commit drifted from experiment 0000")
    reference = plan.get("control_reference", {})
    if reference.get("champion_commit") != BASE_COMMIT or reference.get("champion_branch") != CHAMPION_BRANCH:
        raise ValueError("control reference is not anchored to the immutable 0000 champion")
    if reference.get("checkpoint_policy") != CONTROL_CHECKPOINT_POLICY:
        raise ValueError("control checkpoint policy is not experiment-0000 Gram-only SAC")
    expected_sha = reference.get("canonical_source_sha256")
    if expected_sha != CANONICAL_SOURCE_SHA256 or source_sha256() != expected_sha:
        raise ValueError("canonical source hash does not match the imported 0000 champion")
    source_manifest = json.loads((REPO_ROOT / "vendor" / "canonical" / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if source_manifest.get("sha256") != expected_sha or source_manifest.get("copy_mode") != "byte_for_byte":
        raise ValueError("canonical source manifest does not certify byte-equivalent 0000 provenance")
    baseline_manifest = json.loads((REPO_ROOT / "experiments" / "0000_validated_bdh_baseline" / "manifest.json").read_text(encoding="utf-8"))
    if (
        baseline_manifest.get("experiment_id") != "0000"
        or baseline_manifest.get("branch") != CHAMPION_BRANCH
        or baseline_manifest.get("status") != "BASELINE_REESTABLISHED"
        or baseline_manifest.get("science_delta") is not False
    ):
        raise ValueError("experiment 0000 baseline manifest is not the immutable champion reference")
    baseline_report = json.loads((REPO_ROOT / "experiments" / "0000_validated_bdh_baseline" / "report.json").read_text(encoding="utf-8"))
    required_baseline_science = {
        "model_changed": False,
        "arm": "A",
        "q_equals_k": True,
        "strict_exclusive_same_document_attention": True,
        "softmax": False,
        "attention_scale": False,
        "coordinator": "Cartesian",
        "writer": "dense",
    }
    if baseline_report.get("experiment_id") != "0000" or baseline_report.get("decision") != "KEEP_BASELINE":
        raise ValueError("experiment 0000 baseline report is not retained as the champion")
    for key, expected in required_baseline_science.items():
        if baseline_report.get("science", {}).get(key) != expected:
            raise ValueError(f"experiment 0000 behavior invariant changed: {key}")

    arms = plan.get("arms", [])
    controls = [arm for arm in arms if arm.get("arm_id") == "G0-00"]
    if len(controls) != 1:
        raise ValueError("Generation-0 must contain exactly one G0-00 control treatment")
    control = controls[0]
    if control["patch_spec"].get("active_high_factors"):
        raise ValueError("G0-00 control treatment unexpectedly enables a factor")
    factors = plan.get("factors", [])
    expected_levels = {factor["name"]: factor["levels"][0] for factor in factors}
    selected = control["patch_spec"].get("selected_levels")
    if selected != expected_levels:
        raise ValueError(f"G0-00 low-level treatment drifted from experiment 0000: {selected!r}")
    if control["patch_spec"].get("operations") != []:
        raise ValueError("G0-00 must have no patch operations")
    patch_reference = control["patch_spec"].get("control_reference", {})
    if patch_reference.get("checkpoint_policy") != CONTROL_CHECKPOINT_POLICY:
        raise ValueError("G0-00 patch spec does not carry Gram-only SAC control policy")
    if reference.get("factor_levels") != expected_levels:
        raise ValueError("control reference factor levels do not match the factor catalog")

    if len(arms) != 15 or plan.get("run_order") != randomized_run_order():
        raise ValueError("Generation-0 must contain 15 treatment arms and the original order minus the two F05 slots")
    expected_arms = [{**arm, "patch_spec": patch_spec(arm, _factor_map(load_generation0()))} for arm in all_rows()]
    if arms != expected_arms or factors != load_generation0()["factors"]:
        raise ValueError("Generation-0 arm definitions drifted from the frozen feasible treatments")
    if plan.get("design", {}).get("exact_orthogonality") is not False or plan.get("interaction_aliases", {}).get("main_effects_dealiased_from_two_factor_interactions") is not False:
        raise ValueError("constrained Generation-0 cannot claim exact orthogonality or de-aliasing")
    base_rows = [arm for arm in arms if not arm.get("foldover")]
    fold_rows = [arm for arm in arms if arm.get("foldover")]
    if len(base_rows) != 8 or len(fold_rows) != 7:
        raise ValueError("Generation-0 must contain eight base L8 rows and seven feasible foldover rows")
    for fold in fold_rows:
        source = next((row for row in base_rows if row["arm_id"] == fold.get("source_arm_id")), None)
        if source is None:
            raise ValueError(f"foldover source is missing for {fold['arm_id']}")
        expected_fold = {name: 1 - int(value) for name, value in source["levels"].items()}
        if fold["levels"] != expected_fold:
            raise ValueError(f"foldover row is not a bitwise complement: {fold['arm_id']}")

    # The canonical source is the byte-level control; these markers verify the
    # source still contains its full-Gram control and accepted SAC provenance.
    source = (REPO_ROOT / CANONICAL_SOURCE).read_text(encoding="utf-8")
    for marker in ("full Gram control", "historical accepted Gram-only SAC control", "native_read_sac_policy", "CheckpointPolicy"):
        if marker not in source:
            raise ValueError(f"canonical 0000 control marker missing: {marker}")
