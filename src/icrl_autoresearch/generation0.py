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

from .contract import REPO_ROOT, load_contract
from .doe import load_generation0


PLAN_ID = "G0-L8-SM120-EXACT-B32"
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
ALIAS_GROUPS = {
    "full_symmetric_qk_sm120": ["full_symmetric_qk_sm120", "qk_backward*dy_gemm_backend", "dx_gemm_backend*e_gemm", "dy_relu_epilogue_layout*checkpoint_policy"],
    "qk_backward": ["qk_backward", "full_symmetric_qk_sm120*dy_gemm_backend", "dx_gemm_backend*dy_relu_epilogue_layout", "e_gemm*checkpoint_policy"],
    "dx_gemm_backend": ["dx_gemm_backend", "full_symmetric_qk_sm120*e_gemm", "qk_backward*dy_relu_epilogue_layout", "dy_gemm_backend*checkpoint_policy"],
    "dy_gemm_backend": ["dy_gemm_backend", "full_symmetric_qk_sm120*qk_backward", "e_gemm*dy_relu_epilogue_layout", "dx_gemm_backend*checkpoint_policy"],
    "e_gemm": ["e_gemm", "full_symmetric_qk_sm120*dx_gemm_backend", "dy_gemm_backend*dy_relu_epilogue_layout", "qk_backward*checkpoint_policy"],
    "dy_relu_epilogue_layout": ["dy_relu_epilogue_layout", "qk_backward*dx_gemm_backend", "dy_gemm_backend*e_gemm", "full_symmetric_qk_sm120*checkpoint_policy"],
    "checkpoint_policy": ["checkpoint_policy", "full_symmetric_qk_sm120*dy_relu_epilogue_layout", "qk_backward*e_gemm", "dx_gemm_backend*dy_gemm_backend"],
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
        "high_level": "selective_exact_projection_only",
        "direction": "ambiguous_latency__positive_memory_headroom",
        "magnitude_prior": "variable",
        "confidence": "low",
        "evidence_basis": "checkpointing is a known memory/runtime tradeoff; prior chunked-state failures are not evidence for this isolated policy",
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
    "checkpoint_policy": "Change only the checkpoint boundary to whole-level or selective exact projection-only recomputation; do not checkpoint the killed candidate QK-block read path and do not change the forward graph.",
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
                })
    return rows


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
            operations.append({"factor": name, "operation": PATCH_INTENT[name], "status": "SPEC_ONLY"})
    return {
        "patch_id": arm["arm_id"],
        "parent_experiment": "0000",
        "status": "SPEC_ONLY_NO_EXECUTION",
        "active_high_factors": active_high,
        "selected_levels": selected_levels,
        "operations": operations,
        "allowed_change_class": "execution_representation_only",
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


def randomized_run_order(seed: int = 20260909, blocks: int = 2) -> list[dict[str, Any]]:
    rows = [row["arm_id"] for row in l8_rows()]
    rng = random.Random(seed)
    order = []
    run_index = 0
    for block in range(1, blocks + 1):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        sequence: list[tuple[str, str]] = [("control_anchor", "CONTROL")]
        for pos, arm_id in enumerate(shuffled, 1):
            sequence.append(("arm", arm_id))
            if pos in (2, 4, 6):
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


def build_plan() -> dict[str, Any]:
    config = load_generation0()
    contract = load_contract()
    factor_map = _factor_map(config)
    if tuple(config["factor_order"]) != FACTOR_ORDER:
        raise ValueError("Generation-0 factor order drifted from the L8 generator")
    arms = l8_rows()
    return {
        "schema_version": 1,
        "plan_id": PLAN_ID,
        "base_commit": "908b0b1",
        "generation": 0,
        "status": "SPEC_ONLY_NO_EXECUTION",
        "control_experiment": "0000",
        "science_delta_allowed": False,
        "execution_allowed": False,
        "objective": config["objective"],
        "randomization": config["randomization"],
        "design": {
            "type": "Taguchi-style orthogonal array",
            "array": "L8(2^7)",
            "coding": "0=control, 1=exact execution candidate",
            "generators": {"A": "x", "B": "y", "C": "z", "D": "x xor y", "E": "x xor z", "F": "y xor z", "G": "x xor y xor z"},
            "factor_order": list(FACTOR_ORDER),
        },
        "factors": config["factors"],
        "predicted_effects": PREDICTED_EFFECTS,
        "interaction_aliases": ALIAS_GROUPS,
        "excluded_killed_branches": config["excluded_killed_branches"],
        "arms": [{**arm, "patch_spec": patch_spec(arm, factor_map)} for arm in arms],
        "run_order": randomized_run_order(config["randomization"]["seed"], config["randomization"]["blocks"]),
        "result_contract": {
            "append_only": True,
            "format": "JSONL",
            "schema": "schemas/generation0_result.schema.json",
            "one_record_per_run_id": True,
            "required_fields": ["run_id", "arm_id", "block_id", "oracle", "latency_ms", "real_input_tok_s", "decision", "source_commit"],
        },
        "next_round_handoff": {
            "status": "WAITING_FOR_SCREENING_RESULTS",
            "method": "fit main effects with block/position covariates; preserve alias uncertainty",
            "promotion_rule": "only exact-oracle-passing, memory-safe factors with stable sign across blocks",
            "interaction_round": "construct a full 2^k factorial over the selected factors; do not infer aliased interactions from L8",
            "source_commit_for_next_round": "the screening commit plus append-only result ledger",
        },
        "contract_sha256": __import__("hashlib").sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
