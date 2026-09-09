from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.benchmarks import B32, B64, summarize, throughput_tok_s
from icrl_autoresearch.contract import load_contract, source_sha256
from icrl_autoresearch.decisions import evaluate_report, read_gate
from icrl_autoresearch.doe import screening_plan
from icrl_autoresearch.generation0 import FACTOR_ORDER, assert_control_treatment, build_plan
from icrl_autoresearch.gpu_harness import assert_candidate_worktree, run_context
from icrl_autoresearch.validation import validate_repo


ROOT = Path(__file__).resolve().parents[1]


class InfrastructureTests(unittest.TestCase):
    def test_source_hash_and_contract(self) -> None:
        contract = load_contract()
        self.assertEqual(source_sha256(), contract["canonical_source"]["sha256"])
        self.assertEqual(B32.real_input_tokens, 65536)
        self.assertEqual(B64.real_input_tokens, 131072)

    def test_throughput_summary(self) -> None:
        self.assertAlmostEqual(throughput_tok_s(131072, 1000.0), 131072.0)
        result = summarize([100.0, 110.0, 105.0], B32)
        self.assertEqual(result["median_ms"], 105.0)

    def test_read_gate_matches_imported_negative(self) -> None:
        decision = read_gate(160.34422302246094, 199.0870361328125, correctness_pass=True)
        self.assertEqual(decision.decision, "REVERT_READ_REPRESENTATION")
        self.assertTrue(decision.revert)

    def test_generation_zero_is_explicitly_gated(self) -> None:
        plan = screening_plan()
        self.assertFalse(plan["science_delta_allowed"])
        self.assertEqual(plan["status"], "READY_FOR_EXPLICIT_EXECUTION")
        self.assertTrue(plan["execution_allowed"])
        self.assertTrue(plan["execution_requires_explicit_flag"])
        self.assertTrue(all(row["science_delta"] is False for row in plan["rows"]))

    def test_generation0_l8_foldover_is_balanced_and_control_anchored(self) -> None:
        plan = build_plan()
        assert_control_treatment(plan)
        self.assertEqual(plan["base_commit"], "908b0b1")
        self.assertEqual(plan["arms"][0]["level_string"], "0000000")
        self.assertEqual(plan["arms"][0]["patch_spec"]["selected_levels"]["checkpoint_policy"], "gram_only_sac")
        self.assertEqual(len(plan["arms"]), 16)
        for factor in FACTOR_ORDER:
            self.assertEqual(sum(arm["levels"][factor] for arm in plan["arms"]), 8)
        self.assertEqual(len(plan["run_order"]), 52)
        self.assertTrue(plan["execution_allowed"])
        self.assertTrue(plan["execution_requires_explicit_flag"])
        self.assertTrue(all(arm["patch_spec"]["status"] == "READY_FOR_EXPLICIT_EXECUTION" for arm in plan["arms"]))
        base = {arm["arm_id"]: arm for arm in plan["arms"] if not arm["foldover"]}
        for arm in (arm for arm in plan["arms"] if arm["foldover"]):
            source = base[arm["source_arm_id"]]
            self.assertEqual(arm["levels"], {name: 1 - value for name, value in source["levels"].items()})

    def test_gpu_harness_control_context_is_baseline(self) -> None:
        plan = build_plan()
        slot = next(slot for slot in plan["run_order"] if slot["arm_id"] == "CONTROL")
        context = run_context(plan, slot)
        self.assertTrue(context["champion_immutable"])
        self.assertEqual(context["selected_levels"], plan["control_reference"]["factor_levels"])
        self.assertEqual(context["selected_levels"]["checkpoint_policy"], "gram_only_sac")

    def test_gpu_harness_refuses_planning_workspace(self) -> None:
        with self.assertRaises(ValueError):
            assert_candidate_worktree(ROOT)

    def test_control_checkpoint_drift_fails_closed(self) -> None:
        plan = build_plan()
        plan["arms"][0]["patch_spec"]["selected_levels"]["checkpoint_policy"] = "whole_level"
        with self.assertRaises(ValueError):
            assert_control_treatment(plan)

    def test_repository_validation(self) -> None:
        result = validate_repo(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["baseline_experiment"], "0000")
        self.assertEqual(result["negative_experiment"], "0001")

    def test_exact_oracle_if_torch_is_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed in the local repository-check environment")
        from icrl_autoresearch.oracle import exact_qk_attention, exact_qk_vjp

        torch.manual_seed(0)
        q = torch.randn(1, 2, 4, 3, dtype=torch.float64)
        v = torch.randn(1, 4, 2, dtype=torch.float64)
        seg = torch.tensor([[0, 0, 1, 1]])
        valid = torch.tensor([[True, True, True, False]])
        y = exact_qk_attention(q, v, seg, valid)
        self.assertTrue(torch.allclose(y[:, :, 0], torch.zeros_like(y[:, :, 0])))
        self.assertTrue(torch.allclose(y[:, :, 2], torch.zeros_like(y[:, :, 2])))
        self.assertTrue(torch.all(y[:, :, 3] == 0))
        dq, dv = exact_qk_vjp(q, v, seg, valid, torch.ones_like(y))
        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dv.shape, v.shape)


if __name__ == "__main__":
    unittest.main()
