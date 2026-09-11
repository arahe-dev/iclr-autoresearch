"""CPU-only checks of the C15 plan and strict measurement admission protocol."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_harness_supervisor import ROOT, worker_fixture
from icrl_autoresearch.generation0 import BASE_COMMIT, CHAMPION_BRANCH, build_plan, assert_control_treatment
from icrl_autoresearch.gpu_harness import _manifest_identity, _validate_worker_payload, assert_campaign_paths_writable, run_context
from icrl_autoresearch.results import append_result, read_ledger, validate_result, RESULT_REQUIRED_FIELDS


class C15ProvenanceTests(unittest.TestCase):
    def test_original_slot_bytes_and_rank(self):
        plan = build_plan()
        original = json.loads(subprocess.check_output([
            "git", "show", "d1ae1e57e0635439155a15380538aba2330e75d7:plans/generation0/run_order.json"
        ], cwd=ROOT, text=True))["runs"]
        self.assertEqual(plan["run_order"], [slot for slot in original if slot["arm_id"] != "G0-F05"])
        # Exact rational elimination verifies the claim, without a numeric library.
        from fractions import Fraction
        rows = [[Fraction(1)] + [Fraction(2 * bit - 1) for bit in arm["levels"].values()] for arm in plan["arms"]]
        rank = 0
        for col in range(8):
            pivot = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
            if pivot is None:
                continue
            rows[rank], rows[pivot] = rows[pivot], rows[rank]
            value = rows[rank][col]
            rows[rank] = [x / value for x in rows[rank]]
            for i in range(rank + 1, len(rows)):
                value = rows[i][col]
                rows[i] = [x - value * y for x, y in zip(rows[i], rows[rank])]
            rank += 1
        self.assertEqual(rank, 8)
        plan["run_order"][1]["arm_id"] = "G0-F05"
        with self.assertRaises(ValueError):
            assert_control_treatment(plan)

    def test_complete_record_schema_and_append_only(self):
        record = _validate_worker_payload(*worker_fixture())
        import jsonschema
        result_schema = json.loads((ROOT / "schemas/generation0_result.schema.json").read_text())
        jsonschema.validate(record, result_schema)
        plan = build_plan()
        jsonschema.validate(plan, json.loads((ROOT / "schemas/generation_0_plan.schema.json").read_text()))
        self.assertEqual(set(RESULT_REQUIRED_FIELDS), set(result_schema["required"]))
        self.assertEqual(set(RESULT_REQUIRED_FIELDS), set(plan["result_contract"]["required_fields"]))
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "test.jsonl"
            append_result(ledger, record)
            before = ledger.read_bytes()
            with self.assertRaises(ValueError):
                append_result(ledger, record)
            self.assertEqual(ledger.read_bytes(), before)
            self.assertEqual(read_ledger(ledger), [record])

    def test_missing_and_tampered_provenance_fails_closed(self):
        record = _validate_worker_payload(*worker_fixture())
        for field in RESULT_REQUIRED_FIELDS:
            bad = deepcopy(record)
            bad.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_result(bad)
        mutations = [
            ("source_commit", "a" * 7), ("harness_commit", "d" * 7),
            ("arm_id", "G0-F05"), ("run_id", "G0-R05"),
            ("peak_GiB", 95.0), ("latency_ms", 2.0), ("real_input_tok_s", 3.0),
            ("samples_ms", [1.0] * 4), ("warmups", 0), ("schema_version", 1),
        ]
        for key, value in mutations:
            bad = deepcopy(record)
            bad[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_result(bad)
        for path, value in [
            (("execution", "gpu", "device"), "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"),
            (("execution", "gpu", "sm"), "sm_90"),
            (("execution", "gpu", "gpu_uuid"), None),
            (("provenance", "corpus", "fingerprint"), "0" * 64),
            (("provenance", "frozen", "config_sha256"), "0" * 64),
            (("provenance", "frozen", "champion_commit"), "0" * 40),
            (("provenance", "frozen", "champion_branch"), "codex/champion/other"),
            (("provenance", "candidate", "code_sha256"), "0" * 64),
            (("oracle", "dq_rel_l2"), 0.1),
            (("validation", "oracle_before_timing"), False),
            (("admission", "gpu_released"), False),
            (("timing", "boundaries", "synchronize_before_start"), False),
        ]:
            bad = deepcopy(record)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_result(bad)

    def test_champion_identity_is_self_contained_and_required(self):
        args = worker_fixture()
        record = _validate_worker_payload(*args)
        frozen = record["provenance"]["frozen"]
        self.assertEqual(frozen["champion_commit"], BASE_COMMIT)
        self.assertEqual(frozen["champion_branch"], CHAMPION_BRANCH)
        candidate = {**args[2], "canonical_source_sha256": frozen["canonical_source_sha256"]}
        manifest = _manifest_identity(build_plan(), candidate, args[3], "synthetic-worker",
                                      record["provenance"]["corpus"], 3600, None,
                                      Path("synthetic-results.jsonl"), record["provenance"])
        self.assertEqual(manifest["champion_commit"], BASE_COMMIT)
        self.assertEqual(manifest["champion_branch"], CHAMPION_BRANCH)
        import jsonschema
        schema = json.loads((ROOT / "schemas/generation0_result.schema.json").read_text())
        for field in ("champion_commit", "champion_branch"):
            for replacement in (None, "tampered"):
                bad = deepcopy(record)
                if replacement is None:
                    bad["provenance"]["frozen"].pop(field)
                else:
                    bad["provenance"]["frozen"][field] = replacement
                with self.subTest(field=field, replacement=replacement):
                    with self.assertRaises(ValueError):
                        validate_result(bad)
                    with self.assertRaises(jsonschema.ValidationError):
                        jsonschema.validate(bad, schema)

    def test_schemas_bind_c15_fields_and_factor_arm_relationships(self):
        import jsonschema
        plan = build_plan()
        plan_validator = jsonschema.Draft202012Validator(json.loads((ROOT / "schemas/generation_0_plan.schema.json").read_text()))
        result_validator = jsonschema.Draft202012Validator(json.loads((ROOT / "schemas/generation0_result.schema.json").read_text()))
        plan_validator.validate(plan)
        for field in ("control_reference", "control_verification", "contract_sha256", "result_contract",
                      "execution_mode", "randomization", "objective", "foldover"):
            bad = deepcopy(plan)
            bad.pop(field)
            with self.subTest(missing=field), self.assertRaises(jsonschema.ValidationError):
                plan_validator.validate(bad)
        for path, value in [
            (("factors", 0, "name"), "unknown_factor"),
            (("factors", 0, "levels"), ["arbitrary", "levels"]),
            (("arms", 0, "levels", "checkpoint_policy"), 1),
            (("arms", 0, "patch_spec", "selected_levels", "checkpoint_policy"), "selective_exact_projection_only"),
            (("arms", 1, "patch_spec", "patch_id"), "G0-00"),
            (("arms", 8, "source_arm_id"), "G0-01"),
            (("arms", 0, "arm_id"), "G0-F05"),
            (("run_order", 0, "run_id"), "G0-R05"),
            (("run_order", 0, "kind"), "arm"),
            (("control_reference", "champion_commit"), "908b0b1"),
            (("control_reference", "champion_branch"), "codex/champion/other"),
            (("result_contract", "required_fields"), ["run_id"]),
        ]:
            bad = deepcopy(plan)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(plan_path=path), self.assertRaises(jsonschema.ValidationError):
                plan_validator.validate(bad)
        # Exercise every feasible factor assignment plus the CONTROL alias.
        seen = set()
        for slot in plan["run_order"]:
            if slot["arm_id"] in seen:
                continue
            seen.add(slot["arm_id"])
            args = list(worker_fixture())
            args[1] = run_context(plan, slot)
            args[0].update(args[1])
            record = _validate_worker_payload(*args)
            result_validator.validate(record)
            for key, value in (
                ("treatment_id", "G0-01" if record["treatment_id"] != "G0-01" else "G0-00"),
                ("selected_levels", {**record["selected_levels"], "checkpoint_policy": "arbitrary"}),
                ("selected_levels", {**record["selected_levels"], "checkpoint_policy":
                    "selective_exact_projection_only" if record["selected_levels"]["checkpoint_policy"] == "gram_only_sac" else "gram_only_sac"}),
                ("selected_levels", {f"arbitrary_{i}": "x" for i in range(7)}),
                ("arm_id", "G0-F05"), ("run_id", "G0-R40"),
            ):
                bad = {**record, key: value}
                with self.subTest(arm=slot["arm_id"], field=key), self.assertRaises(jsonschema.ValidationError):
                    result_validator.validate(bad)

    def test_worker_mismatch_not_normalized_away(self):
        for field, value in (("plan_id", "old-plan"), ("attempt_id", "f" * 32), ("harness_commit", "f" * 40)):
            args = worker_fixture()
            args[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                _validate_worker_payload(*args)

    def test_old_evidence_campaign_refused_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "iclr-g0-0ce16e9922b2" / "results.jsonl"
            with self.assertRaises(ValueError):
                assert_campaign_paths_writable([target])
            self.assertFalse(target.parent.exists())

    def test_colab_cell_defaults_to_preparation_without_colab_import(self):
        # Normal Python has no google.colab. The preparation path must still run.
        result = subprocess.run([sys.executable, str(ROOT / "scripts/colab_generation0.py")], cwd=ROOT,
                                text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "PREPARED_NO_EXECUTION")
        self.assertEqual(payload["run_slots"], 50)
        self.assertEqual(payload["original_control_slots"], 20)


if __name__ == "__main__":
    unittest.main()
