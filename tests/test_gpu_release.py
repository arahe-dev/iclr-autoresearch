"""CPU-only supervisor fault injection; synthetic protocol data is not evidence."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_harness_supervisor import ROOT, worker_fixture
from icrl_autoresearch import gpu_harness as harness
from icrl_autoresearch.generation0 import BASE_COMMIT, build_plan
from icrl_autoresearch.gpu import EXPECTED_GPU_NAME, GpuSnapshot, NvidiaSmiError
from icrl_autoresearch.processes import ProcessResult


class GpuReleaseTests(unittest.TestCase):
    def campaign(self, stage, fault=None, *, release_fails=False):
        payload, _, candidate, gpu, _, identity, _ = worker_fixture()
        provenance = identity["provenance"]
        plan = build_plan()
        order = []
        selected = GpuSnapshot(gpu["uuid"], EXPECTED_GPU_NAME, 94 * 1024, 0, 94 * 1024, "synthetic")
        primary = KeyboardInterrupt("worker interrupted") if fault == "interrupt" else OSError("worker launch failed")
        release_error = NvidiaSmiError("physical GPU release failed")

        def process(command, **kwargs):
            env = kwargs["env"]
            current = "preflight" if env["ICRL_G0_RUN_ID"] == "PREFLIGHT" else "benchmark"
            order.append(("process", current))
            kwargs["started"](123)
            active_fault = fault if current == stage else None
            if active_fault in {"interrupt", "launch"}:
                raise primary
            if current == "preflight":
                output = {
                    **payload["execution"]["gpu"], "status": "PREFLIGHT_PASS",
                    "plan_id": plan["plan_id"], "source_commit": candidate["candidate_commit"],
                    "harness_commit": provenance["harness"]["commit"],
                    "provenance": deepcopy(provenance), "corpus": provenance["corpus"],
                    "benchmark_limits": {"microbatch": 32, "warmups": 2, "timed_repetitions": 5},
                }
            else:
                output = deepcopy(payload)
                output["execution"]["attempt_id"] = env["ICRL_G0_ATTEMPT_ID"]
            output["attempt_id"] = env["ICRL_G0_ATTEMPT_ID"]
            if active_fault == "protocol":
                # For preflight this occurs after _validate_preflight succeeds.
                output["attempt_id"] = "wrong-attempt"
            elif active_fault == "oracle":
                output["oracle"]["status"] = "FAIL"
            elif active_fault == "memory":
                output["peak_GiB"] = 95.0
            elif active_fault == "champion":
                output["provenance"]["frozen"].pop("champion_commit")
            log = kwargs["log_path"]
            log.parent.mkdir(parents=True, exist_ok=True)
            text = "" if active_fault == "empty" else "not JSON" if active_fault == "json" else json.dumps(output)
            log.write_text(text, encoding="utf-8")
            return ProcessResult(123, 9 if active_fault == "nonzero" else 0, text, 1.0, active_fault == "timeout")

        def release(device, **kwargs):
            current = order[-1][1]
            self.assertEqual(device, selected)
            self.assertEqual(kwargs, {"baseline_free_mib": selected.free_mib})
            order.append(("release", current))
            if release_fails and current == stage:
                raise release_error
            return selected

        real_append = harness.append_result

        def admit(ledger, record):
            self.assertEqual(order[-1], ("release", "benchmark"))
            order.append(("admission", "benchmark"))
            if fault == "admission":
                raise ValueError("admission failed")
            return real_append(ledger, record)

        with tempfile.TemporaryDirectory(prefix="icrl-release-test-") as directory, ExitStack() as stack:
            root = Path(directory)
            candidate = {
                **candidate, "candidate_root": str(root / "candidate"), "base_commit": BASE_COMMIT,
                "champion_commit": BASE_COMMIT,
                "canonical_source_sha256": provenance["frozen"]["canonical_source_sha256"],
            }
            stack.enter_context(patch.object(harness, "assert_candidate_worktree", return_value=candidate))
            stack.enter_context(patch.object(harness, "_corpus_fingerprint", return_value=provenance["corpus"]))
            stack.enter_context(patch.object(harness, "code_identity", side_effect=lambda path: provenance["harness" if Path(path).resolve() == ROOT.resolve() else "candidate"]))
            stack.enter_context(patch.object(harness, "select_target_gpu", return_value=selected))
            stack.enter_context(patch.object(harness, "assert_idle_gpu"))
            stack.enter_context(patch.object(harness, "gpu_lock_path", return_value=root / "physical-gpu.lock"))
            # Keep the real _fake_or_real_gpu, _run_attempt, validation and state
            # machinery; intercept only process execution and physical GPU I/O.
            runner = stack.enter_context(patch.object(harness, "run_process", side_effect=process))
            wait = stack.enter_context(patch.object(harness, "wait_gpu_released", side_effect=release))
            stack.enter_context(patch.object(harness, "append_result", side_effect=admit))
            args = ("synthetic-worker", root / "candidate", root / "campaign" / "results.jsonl")
            expected_order = [("process", "preflight"), ("release", "preflight")]
            if stage == "benchmark":
                expected_order += [("process", "benchmark"), ("release", "benchmark")]
            if fault == "admission" or (fault is None and not release_fails):
                expected_order.append(("admission", "benchmark"))

            if fault is None and not release_fails:
                result = harness.execute_plan(*args, plan=plan, limit=1)
                self.assertEqual(result["status"], "PARTIAL")
                self.assertEqual(result["completed_count"], 1)
                self.assertTrue(args[2].exists())
            else:
                error_type = {
                    "nonzero": RuntimeError, "timeout": TimeoutError,
                    "interrupt": KeyboardInterrupt, "launch": OSError,
                }.get(fault, ValueError if fault else NvidiaSmiError)
                with self.assertRaises(error_type) as caught:
                    harness.execute_plan(*args, plan=plan, limit=1)
                if fault in {"interrupt", "launch"}:
                    self.assertIs(caught.exception, primary)
                if release_fails and fault:
                    self.assertIs(caught.exception.__cause__, release_error)
                elif release_fails:
                    self.assertIs(caught.exception, release_error)
                self.assertFalse(args[2].exists())
                state = json.loads((args[2].parent / "generation0_state.json").read_text())
                failed_run = "PREFLIGHT" if stage == "preflight" else plan["run_order"][0]["run_id"]
                self.assertEqual(state["status"], "FAILED")
                self.assertEqual(state["failed_run_id"], failed_run)
                self.assertEqual(state["completed_run_ids"], [])
                failure = json.loads((args[2].parent / "generation0_artifacts" / failed_run / "failure.json").read_text())
                self.assertEqual(failure["error_type"], error_type.__name__)
                if release_fails:
                    self.assertIn(str(release_error), failure["message"])
                # A subsequent invocation cannot resume or start a worker by default.
                counts = runner.call_count, wait.call_count
                with self.assertRaisesRegex(RuntimeError, "sticky-failed"):
                    harness.execute_plan(*args, plan=plan, limit=1)
                self.assertEqual((runner.call_count, wait.call_count), counts)
            self.assertEqual(order, expected_order)

    def test_release_on_worker_failures_and_interrupts_in_both_stages(self):
        for stage in ("preflight", "benchmark"):
            for fault in ("nonzero", "timeout", "launch", "interrupt", "empty", "json", "protocol", "champion"):
                with self.subTest(stage=stage, fault=fault):
                    self.campaign(stage, fault)

    def test_release_before_oracle_memory_and_admission_failures(self):
        for fault in ("oracle", "memory", "admission"):
            with self.subTest(fault=fault):
                self.campaign("benchmark", fault)

    def test_release_failure_is_sticky_and_preserves_worker_failure(self):
        for stage in ("preflight", "benchmark"):
            for fault in (None, "nonzero", "timeout", "interrupt", "json"):
                with self.subTest(stage=stage, fault=fault):
                    self.campaign(stage, fault, release_fails=True)

    def test_success_releases_both_workers_before_ledger_admission(self):
        self.campaign("benchmark")


if __name__ == "__main__":
    unittest.main()
