from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icrl_autoresearch.gpu_harness import (  # noqa: E402
    _active_attempts,
    _command_argv,
    _corpus_fingerprint,
    _validate_worker_payload,
    _run_attempt,
)
from icrl_autoresearch.processes import ProcessResult, run_process  # noqa: E402
from icrl_autoresearch.generation0 import build_plan
from icrl_autoresearch.gpu_harness import run_context
from icrl_autoresearch.contract import load_contract
from icrl_autoresearch.provenance import CORPUS_FILES, TIMING_BOUNDARIES, frozen_identity, json_sha256


def worker_fixture():
    """Synthetic protocol data only; never GPU evidence."""
    plan = build_plan()
    context = run_context(plan, plan["run_order"][0])
    frozen = frozen_identity(plan)
    candidate = {"candidate_commit": "a" * 40, "candidate_branch": "codex/generation0/test"}
    gpu = {"uuid": "GPU-12345678-1234-1234-1234-123456789abc", "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "vram_GiB": 94.0}
    files = {"configs/generation_0_screening.json": frozen["config_sha256"],
             "vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py": frozen["canonical_source_sha256"]}
    code = {"commit": "a" * 40, "code_sha256": json_sha256(files), "files": files}
    corpus = {"root": "/synthetic-corpus", "status": "RESOLVED", "frozen_json_sha256": "b" * 64,
              "frozen_json": {"status": "FROZEN", **{key: load_contract()["data"][key] for key in (
                  "corpus_id", "context_length", "tokenizer_sha256", "logical_replay_sha256")}},
              "required_files": [{"path": name, "exists": True, "size": 1, "sha256": "c" * 64} for name in CORPUS_FILES]}
    corpus["fingerprint"] = json_sha256(corpus)
    provenance = {"candidate": code, "harness": {**code, "commit": "d" * 40}, "frozen": frozen, "corpus": corpus}
    identity = {"provenance": provenance, "harness_commit": "d" * 40}
    intervals = [{"phase": f"warmup_{i + 1}" if i < 2 else f"timing_{i - 1}",
                  "started_at": f"2026-01-01T00:00:{2 * i + 1:02d}+00:00",
                  "ended_at": f"2026-01-01T00:00:{2 * i + 2:02d}+00:00",
                  "start_monotonic_ns": (2 * i + 1) * 10**9, "end_monotonic_ns": (2 * i + 2) * 10**9,
                  "elapsed_ms": 1.0, "peak_GiB": 1.0} for i in range(7)]
    payload = {
        **context, "schema_version": 2, "source_commit": "a" * 40, "harness_commit": "d" * 40,
        "attempt_id": "e" * 32, "provenance": provenance,
        "oracle": {"status": "PASS", "same_document_reset": "PASS", "tied_qk_backward": "PASS",
                   "initial_loss_gate": "PASS", "initial_loss": load_contract()["training"]["initial_loss_gate"],
                   **{key: 0.0 for key in ("forward_rel_l2", "dq_rel_l2", "dv_rel_l2", "projection_rel_l2", "projection_grad_rel_l2", "forward_max_abs")}},
        "latency_ms": 1.0, "real_input_tok_s": 65536000.0, "peak_GiB": 1.0,
        "samples_ms": [1.0] * 5, "warmups": 2, "timed_repetitions": 5,
        "timing": {"boundaries": TIMING_BOUNDARIES, "real_input_tokens": 65536, "valid_targets": 65000, "intervals": intervals},
        "validation": {"status": "PASS", "frozen_corpus": "PASS", "frozen_contract": "PASS", "selected_treatment": "PASS", "initial_loss": "PASS", "oracle_before_timing": True, "completed_before_timing_at": "2026-01-01T00:00:00+00:00"},
        "memory": {"status": "PASS", "peak_GiB": 1.0, "capacity_GiB": 94.0},
        "execution": {"candidate_commit": "a" * 40, "attempt_id": "e" * 32, "pid": 123,
                      "champion_immutable": True, "gpu": {"device": gpu["device"], "gpu_uuid": gpu["uuid"], "sm": "sm_120", "vram_GiB": 94.0, "torch": "synthetic", "cuda": "synthetic"}},
        "decision": "MEASURED_G0_B32_FB",
    }
    return deepcopy((payload, context, candidate, gpu, "e" * 32, identity, ProcessResult(123, 0, "", 1.0, False)))


class HarnessSupervisorTests(unittest.TestCase):
    def test_command_template_is_argv_only(self) -> None:
        self.assertEqual(_command_argv('python -c "print(1)"'), ["python", "-c", "print(1)"])
        with self.assertRaises(ValueError):
            _command_argv("python worker.py && echo unsafe")

    def test_process_timeout_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory(prefix="icrl-process-") as directory:
            root = Path(directory)
            result = run_process(
                [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
                cwd=root,
                env=os.environ.copy(),
                log_path=root / "worker.log",
                timeout_seconds=0.2,
            )
            self.assertTrue(result.timed_out)
            self.assertIn("started", result.tail)

    def test_attempt_events_have_begin_started_end_and_payload(self) -> None:
        with tempfile.TemporaryDirectory(prefix="icrl-attempt-") as directory:
            root = Path(directory)
            console_path = root / "console.log"
            release = Mock()
            with console_path.open("a", encoding="utf-8") as console:
                attempt_id, result, payload = _run_attempt(
                    stage="preflight",
                    run_id="PREFLIGHT",
                    command=[sys.executable, "-c", "import json; print(json.dumps({'status':'PREFLIGHT_PASS'}))"],
                    env=os.environ.copy(),
                    cwd=ROOT,
                    log_path=root / "worker.log",
                    console=console,
                    attempts_path=root / "attempts.jsonl",
                    artifact_root=root / "artifacts",
                    context={},
                    gpu={},
                    timeout_seconds=5,
                    inherit_fds=(),
                    await_release=release,
                )
            release.assert_called_once_with()
            self.assertEqual(result.returncode, 0)
            self.assertEqual(payload["status"], "PREFLIGHT_PASS")
            events = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["begin", "started", "end"])
            self.assertTrue(all(event["attempt_id"] == attempt_id for event in events))

    def test_unfinished_attempt_requires_explicit_recovery(self) -> None:
        active = _active_attempts([{"event": "begin", "attempt_id": "a", "run_id": "G0-R01", "stage": "benchmark"}])
        self.assertEqual(active["a"]["run_id"], "G0-R01")
        with self.assertRaises(ValueError):
            _active_attempts([{"event": "end", "attempt_id": "a"}])

    def test_worker_payload_requires_matching_attempt_and_context(self) -> None:
        args = worker_fixture()
        record = _validate_worker_payload(*args)
        self.assertEqual(record["execution"]["candidate_commit"], "a" * 40)
        args[0]["latency_ms"] = float("nan")
        with self.assertRaises(ValueError):
            _validate_worker_payload(*args)

    def test_corpus_fingerprint_marks_missing_required_files_incomplete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="icrl-corpus-") as directory:
            root = Path(directory)
            (root / "FROZEN.json").write_text("{}\n", encoding="utf-8")
            fingerprint = _corpus_fingerprint(root)
            self.assertEqual(fingerprint["status"], "INCOMPLETE")
            self.assertEqual(len(fingerprint["required_files"]), 3)


if __name__ == "__main__":
    unittest.main()
