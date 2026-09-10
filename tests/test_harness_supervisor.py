from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icrl_autoresearch.gpu_harness import (  # noqa: E402
    _active_attempts,
    _command_argv,
    _validate_worker_payload,
    _run_attempt,
)
from icrl_autoresearch.processes import run_process  # noqa: E402


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
                )
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
        context = {
            "plan_id": "G0-L8-SM120-EXACT-B32",
            "run_id": "G0-R01",
            "arm_id": "CONTROL",
            "block_id": 1,
            "within_block_position": 1,
            "selected_levels": {"x": "control"},
        }
        candidate = {"candidate_commit": "a" * 40, "candidate_branch": "codex/generation0/test"}
        gpu = {"uuid": "test", "vram_GiB": 94.0}
        payload = {
            **context,
            "source_commit": "a" * 40,
            "attempt_id": "attempt",
            "oracle": {"status": "PASS", "same_document_reset": "PASS", "tied_qk_backward": "PASS"},
            "latency_ms": 1.0,
            "real_input_tok_s": 2.0,
            "peak_GiB": 1.0,
            "decision": "MEASURED_G0_B32_FB",
        }
        record = _validate_worker_payload(payload, context, candidate, gpu, "attempt")
        self.assertEqual(record["execution"]["candidate_commit"], "a" * 40)
        payload["latency_ms"] = float("nan")
        with self.assertRaises(ValueError):
            _validate_worker_payload(payload, context, candidate, gpu, "attempt")


if __name__ == "__main__":
    unittest.main()

