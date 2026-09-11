"""Exact target identity checks without importing torch or accessing a GPU."""

import ast
from copy import deepcopy
from dataclasses import replace
import os
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

from test_harness_supervisor import ROOT, worker_fixture
from icrl_autoresearch.contract import load_contract
from icrl_autoresearch.gpu import EXPECTED_GPU_NAME, GpuSnapshot, NvidiaSmiError, select_target_gpu
from icrl_autoresearch.gpu_harness import _validate_preflight
from icrl_autoresearch.results import validate_gpu


WRONG_NAMES = (
    "RTX PRO 6000",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    EXPECTED_GPU_NAME.lower(),
    "prefix " + EXPECTED_GPU_NAME,
    EXPECTED_GPU_NAME + " suffix",
)


class GpuIdentityTests(unittest.TestCase):
    def setUp(self):
        self.payload, _, self.candidate, self.gpu, _, self.identity, _ = worker_fixture()
        self.target = GpuSnapshot(self.gpu["uuid"], EXPECTED_GPU_NAME, 94 * 1024, 0, 94 * 1024, "synthetic")

    def test_inventory_requires_exact_frozen_name_for_all_selection_modes(self):
        self.assertEqual(EXPECTED_GPU_NAME, load_contract()["target"]["gpu"])
        for name in (EXPECTED_GPU_NAME, *WRONG_NAMES):
            for selection in ({}, {"gpu_uuid": self.target.uuid}, {"gpu_index": 0}):
                with self.subTest(name=name, selection=selection), patch("icrl_autoresearch.gpu.list_gpus", return_value=[replace(self.target, name=name)]):
                    if name == EXPECTED_GPU_NAME:
                        self.assertEqual(select_target_gpu(**selection), self.target)
                    else:
                        with self.assertRaises(NvidiaSmiError):
                            select_target_gpu(**selection)
        with patch("icrl_autoresearch.gpu.list_gpus", return_value=[replace(self.target, total_mib=89 * 1024)]):
            with self.assertRaises(NvidiaSmiError):
                select_target_gpu()

    def test_external_uuid_and_index_selection_remain_available(self):
        second = replace(self.target, uuid="GPU-aaaaaaaa-1234-1234-1234-123456789abc")
        with patch("icrl_autoresearch.gpu.list_gpus", return_value=[self.target, second]):
            with self.assertRaises(NvidiaSmiError):
                select_target_gpu()
            self.assertEqual(select_target_gpu(gpu_uuid=second.uuid), second)
            self.assertEqual(select_target_gpu(gpu_index=1), second)
        with patch("icrl_autoresearch.gpu.list_gpus", return_value=[replace(second, name=WRONG_NAMES[1]), self.target]):
            self.assertEqual(select_target_gpu(), self.target)

    def test_result_and_preflight_reject_other_names_and_keep_identity_gates(self):
        runtime = self.payload["execution"]["gpu"]
        preflight = {
            **runtime, "status": "PREFLIGHT_PASS", "plan_id": self.payload["plan_id"],
            "source_commit": self.candidate["candidate_commit"],
            "provenance": self.identity["provenance"], "corpus": self.identity["provenance"]["corpus"],
            "benchmark_limits": {"microbatch": 32, "warmups": 2, "timed_repetitions": 5},
        }
        validate_gpu(runtime, self.gpu)
        _validate_preflight(preflight, self.gpu, preflight["corpus"], self.candidate["candidate_commit"])
        for key, value in [*(('device', name) for name in WRONG_NAMES),
                           ("sm", "sm_90"), ("vram_GiB", 89.0), ("gpu_uuid", "GPU-unavailable"),
                           ("gpu_uuid", "GPU-aaaaaaaa-1234-1234-1234-123456789abc"),
                           ("torch", ""), ("cuda", None)]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_gpu({**runtime, key: value}, self.gpu)
                with self.assertRaises(ValueError):
                    _validate_preflight({**preflight, key: value}, self.gpu, preflight["corpus"], self.candidate["candidate_commit"])
        for field in ("champion_commit", "champion_branch"):
            for value in (None, "tampered"):
                bad = deepcopy(preflight)
                if value is None:
                    bad["provenance"]["frozen"].pop(field)
                else:
                    bad["provenance"]["frozen"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _validate_preflight(bad, self.gpu, bad["corpus"], self.candidate["candidate_commit"])

    def test_runtime_capacity_uses_idle_free_memory_and_preserves_identity_gates(self):
        locked_snapshot = GpuSnapshot(
            self.target.uuid, EXPECTED_GPU_NAME, 97887, 0, 97251, "synthetic"
        )
        locked = locked_snapshot.as_dict()
        runtime = {
            "device": EXPECTED_GPU_NAME,
            "sm": "sm_120",
            "gpu_uuid": self.target.uuid,
            "torch": "synthetic-torch",
            "cuda": "synthetic-cuda",
            "vram_GiB": 94.97076416015625,
        }

        self.assertEqual(locked["memory_total_MiB"], 97887)
        self.assertEqual(locked["memory_free_MiB"], 97251)
        self.assertEqual(locked["vram_GiB"], 97251 / 1024)
        validate_gpu(runtime, locked)

        # The physical nvidia-smi total is intentionally not the worker's
        # CUDA-addressable capacity and must not be accepted as a match.
        with self.assertRaises(ValueError):
            validate_gpu({**runtime}, {**locked, "vram_GiB": 97887 / 1024})

        for key, value in (
            ("device", "wrong GPU"),
            ("gpu_uuid", "GPU-aaaaaaaa-1234-1234-1234-123456789abc"),
            ("sm", "sm_90"),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_gpu({**runtime, key: value}, locked)

    def test_worker_uses_actual_cuda_properties_with_exact_name(self):
        # Compile the real identity functions in isolation so this test has no
        # PyTorch dependency and cannot initialize CUDA or run the model loader.
        source = ROOT / "scripts/generation0_candidate_runner.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if (
            isinstance(node, ast.FunctionDef) and node.name in {"_target_gpu", "_required_env"}
        ) or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"EXPECTED_SM", "MIN_VRAM_GIB"} for target in node.targets
        ))]
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = EXPECTED_GPU_NAME
        cuda.get_device_capability.return_value = (12, 0)
        props = SimpleNamespace(total_memory=94 * 2**30, uuid=self.target.uuid)
        cuda.get_device_properties.return_value = props
        namespace = {"Any": Any, "os": os, "EXPECTED_GPU_NAME": EXPECTED_GPU_NAME,
                     "torch": SimpleNamespace(cuda=cuda, __version__="synthetic-torch", version=SimpleNamespace(cuda="synthetic-cuda"))}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
        target_gpu = namespace["_target_gpu"]
        with patch.dict(os.environ, {"ICRL_G0_GPU_UUID": self.target.uuid}):
            actual = target_gpu()
            self.assertEqual(actual["gpu_uuid"], props.uuid)
            self.assertEqual(actual["torch"], "synthetic-torch")
            self.assertEqual(actual["cuda"], "synthetic-cuda")
            for name in WRONG_NAMES:
                cuda.get_device_name.return_value = name
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    target_gpu()
            cuda.get_device_name.return_value = EXPECTED_GPU_NAME
            cuda.get_device_capability.return_value = (9, 0)
            with self.assertRaises(RuntimeError):
                target_gpu()
            cuda.get_device_capability.return_value = (12, 0)
            props.total_memory = 89 * 2**30
            with self.assertRaises(RuntimeError):
                target_gpu()
            props.total_memory = 94 * 2**30
            props.uuid = "GPU-aaaaaaaa-1234-1234-1234-123456789abc"
            with self.assertRaises(RuntimeError):
                target_gpu()


if __name__ == "__main__":
    unittest.main()
