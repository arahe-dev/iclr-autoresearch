from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "generation0_candidate_runner.py"


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


@unittest.skipUnless(
    _torch_available(),
    "TorchDynamo regression test requires PyTorch",
)
class CandidateRunnerLoaderTests(unittest.TestCase):
    def test_canonical_module_is_importable_during_checkpoint_compile(self) -> None:
        import torch
        from torch import nn
        from torch.utils.checkpoint import checkpoint

        module_name = "generation0_candidate_runner_regression"
        spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = runner
        try:
            spec.loader.exec_module(runner)

            canonical_name = "icrl_canonical_0000_definitions"
            canonical = importlib.import_module(canonical_name)
            self.assertIs(canonical, sys.modules[canonical_name])
            self.assertIs(canonical.__dict__, runner.CANONICAL)

            class CheckpointedCanonical(nn.Module):
                def forward(self, q: torch.Tensor, pos: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
                    def body(value: torch.Tensor) -> torch.Tensor:
                        return runner.CANONICAL["rope_bhtk"](value, pos, freq)

                    return checkpoint(body, q, use_reentrant=False)

            model = torch.compile(
                CheckpointedCanonical(),
                backend="eager",
                dynamic=False,
                fullgraph=True,
            )
            q = torch.randn(1, 1, 2, 4, requires_grad=True)
            pos = torch.tensor([[0, 1]], dtype=torch.int32)
            freq = runner.CANONICAL["pair_freq"](4, q.device)
            output = model(q, pos, freq)
            self.assertEqual(tuple(output.shape), tuple(q.shape))
        finally:
            sys.modules.pop(module_name, None)
            sys.modules.pop("icrl_canonical_0000_definitions", None)


if __name__ == "__main__":
    unittest.main()
