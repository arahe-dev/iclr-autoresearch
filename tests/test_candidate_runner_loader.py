from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "generation0_candidate_runner.py"


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


class ScriptImportTests(unittest.TestCase):
    def test_cprofile_import_with_runner_script_directory_first(self) -> None:
        # A fresh process matters: importing cProfile earlier in the test
        # process would hide the collision seen by the standalone GPU runner.
        probe = textwrap.dedent("""
            from pathlib import Path
            import sys
            import sysconfig
            sys.path.insert(0, sys.argv[1])
            import cProfile
            import profile
            assert Path(profile.__file__).resolve() == (
                Path(sysconfig.get_path("stdlib")) / "profile.py"
            ).resolve(), profile.__file__
            assert callable(profile.run)
            cProfile.Profile().runcall(lambda: sum(range(4)))
        """)
        result = subprocess.run(
            [sys.executable, "-c", probe, str(RUNNER_PATH.parent)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


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
