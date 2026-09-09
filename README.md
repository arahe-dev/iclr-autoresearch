# icrl-autoresearch

Evidence-first performance research infrastructure for the exact Phase-BDH Arm-A path on an RTX PRO 6000 (`sm_120`). The repository is deliberately infrastructure-only in this initial import: it does not invent, edit, or run a new model/operator variant.

## Frozen contract

The canonical source is preserved byte-for-byte at [vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py](vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py). Its SHA-256 and attachment provenance are recorded in [SOURCE_MANIFEST.json](vendor/canonical/SOURCE_MANIFEST.json). The immutable Arm-A contract is in [configs/frozen_contract.json](configs/frozen_contract.json): exact Q=K, strict-exclusive same-document attention, shared V, document resets, document-local RoPE, Cartesian Arm-A coordinator, dense writer, frozen corpus/replay, optimizer, loss, and training semantics.

The target is 90,000 real input tokens/s, equivalent to 1,456.356 ms per exact B64 update. The imported validated baseline is experiment `0000`: 3,303.470 ms/B64, 39,673 derived real tok/s (reported as approximately 39.7k tok/s). It is the initial champion.

## Imported evidence

The machine-readable ledger is [evidence/history.jsonl](evidence/history.jsonl). It includes the validated control, prior positive component/oracle evidence, rejected tile/resource evidence, compiler/OOM failures, the pre-training dress-rehearsal failure, and the new native-read negative result as experiment `0001`.

Experiment `0001` records the exact `[B,T,H,K]` plus causal-prefix candidate: FP32 gradients passed, but the real B32 read gate measured 199.09 ms versus 160.34 ms control (0.805x) and 29.1 versus 16.3 GiB. It is `REVERTED_AT_READ_GATE`; full B64 was intentionally not run.

## Generation-0 screen

[configs/generation_0_screening.json](configs/generation_0_screening.json) and `python scripts/run_doe.py` define the one-factor-at-a-time screen for exact attention/Q=K backward, native SM120 GEMMs, Dy+ReLU epilogues/layout, E GEMM, and checkpoint policy. All rows are marked `PLANNED_NO_EXECUTION`; `--execute` is fail-closed. No approximate operator or architecture change is included.

The reusable infrastructure lives under `src/icrl_autoresearch/`:

- `oracle.py`: independent strict-exclusive tied-QK reference and VJP.
- `benchmarks.py`: B32 and exact B32+B32 B64 fixtures and measurement summaries.
- `profiler.py`: explicit profiler plans; execution requires `--execute`.
- `doe.py`: Generation-0 plan generation without model execution.
- `decisions.py`: declared keep/revert gates, including the 1.10x read gate and 90k target.
- `git_ops.py`: candidate/champion branch naming and explicit promotion plans.

## Safe checks

```powershell
python scripts/validate_repo.py
python scripts/import_evidence.py
python scripts/benchmark.py
python scripts/run_doe.py
python scripts/decide.py experiments/0001_native_read_reverted/report.json
```

The first command is the repository integrity check. No CUDA runtime is needed for these checks; CUDA benchmarks and profiling are intentionally not launched by this scaffold.

