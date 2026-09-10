# icrl-autoresearch

Evidence-first performance research infrastructure for the exact Phase-BDH Arm-A path on an RTX PRO 6000 (`sm_120`). The repository is deliberately infrastructure-only in this initial import: it does not invent, edit, or run a new model/operator variant.

## Frozen contract

The canonical source is preserved byte-for-byte at [vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py](vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py). Its SHA-256 and attachment provenance are recorded in [SOURCE_MANIFEST.json](vendor/canonical/SOURCE_MANIFEST.json). The immutable Arm-A contract is in [configs/frozen_contract.json](configs/frozen_contract.json): exact Q=K, strict-exclusive same-document attention, shared V, document resets, document-local RoPE, Cartesian Arm-A coordinator, dense writer, frozen corpus/replay, optimizer, loss, and training semantics.

The target is 90,000 real input tokens/s, equivalent to 1,456.356 ms per exact B64 update. The imported validated baseline is experiment `0000`: 3,303.470 ms/B64, 39,673 derived real tok/s (reported as approximately 39.7k tok/s). It is the initial champion.

## Imported evidence

The machine-readable ledger is [evidence/history.jsonl](evidence/history.jsonl). It includes the validated control, prior positive component/oracle evidence, rejected tile/resource evidence, compiler/OOM failures, the pre-training dress-rehearsal failure, and the new native-read negative result as experiment `0001`.

Experiment `0001` records the exact `[B,T,H,K]` plus causal-prefix candidate: FP32 gradients passed, but the real B32 read gate measured 199.09 ms versus 160.34 ms control (0.805x) and 29.1 versus 16.3 GiB. It is `REVERTED_AT_READ_GATE`; full B64 was intentionally not run.

## Generation-0 screen

[configs/generation_0_screening.json](configs/generation_0_screening.json) and [plans/generation0/matrix.json](plans/generation0/matrix.json) define the L8(2^7) Taguchi-style screen plus its 8-arm full foldover for exact full/symmetric QK, tied-QK backward, separate Dx/Dy/E native SM120 GEMMs, Dy+ReLU epilogues/layout, and checkpoint policy. The Generation-0 low checkpoint level is the exact experiment-0000 `gram_only_sac` policy. The objective is exact B32 F+B latency: approximately 1,655 ms for the current control and approximately 728 ms at 90k real tok/s. The plan has 16 unique treatment arms and 52 randomized/interleaved-control run slots across two blocks, with oracle correctness required before timing.

The plan is `READY_FOR_EXPLICIT_EXECUTION`, but execution remains fail-closed: `scripts/run_generation0.py --execute` requires a separate candidate worktree, a user-supplied GPU command, an SM120 GPU, and oracle-passing JSON output. The harness refuses all champion branches and never promotes or mutates a champion. The plan excludes every branch already killed in the evidence ledger and contains no approximate operator or architecture change. See [run_order.json](plans/generation0/run_order.json), [arm_patch_specs.json](plans/generation0/arm_patch_specs.json), and [plans/generation0/README.md](plans/generation0/README.md).

The reusable infrastructure lives under `src/icrl_autoresearch/`:

- `oracle.py`: independent strict-exclusive tied-QK reference and VJP.
- `benchmarks.py`: B32 and exact B32+B32 B64 fixtures and measurement summaries.
- `profiler.py`: explicit profiler plans; execution requires `--execute`.
- `doe.py`: Generation-0 plan generation without model execution.
- `decisions.py`: declared keep/revert gates, including the 1.10x read gate and 90k target.
- `git_ops.py`: candidate/champion branch naming and explicit promotion plans.
- `generation0.py`: L8 plus full-foldover matrix, control verification, predicted effects, interaction aliases, randomized run order, and per-arm patch specs.
- `results.py`: append-only oracle-gated JSONL result writer.
- `gpu_harness.py`: fail-closed SM120 supervisor; it owns the campaign lock, physical-GPU lock, ordered resume state, child process groups, preflight, attempt log, and ledger admission. It never imports PyTorch.
- `processes.py` and `gpu.py`: durable file locks, process-tree cleanup, timeout/interrupt handling, torch-free NVIDIA discovery, and UUID-bound worker environments.
- `scripts/run_generation0_sm120.py`: stable target-host entrypoint delegating to the same supervisor used by `run_generation0.py`.
- `scripts/colab_generation0.py`: one self-contained Colab bootstrap pinned to a full harness SHA. It creates isolated planning/candidate worktrees, uses a short-lived askpass helper when authentication is needed, and persists campaign state, attempts, logs, and artifacts on Drive.

For the Colab diagnostic repair, replace the entire old cell with the current `scripts/colab_generation0.py`. Fetching newer repository commits does not update Python functions already pasted into a notebook. The cell uses the notebook's Python interpreter for both the supervisor and candidate workers. It streams combined supervisor stdout/stderr into notebook output and appends it to `generation0_cell.log` in the existing Drive campaign directory, including failures before the harness opens `generation0_console.log`. Nonzero exits still stop fail-closed and include the last 200 output lines and transcript path in the notebook exception. The runtime harness remains pinned to `0ce16e9922b2766df24698c96bf5eac34d5eb2b4`; existing campaign evidence and explicit failed-slot retry requirements are preserved.

## Safe checks

```powershell
python scripts/validate_repo.py
python scripts/import_evidence.py
python scripts/benchmark.py
python scripts/run_doe.py
python scripts/decide.py experiments/0001_native_read_reverted/report.json
python scripts/generate_generation0.py
python scripts/run_generation0.py
python scripts/run_generation0_sm120.py
```

The first command is the repository integrity check. The plan-only commands do not launch CUDA. To execute, supply a separate candidate worktree and command, for example:

```powershell
python scripts/run_generation0.py --execute `
  --candidate-root C:\path\to\candidate-worktree `
  --command-template "python candidate_runner.py --run-id {run_id} --arm-id {arm_id}"
```

The command must emit one oracle-passing result JSON object per slot. The harness appends validated records to `results/generation0.jsonl`; it never modifies the champion branch.

On the exact target host, use the strict launcher for the full campaign. The candidate runner must implement the already-reviewed factor patch specs and emit a final JSON object containing `oracle.status=PASS`, `oracle.same_document_reset=PASS`, `oracle.tied_qk_backward=PASS`, positive `latency_ms` and `real_input_tok_s`, and a positive `peak_GiB` below device capacity. The launcher stops at the first failed oracle, memory, process, or contract gate and appends no invalid record:

```bash
python scripts/run_generation0_sm120.py --execute \
  --candidate-root /path/to/codex/experiment/generation0-sm120 \
  --command-template 'python candidate_runner.py --run-id {run_id} --arm-id {arm_id}'
```

The launcher writes the append-only ledger to `results/generation0.jsonl` and tees all runner output to `results/generation0_console.log`. It requires the exact RTX PRO 6000 SM120 environment and leaves the champion immutable. Each slot gets a fresh worker process; the next slot is not admitted until the worker exits, the GPU has returned to its baseline, and the result has passed every oracle/protocol/provenance gate. A failure writes an attempt-scoped artifact and sticky state; resume requires `--retry-failed` explicitly and never skips the failed ordered slot.
