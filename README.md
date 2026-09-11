# icrl-autoresearch

Evidence-first performance research infrastructure for the exact Phase-BDH Arm-A path on the NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`). The C15 supervisor and candidate runner support explicitly authorized screening of the frozen execution factors.

## Frozen contract

The canonical source is preserved byte-for-byte at [vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py](vendor/canonical/icrl_reintegrated_bdh_baseline_v1.py). Its SHA-256 and attachment provenance are recorded in [SOURCE_MANIFEST.json](vendor/canonical/SOURCE_MANIFEST.json). The immutable Arm-A contract is in [configs/frozen_contract.json](configs/frozen_contract.json): exact Q=K, strict-exclusive same-document attention, shared V, document resets, document-local RoPE, Cartesian Arm-A coordinator, dense writer, frozen corpus/replay, optimizer, loss, and training semantics.

The target is 90,000 real input tokens/s, equivalent to 1,456.356 ms per exact B64 update. The imported validated baseline is experiment `0000`: 3,303.470 ms/B64, 39,673 derived real tok/s (reported as approximately 39.7k tok/s). It is the initial champion.

## Imported evidence

The machine-readable ledger is [evidence/history.jsonl](evidence/history.jsonl). It includes the validated control, prior positive component/oracle evidence, rejected tile/resource evidence, compiler/OOM failures, the pre-training dress-rehearsal failure, and the new native-read negative result as experiment `0001`.

Experiment `0001` records the exact `[B,T,H,K]` plus causal-prefix candidate: FP32 gradients passed, but the real B32 read gate measured 199.09 ms versus 160.34 ms control (0.805x) and 29.1 versus 16.3 GiB. It is `REVERTED_AT_READ_GATE`; full B64 was intentionally not run.

## Generation-0 screen

[configs/generation_0_screening.json](configs/generation_0_screening.json) and [plans/generation0/matrix.json](plans/generation0/matrix.json) define `G0-L8-SM120-EXACT-B32-C15`: the original L8 and foldover treatments with structurally infeasible `G0-F05` excluded after the observed B32 warmup OOM. There are 15 treatment arms and 50 slots across two blocks. All 20 original CONTROL slots, randomized order, run IDs and within-block positions are retained; only the two F05 occurrences are removed. The intercept plus seven main-effect columns retain rank 8. The missing cell removes exact orthogonality and main-effect/two-factor de-aliasing; estimates are conditional on the feasible cells and model. No F05 measurement is fabricated or imputed. The exact experiment-0000 `gram_only_sac` control and B32 F+B science contract remain unchanged.

The plan is `READY_FOR_EXPLICIT_EXECUTION`, but execution remains fail-closed: `scripts/run_generation0.py --execute` requires a separate candidate worktree, a user-supplied GPU command, an SM120 GPU, and oracle-passing JSON output. The harness refuses all champion branches and never promotes or mutates a champion. The plan excludes every branch already killed in the evidence ledger and contains no approximate operator or architecture change. See [run_order.json](plans/generation0/run_order.json), [arm_patch_specs.json](plans/generation0/arm_patch_specs.json), and [plans/generation0/README.md](plans/generation0/README.md).

The reusable infrastructure lives under `src/icrl_autoresearch/`:

- `oracle.py`: independent strict-exclusive tied-QK reference and VJP.
- `benchmarks.py`: B32 and exact B32+B32 B64 fixtures and measurement summaries.
- `profiler.py`: explicit profiler plans; execution requires `--execute`.
- `doe.py`: Generation-0 plan generation without model execution.
- `decisions.py`: declared keep/revert gates, including the 1.10x read gate and 90k target.
- `git_ops.py`: candidate/champion branch naming and explicit promotion plans.
- `generation0.py`: constrained C15 matrix, immutable control verification, missing-cell caveat, original randomized order, and per-arm patch specs.
- `results.py` and `provenance.py`: strict version-2 append-only admission with full source/candidate and harness commits, tracked-file code hashes, frozen plan/config/contract identity, corpus manifest/file hashes, actual CUDA GPU identity, timing samples/boundaries, memory peaks, and validation/admission status.
- `gpu_harness.py`: fail-closed SM120 supervisor; it owns the campaign lock, physical-GPU lock, ordered resume state, child process groups, preflight, attempt log, and ledger admission. It never imports PyTorch.
- `processes.py` and `gpu.py`: durable file locks, process-tree cleanup, timeout/interrupt handling, torch-free NVIDIA discovery, and UUID-bound worker environments.
- `scripts/run_generation0_sm120.py`: stable target-host entrypoint delegating to the same supervisor used by `run_generation0.py`.
- `scripts/colab_generation0.py`: one self-contained Colab bootstrap pinned to a full harness SHA. It creates isolated planning/candidate worktrees, uses a short-lived askpass helper when authentication is needed, and persists campaign state, attempts, logs, and artifacts on Drive.

The self-contained `scripts/colab_generation0.py` defaults to `EXECUTE=False`: pasting it prints preparation information without mounting Drive, cloning, probing a GPU or launching a worker. The parent must first commit this implementation. For a later manually authorized run, set `ICRL_G0_HARNESS_COMMIT` to that full 40-character SHA before pasting and set `EXECUTE=True`. The cell validates that the pinned checkout emits the C15 plan and uses a new `iclr-g0-c15-<sha>` campaign. The original `iclr-g0-0ce16e9922b2` Drive evidence campaign is immutable and explicitly refused as an output target. GPU UUID/index selection remains external, and retries remain explicit. No campaign was run as part of this implementation.

Each accepted ledger record is self-contained; worker artifacts retain `PENDING_SUPERVISOR` until the parent verifies successful worker exit and GPU release. Source and corpus hashes are checked outside timed regions. Recorded CUDA-event intervals include forward, masked cross-entropy and backward; transfer, optimizer steps, zeroing gradients and cleanup remain outside those intervals. Both warmups and all five measured repetitions include wall/monotonic boundaries and allocation peaks. A legacy result cannot be mixed into the C15 ledger.

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

The launcher writes the append-only ledger to `results/generation0.jsonl` and tees all runner output to `results/generation0_console.log`. It requires the exact NVIDIA RTX PRO 6000 Blackwell Server Edition name, SM120 and at least 90 GiB VRAM, and leaves the champion immutable. Each preflight and benchmark attempt gets a fresh worker process and must wait for physical GPU release on success, failure or interrupt. The next slot is not admitted until the worker exits, the GPU has returned to its baseline, and the result has passed every oracle/protocol/provenance gate. Accepted provenance includes the immutable champion commit and branch alongside full candidate/harness commits and source hashes. A failure writes an attempt-scoped artifact and sticky state; resume requires `--retry-failed` explicitly and never skips the failed ordered slot.
