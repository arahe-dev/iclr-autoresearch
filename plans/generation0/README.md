# Generation-0 exact-semantics screen

This directory is the explicit-execution handoff from commit `908b0b1`. It uses an `L8(2^7)` orthogonal array plus an eight-arm full foldover over seven binary execution factors:

1. native SM120 full/symmetric QK attention;
2. native tied-QK backward VJP;
3. native SM120 Dx GEMM;
4. native SM120 Dy GEMM;
5. native SM120 E GEMM;
6. exact Dy→ReLU epilogue/layout;
7. checkpoint policy, with the experiment-0000 low level fixed to exact Gram-only SAC and the candidate high level set to selective exact projection-only recomputation.

The low level of every factor is the experiment-`0000` control. In particular, G0-00 uses the canonical `gram_only_sac` checkpoint policy, not a generic `whole_level` label. The high levels change only execution representation. The killed prefix-read, chunked-state, rejected-tile, and pre-training dtype-failure branches are excluded from the factor catalog.

`matrix.json` contains the factor priors, 8 base L8 arms, 8 bitwise-complement foldover arms, de-aliased main-effect metadata, gates, excluded branches, and per-arm patch specifications. `run_order.json` contains two deterministic randomized blocks with 52 planned B32 slots: each block has all 16 treatment arms plus anchor/interleaved controls for thermal drift. `arm_patch_specs.json` is the patch-only view for handoff to a separate candidate branch.

The objective is exact B32 F+B latency: approximately 1,655 ms for the current control and approximately 728 ms at 90k real tok/s. Every arm requires the independent oracle, same-document reset, and tied-QK backward checks before timing. `G0-00` is statically anchored to the byte-hashed canonical source and cannot carry patch operations.

The plan is executable only through `scripts/run_generation0.py --execute`, which requires a separate `codex/experiment/*` or `codex/generation0/*` worktree, an SM120 GPU, and a user-supplied candidate command. The harness refuses the champion branch/workspace and appends only oracle-passing JSONL records.
