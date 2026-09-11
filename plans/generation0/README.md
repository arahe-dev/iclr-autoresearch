# Generation-0 exact-semantics screen

This directory is the `G0-L8-SM120-EXACT-B32-C15` handoff, anchored to champion `908b0b1438ba038d319adf97787aac08f213b590`. It retains the original L8 and foldover definitions except structurally infeasible `G0-F05`, over seven binary execution factors:

1. native SM120 full/symmetric QK attention;
2. native tied-QK backward VJP;
3. native SM120 Dx GEMM;
4. native SM120 Dy GEMM;
5. native SM120 E GEMM;
6. exact Dy→ReLU epilogue/layout;
7. checkpoint policy, with the experiment-0000 low level fixed to exact Gram-only SAC and the candidate high level set to selective exact projection-only recomputation.

The low level of every factor is the experiment-`0000` control. In particular, G0-00 uses the canonical `gram_only_sac` checkpoint policy, not a generic `whole_level` label. The high levels change only execution representation. The killed prefix-read, chunked-state, rejected-tile, and pre-training dtype-failure branches are excluded from the factor catalog.

`matrix.json` contains the factor priors, 8 base L8 arms, 7 feasible original foldover arms, missing-cell caveat, gates, excluded cells/branches, and per-arm patch specifications. `run_order.json` contains the original randomized/interleaved order with only the two `G0-F05` occurrences removed: 30 treatment slots plus all 20 original CONTROL slots, totaling 50. Original run IDs and position covariates are preserved, including gaps at `G0-R05` and `G0-R40`. `arm_patch_specs.json` is the patch-only view for a separate candidate branch.

The design is constrained and rank-complete for the intercept plus seven main-effect columns (rank 8). It is no longer exactly orthogonal or dealiased from two-factor interactions. Fit the declared regression with original block/position covariates and retain the missing-cell and interaction uncertainty. The observed F05 warmup OOM is recorded only as an exclusion rationale referencing the untouched `iclr-g0-0ce16e9922b2` campaign; no F05 timing is imputed or admitted.

The objective is exact B32 F+B latency: approximately 1,655 ms for the current control and approximately 728 ms at 90k real tok/s. Every arm requires the independent oracle, same-document reset, and tied-QK backward checks before timing. `G0-00` is statically anchored to the byte-hashed canonical source and cannot carry patch operations.

The plan is executable only through `scripts/run_generation0.py --execute`, which requires a separate `codex/experiment/*` or `codex/generation0/*` worktree, an SM120 GPU, and a user-supplied candidate command. The harness refuses the champion branch/workspace and appends only oracle-passing JSONL records.
