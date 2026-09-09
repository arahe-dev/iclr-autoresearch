# Generation-0 exact-semantics screen

This directory is the design-only handoff from commit `908b0b1`. It uses an `L8(2^7)` orthogonal array over seven binary execution factors:

1. native SM120 full/symmetric QK attention;
2. native tied-QK backward VJP;
3. native SM120 Dx GEMM;
4. native SM120 Dy GEMM;
5. native SM120 E GEMM;
6. exact Dy→ReLU epilogue/layout;
7. selective exact projection-only checkpointing.

The low level of every factor is the experiment-`0000` control. The high levels change only execution representation. The killed prefix-read, chunked-state, rejected-tile, and pre-training dtype-failure branches are excluded from the factor catalog.

`matrix.json` contains the factor priors, L8 arms, interaction aliases, gates, excluded branches, and per-arm patch specifications. `run_order.json` contains two deterministic randomized blocks with 26 planned B32 runs: each block has all eight arms, the control row, and interleaved/anchor controls for thermal drift. `arm_patch_specs.json` is the patch-only view for handoff to a future candidate branch.

The objective is exact B32 F+B latency: approximately 1,655 ms for the current control and approximately 728 ms at 90k real tok/s. Every arm requires the independent oracle, same-document reset, and tied-QK backward checks before timing.
