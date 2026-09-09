# Generation-0 result ledger

The eventual `generation0.jsonl` file is append-only: one JSON object per planned run, keyed by `run_id`. Results may be appended only after the independent exact oracle, same-document reset, and tied-QK backward gates pass. The append helper rejects duplicate run IDs and non-oracle-passing records.

The current repository contains no screening measurements. This is intentional: Generation-0 is ready for explicit execution from commit `908b0b1`, but no GPU command has been launched and the champion has not been modified.

Use `scripts/run_generation0.py --execute` only with a separate `codex/experiment/*` or `codex/generation0/*` candidate worktree and a command that emits one oracle-passing JSON result per slot. The harness checks SM120, refuses champion branches, and appends records only after validation.

For the target machine, `scripts/run_generation0_sm120.py --execute` additionally requires the RTX PRO 6000/SM120 device and a positive `peak_GiB` below device capacity, streams the full candidate transcript to `results/generation0_console.log`, and stops immediately on any failed oracle, memory, process, or result-contract gate.
