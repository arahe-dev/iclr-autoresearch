# Generation-0 result ledger

The eventual `generation0.jsonl` file is append-only: one JSON object per planned run, keyed by `run_id`. Results may be appended only after the independent exact oracle, same-document reset, and tied-QK backward gates pass. The append helper rejects duplicate run IDs and non-oracle-passing records.

The current repository contains no screening measurements. This is intentional: Generation-0 is ready for explicit execution from commit `908b0b1`, but no GPU command has been launched and the champion has not been modified.

Use `scripts/run_generation0.py --execute` only with a separate `codex/experiment/*` or `codex/generation0/*` candidate worktree and a command that emits one oracle-passing JSON result per slot. The harness checks SM120, refuses champion branches, and appends records only after validation.
