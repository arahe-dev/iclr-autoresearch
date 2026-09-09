# Generation-0 result ledger

The eventual `generation0.jsonl` file is append-only: one JSON object per planned run, keyed by `run_id`. Results may be appended only after the independent exact oracle, same-document reset, and tied-QK backward gates pass. The append helper rejects duplicate run IDs and non-oracle-passing records.

The current repository contains no screening measurements. This is intentional: Generation-0 is a design-only handoff from commit `908b0b1`, and the champion has not been modified.
