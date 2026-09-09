# Evidence ledger

`history.jsonl` is append-only machine-readable provenance. Each line is one measured or explicitly failed run, with source references, correctness status, metrics, decision, and the search space killed by a negative result.

Rules:

- A component timing is never promoted to an end-to-end throughput claim.
- A correctness pass does not imply a performance keep.
- Compile/OOM failures are recorded as failures before timing, not silently omitted.
- A candidate that fails a declared gate is reverted before later stages.
- New records must retain the frozen Arm-A contract and real-token accounting.

