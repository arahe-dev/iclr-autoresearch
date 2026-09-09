from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the tracked machine-readable evidence ledger")
    parser.add_argument("--ledger", type=Path, default=Path("evidence/history.jsonl"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"status": "PASS", "records": len(rows), "ids": [r["evidence_id"] for r in rows]}, indent=2))


if __name__ == "__main__":
    main()

