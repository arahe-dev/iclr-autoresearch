from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.results import append_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one oracle-passing Generation-0 result")
    parser.add_argument("record", type=Path, help="JSON object containing one result")
    parser.add_argument("--ledger", type=Path, default=Path("results/generation0.jsonl"))
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    append_result(args.ledger, record)
    print(json.dumps({"status": "APPENDED", "run_id": record["run_id"], "ledger": str(args.ledger)}, indent=2))


if __name__ == "__main__":
    main()
