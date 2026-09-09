from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.generation0 import build_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit the frozen Generation-0 L8 screening design")
    parser.add_argument("--output", type=Path, help="optional path for the full plan JSON")
    parser.add_argument("--execute", action="store_true", help="always rejected in the infrastructure-only scaffold")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("Generation-0 execution is disabled: this scaffold does not run new model changes.")

    payload = json.dumps(build_plan(), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
