from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.benchmarks import B32, B64


if __name__ == "__main__":
    # Fixture declaration only. A benchmark callable must be supplied by a
    # later, explicitly authorized run; this command never edits model code.
    print(json.dumps({"status": "PLANNED_NO_EXECUTION", "benchmarks": [B32.__dict__, B64.__dict__]}, indent=2))

