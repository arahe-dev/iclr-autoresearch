from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icrl_autoresearch.generation0 import build_plan


def main() -> None:
    plan = build_plan()
    output = Path(__file__).resolve().parents[1] / "plans" / "generation0"
    output.mkdir(parents=True, exist_ok=True)
    (output / "matrix.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "run_order.json").write_text(json.dumps({"plan_id": plan["plan_id"], "runs": plan["run_order"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "arm_patch_specs.json").write_text(json.dumps({"plan_id": plan["plan_id"], "arms": [{"arm_id": a["arm_id"], "patch_spec": a["patch_spec"]} for a in plan["arms"]]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "GENERATED_SPEC_ONLY", "plan_id": plan["plan_id"], "output": str(output), "arms": len(plan["arms"]), "runs": len(plan["run_order"]), "execution_allowed": plan["execution_allowed"]}, indent=2))


if __name__ == "__main__":
    main()
