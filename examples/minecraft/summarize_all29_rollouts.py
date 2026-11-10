"""Verify isolated Craftium rollouts for all eight semantic craft actions."""

import argparse
import json
from pathlib import Path

from vla_foundry.data.minecraft.actions import D29_NAMES

EXPECTED = {
    "craft_planks_from_log": "craft_planks",
    "craft_crafting_table_from_planks": "craft_crafting_table",
    "craft_stick_from_crafting_table": "craft_stick",
    "craft_wooden_pickaxe_from_stick": "craft_wooden_pickaxe",
    "craft_stone_pickaxe_from_cobblestone": "craft_stone_pickaxe",
    "craft_furnace_from_iron_ore": "craft_furnace",
    "craft_iron_ingot_from_furnace": "smelt_iron_ingot",
    "craft_iron_pickaxe_from_iron_ingot": "craft_iron_pickaxe",
}


def summarize(directories):
    rows = []
    for directory in directories:
        result = json.loads((directory / "result.json").read_text())
        task = result["task"]
        expected = EXPECTED[task]
        actions = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
        names = [row["name"] for row in actions]
        unknown = set(names) - set(D29_NAMES)
        if unknown:
            raise ValueError(f"Unknown actions in {directory}: {sorted(unknown)}")
        checks = {
            "nonblank_observations": result["nonblank_observations"],
            "native_d29_integration": result["projection"] == "native categorical D29",
            "expected_action_emitted": expected in names,
            "positive_task_reward": result["total_reward"] > 0,
        }
        rows.append(
            {
                "task": task,
                "expected_action": expected,
                "steps": result["steps"],
                "reward": result["total_reward"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    missing = set(EXPECTED) - {row["task"] for row in rows}
    if missing:
        raise ValueError(f"Missing Craftium tasks: {sorted(missing)}")
    return {"verified": all(row["passed"] for row in rows), "tasks": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.rollouts)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["verified"]:
        raise SystemExit("All-29 Craftium verification failed")


if __name__ == "__main__":
    main()
