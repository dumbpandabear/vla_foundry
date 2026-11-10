"""Validate and summarize completed categorical D29 Craftium rollouts."""

import argparse
import json
from collections import Counter
from pathlib import Path

from vla_foundry.data.minecraft.actions import D29_NAMES


def summarize(directories: list[Path], expected_steps: int) -> dict:
    rollouts = []
    total_actions = Counter()
    for directory in directories:
        result = json.loads((directory / "result.json").read_text())
        actions = [json.loads(line) for line in (directory / "actions.jsonl").read_text().splitlines()]
        if result["steps"] != expected_steps or len(actions) != expected_steps:
            raise ValueError(f"Incomplete rollout {directory}: result={result['steps']} actions={len(actions)}")
        if not result["nonblank_observations"]:
            raise ValueError(f"Blank observations in {directory}")
        if result["projection"] != "native categorical D29" or result["action_space"] != "discrete_29":
            raise ValueError(f"Unexpected action integration in {directory}")
        counts = Counter(row["name"] for row in actions)
        unknown = set(counts) - set(D29_NAMES)
        if unknown:
            raise ValueError(f"Unknown D29 actions in {directory}: {sorted(unknown)}")
        total_actions.update(counts)
        rollouts.append(
            {
                "directory": str(directory),
                "seed": result["seed"],
                "steps": result["steps"],
                "total_reward": result["total_reward"],
                "unique_actions": len(counts),
                "action_counts": dict(counts),
            }
        )
    return {
        "verified": True,
        "rollout_count": len(rollouts),
        "total_steps": sum(row["steps"] for row in rollouts),
        "total_reward": sum(row["total_reward"] for row in rollouts),
        "unique_actions": len(total_actions),
        "action_counts": dict(total_actions),
        "rollouts": rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_steps <= 0:
        parser.error("--expected-steps must be positive")
    report = summarize(args.rollouts, args.expected_steps)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
