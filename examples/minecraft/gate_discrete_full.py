"""Gate full-corpus D29 Craftium rollouts on offline learning evidence."""

import argparse
import json
import re
from pathlib import Path

LOSS_PATTERN = re.compile(r"avg_loss=([0-9]+(?:\.[0-9]+)?)")


def build_report(evaluation: dict, policy: dict, training_log: str) -> dict:
    cases = evaluation["cases"]
    learned = cases["policy"]["chunk"]
    random_prior = cases["random_train_prior"]["chunk"]
    shuffled = cases["shuffled_instructions"]["chunk"]
    black = cases["black_images"]["chunk"]
    losses = [float(value) for value in LOSS_PATTERN.findall(training_log)]
    macro_f1 = learned["supported_macro_f1"]
    checks = {
        "validation_loss_decreased": len(losses) >= 2 and min(losses) <= losses[0] - 0.02,
        "supported_macro_f1_at_least_0_15": macro_f1 >= 0.15,
        "beats_random_prior_by_1_5x": macro_f1 >= 1.5 * random_prior["supported_macro_f1"],
        "camera_macro_f1_at_least_0_05": learned["camera_macro_f1"] >= 0.05,
        "beats_shuffled_instructions": macro_f1 >= shuffled["supported_macro_f1"] + 0.01,
        "beats_black_images": macro_f1 >= black["supported_macro_f1"] + 0.01,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "validation_losses": losses,
        "selected_validation_loss": policy["validation_loss"],
        "selected_checkpoint": policy["checkpoint"],
        "policy_supported_macro_f1": macro_f1,
        "random_prior_supported_macro_f1": random_prior["supported_macro_f1"],
        "shuffled_instruction_supported_macro_f1": shuffled["supported_macro_f1"],
        "black_image_supported_macro_f1": black["supported_macro_f1"],
        "camera_macro_f1": learned["camera_macro_f1"],
        "actions_with_f1_above_0_02": sum(value > 0.02 for value in learned["per_action_f1"].values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.evaluation.read_text()),
        json.loads(args.policy.read_text()),
        args.training_log.read_text(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("Full D29 policy did not pass the rollout quality gate")


if __name__ == "__main__":
    main()
