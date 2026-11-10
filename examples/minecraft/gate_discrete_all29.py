"""Gate all-29 fine-tuning on offline learning and crafting evidence."""

import argparse
import json
import re
from pathlib import Path

from vla_foundry.data.minecraft.actions import D29_NAMES

LOSS_PATTERN = re.compile(r"avg_loss=([0-9]+(?:\.[0-9]+)?)")
CRAFT_ACTIONS = D29_NAMES[18:26]


def build_report(evaluation, policy, audit, training_log):
    learned = evaluation["cases"]["policy"]["chunk"]
    random_prior = evaluation["cases"]["random_train_prior"]["chunk"]
    shuffled = evaluation["cases"]["shuffled_instructions"]["chunk"]
    black = evaluation["cases"]["black_images"]["chunk"]
    losses = [float(value) for value in LOSS_PATTERN.findall(training_log)]
    craft_f1 = {name: learned["per_action_f1"][name] for name in CRAFT_ACTIONS}
    craft_macro_f1 = sum(craft_f1.values()) / len(craft_f1)
    train_counts = audit["splits"]["train"]["class_counts"]
    macro_f1 = learned["supported_macro_f1"]
    checks = {
        "all_29_actions_exported": policy["supported_ids"] == list(range(29)),
        "all_29_actions_have_training_targets": all(train_counts[name] > 0 for name in D29_NAMES),
        "validation_loss_decreased": len(losses) >= 2 and min(losses) < losses[0],
        "beats_random_prior_by_1_25x": macro_f1 >= 1.25 * random_prior["supported_macro_f1"],
        "craft_macro_f1_at_least_0_05": craft_macro_f1 >= 0.05,
        "at_least_6_craft_actions_learned": sum(value > 0.02 for value in craft_f1.values()) >= 6,
        "beats_shuffled_instructions": macro_f1 > shuffled["supported_macro_f1"],
        "beats_black_images": macro_f1 > black["supported_macro_f1"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "validation_losses": losses,
        "macro_f1": macro_f1,
        "random_prior_macro_f1": random_prior["supported_macro_f1"],
        "craft_macro_f1": craft_macro_f1,
        "craft_per_action_f1": craft_f1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.evaluation.read_text()),
        json.loads(args.policy.read_text()),
        json.loads(args.audit.read_text()),
        args.training_log.read_text(),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("All-29 policy did not pass the offline quality gate")


if __name__ == "__main__":
    main()
