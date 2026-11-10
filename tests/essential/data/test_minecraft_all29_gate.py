"""Check all-29 offline and native Craftium quality gates."""

import json

from examples.minecraft.gate_discrete_all29 import CRAFT_ACTIONS, build_report
from examples.minecraft.summarize_all29_rollouts import EXPECTED, summarize
from vla_foundry.data.minecraft.actions import D29_NAMES


def metric(macro_f1, craft_f1=0.1):
    per_action = {name: 0.08 for name in D29_NAMES}
    per_action.update({name: craft_f1 for name in CRAFT_ACTIONS})
    return {"supported_macro_f1": macro_f1, "per_action_f1": per_action}


def evaluation(policy=0.20):
    return {
        "cases": {
            "policy": {"chunk": metric(policy)},
            "random_train_prior": {"chunk": metric(0.05)},
            "shuffled_instructions": {"chunk": metric(0.12)},
            "black_images": {"chunk": metric(0.10)},
        }
    }


def audit(missing=None):
    counts = {name: 100 for name in D29_NAMES}
    if missing is not None:
        counts[missing] = 0
    return {"splits": {"train": {"class_counts": counts}}}


def test_all29_gate_accepts_learning_and_all_action_coverage():
    report = build_report(
        evaluation(),
        {"supported_ids": list(range(29))},
        audit(),
        "Validation after checkpoint 1: avg_loss=1.8\nValidation after checkpoint 2: avg_loss=1.5\n",
    )

    assert report["passed"]
    assert report["craft_macro_f1"] == 0.1


def test_all29_gate_rejects_missing_craft_targets():
    report = build_report(
        evaluation(),
        {"supported_ids": list(range(29))},
        audit("craft_furnace"),
        "avg_loss=1.8\navg_loss=1.5\n",
    )

    assert not report["passed"]
    assert not report["checks"]["all_29_actions_have_training_targets"]


def write_rollout(path, task, *, expected=True, reward=1.0):
    path.mkdir()
    (path / "result.json").write_text(
        json.dumps(
            {
                "task": task,
                "steps": 2,
                "total_reward": reward,
                "nonblank_observations": True,
                "projection": "native categorical D29",
            }
        )
    )
    names = ["nop", EXPECTED[task] if expected else "forward"]
    (path / "actions.jsonl").write_text("".join(json.dumps({"name": name}) + "\n" for name in names))


def test_all29_rollout_summary_requires_each_successful_semantic_task(tmp_path):
    paths = []
    for task in EXPECTED:
        path = tmp_path / task
        write_rollout(path, task)
        paths.append(path)

    report = summarize(paths)

    assert report["verified"]
    assert {row["expected_action"] for row in report["tasks"]} == set(EXPECTED.values())


def test_all29_rollout_summary_rejects_action_without_task_reward(tmp_path):
    paths = []
    for task in EXPECTED:
        path = tmp_path / task
        write_rollout(path, task, reward=0.0 if task == "craft_furnace_from_iron_ore" else 1.0)
        paths.append(path)

    report = summarize(paths)

    assert not report["verified"]
    furnace = next(row for row in report["tasks"] if row["task"] == "craft_furnace_from_iron_ore")
    assert not furnace["checks"]["positive_task_reward"]
