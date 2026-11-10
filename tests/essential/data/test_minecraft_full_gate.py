"""Check the full-corpus rollout quality gate."""

from examples.minecraft.gate_discrete_full import build_report


def metric(macro_f1, camera_f1=0.06):
    return {
        "supported_macro_f1": macro_f1,
        "camera_macro_f1": camera_f1,
        "per_action_f1": {str(index): 0.03 for index in range(29)},
    }


def evaluation(policy=0.18):
    return {
        "cases": {
            "policy": {"chunk": metric(policy)},
            "random_train_prior": {"chunk": metric(0.05)},
            "shuffled_instructions": {"chunk": metric(0.12)},
            "black_images": {"chunk": metric(0.10)},
        }
    }


def test_full_gate_accepts_loss_and_offline_learning_evidence():
    report = build_report(
        evaluation(),
        {"validation_loss": 1.5, "checkpoint": "checkpoint_8.pt"},
        "avg_loss=1.80\navg_loss=1.74\n",
    )

    assert report["passed"]
    assert report["actions_with_f1_above_0_02"] == 29


def test_full_gate_rejects_policy_without_loss_improvement():
    report = build_report(
        evaluation(),
        {"validation_loss": 1.79, "checkpoint": "checkpoint_2.pt"},
        "avg_loss=1.80\navg_loss=1.79\n",
    )

    assert not report["passed"]
    assert not report["checks"]["validation_loss_decreased"]
