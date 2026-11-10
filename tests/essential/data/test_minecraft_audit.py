"""Check full-dataset invariants enforced by the D29 label audit."""

import pytest

from examples.minecraft.audit_discrete import validate_split


def audit_result(examples=2, ticks=30):
    return {
        "examples": examples,
        "ticks": ticks,
        "class_counts": {"nop": ticks},
        "camera_quantization_mae": [0.1, 0.2],
    }


def test_audit_split_matches_dataset_example_and_horizon_totals():
    validate_split("train", audit_result(), {"samples": {"train": 2}, "horizon": 15})


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (audit_result(examples=1), "audit/manifest mismatch"),
        (audit_result(ticks=29), "tick mismatch"),
        ({**audit_result(), "class_counts": {"nop": 29}}, "class-count mismatch"),
        ({**audit_result(), "camera_quantization_mae": [float("nan"), 0.2]}, "camera audit is nonfinite"),
    ],
)
def test_audit_split_rejects_incomplete_or_invalid_data(result, message):
    with pytest.raises(ValueError, match=message):
        validate_split("train", result, {"samples": {"train": 2}, "horizon": 15})
