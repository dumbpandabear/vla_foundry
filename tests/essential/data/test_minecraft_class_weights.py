"""Check D29 weights derived from full-corpus label counts."""

import pytest

from examples.minecraft.compute_class_weights import compute_weights
from vla_foundry.data.minecraft.actions import D29_NAMES


def test_class_weights_use_sqrt_inverse_frequency_with_cap_and_zeros():
    counts = {name: 25 for name in D29_NAMES}
    counts[D29_NAMES[0]] = 100
    counts[D29_NAMES[1]] = 1
    counts[D29_NAMES[2]] = 0

    weights = compute_weights({"splits": {"train": {"class_counts": counts}}}, cap=8.0)

    assert weights[0] == 1.0
    assert weights[1] == 8.0
    assert weights[2] == 0.0
    assert weights[3] == 2.0


def test_class_weights_reject_empty_training_audit():
    counts = dict.fromkeys(D29_NAMES, 0)

    with pytest.raises(ValueError, match="no D29 targets"):
        compute_weights({"splits": {"train": {"class_counts": counts}}})
