"""Tests for raw-VPT and native-D29 supervision paths."""

import numpy as np
import pytest

from vla_foundry.data.minecraft.discrete_actions import action_targets, discretize


def test_action_targets_preserves_existing_vpt_projection():
    actions = np.full((2, 15, 25), -1, dtype=np.float32)
    actions[..., -2:] = 0

    np.testing.assert_array_equal(action_targets(actions), discretize(actions))


def test_action_targets_accepts_all_native_d29_ids():
    actions = np.arange(29, dtype=np.float32).reshape(1, 29, 1)

    np.testing.assert_array_equal(action_targets(actions), np.arange(29).reshape(1, 29))


@pytest.mark.parametrize("value", [-1, 29, 2.5, float("nan")])
def test_action_targets_rejects_invalid_native_id(value):
    with pytest.raises(ValueError, match="D29 labels"):
        action_targets(np.asarray([[[value]]], dtype=np.float32))
