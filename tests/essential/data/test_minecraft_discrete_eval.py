"""Tests for bundle-scoped Discrete-29 evaluation metrics."""

import numpy as np

from examples.minecraft.evaluate_discrete import metrics
from vla_foundry.data.minecraft.discrete_actions import SUPPORTED_IDS


def test_metrics_include_crafting_classes_when_bundle_supports_all_29():
    target = np.arange(29, dtype=np.int64)[None, :]
    predicted = target.copy()
    predicted[:, 18:26] = 0

    vpt_report = metrics(predicted, target, SUPPORTED_IDS)
    all29_report = metrics(predicted, target, tuple(range(29)))

    assert all29_report["supported_macro_f1"] < vpt_report["supported_macro_f1"]
    assert all29_report["per_action_f1"]["craft_planks"] == 0
