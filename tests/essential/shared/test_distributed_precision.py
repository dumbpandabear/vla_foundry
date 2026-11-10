"""Check mixed-precision model placement shared by single-process and DDP paths."""

from types import SimpleNamespace

import torch

from vla_foundry.distributed import move_model_to_device


def test_amp_pretrained_backbone_preserves_fp32_trainable_parameters():
    model = torch.nn.Module()
    model.register_parameter(
        "frozen_base", torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
    )
    model.register_parameter("adapter", torch.nn.Parameter(torch.ones(1, dtype=torch.float32)))
    cfg = SimpleNamespace(
        model=SimpleNamespace(vision_language_backbone=SimpleNamespace(pretrained_dtype="bfloat16")),
        hparams=SimpleNamespace(precision_amp=True, precision_pure_bf16=False),
    )

    placed = move_model_to_device(model, "cpu", cfg)

    assert placed.frozen_base.dtype == torch.bfloat16
    assert placed.adapter.dtype == torch.float32


def test_pure_bf16_placement_casts_all_parameters():
    model = torch.nn.Linear(2, 2)
    cfg = SimpleNamespace(
        model=SimpleNamespace(vision_language_backbone=None),
        hparams=SimpleNamespace(precision_amp=False, precision_pure_bf16=True),
    )

    placed = move_model_to_device(model, "cpu", cfg)

    assert all(parameter.dtype == torch.bfloat16 for parameter in placed.parameters())
