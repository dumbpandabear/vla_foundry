"""Exercise a portable native ViT checkpoint with train camera normalization."""

import json

import draccus
import numpy as np
import torch
import yaml

from vla_foundry.inference.minecraft.vpt_vit import VPTViTPolicy, decode_vpt_action, export_vpt
from vla_foundry.models import create_model
from vla_foundry.params.model_params import DiffusionPolicyParams, TransformerParams, ViTBackboneParams


def test_vpt_checkpoint_port_preserves_camera_units_and_image_history(tmp_path):
    torch.set_num_threads(2)
    params = DiffusionPolicyParams(
        vision_language_backbone=ViTBackboneParams(
            img_size=128, patch_size=16, hidden_dim=32, inter_dim=64, n_layers=1, n_heads=4
        ),
        transformer=TransformerParams(hidden_dim=32, n_layers=1, n_heads=4, vocab_size=0, is_causal=False),
        action_dim=22,
        proprioception_dim=0,
        use_flow_matching_scheduler=True,
        num_action_head_repeats=1,
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "config_model.yaml").write_text(yaml.safe_dump(draccus.encode(params)))
    processor = {
        "processor": "none",
        "image_size": 128,
        "action_fields": ["keys", "camera"],
        "camera_names": ["camera"],
        "image_indices": [-1, 0],
        "proprioception_fields": [],
        "lowdim_past_timesteps": 0,
        "lowdim_future_timesteps": 14,
        "normalization": {
            "enabled": True,
            "centered_norm": True,
            "include_fields": ["keys", "camera"],
            "field_configs": {
                "keys": {"enabled": False},
                "camera": {"enabled": True, "method": "percentile_1_99", "scope": "global"},
            },
        },
    }
    (run / "config_processor.yaml").write_text(yaml.safe_dump(processor))
    stats = tmp_path / "train-stats.json"
    stats.write_text(
        json.dumps(
            {
                "keys": {"mean": [0] * 20},
                "camera": {
                    "mean": [0, 0],
                    "percentile_1": [-10, -20],
                    "percentile_99": [10, 20],
                },
            }
        )
    )
    checkpoint = run / "checkpoint.pt"
    torch.save({"state_dict": create_model(params, load_pretrained=False).state_dict()}, checkpoint)
    bundle = tmp_path / "bundle"
    export_vpt(run, checkpoint, stats, bundle)
    # Remove source files; inference must depend only on the portable bundle.
    checkpoint.unlink()
    stats.unlink()
    policy = VPTViTPolicy(str(bundle), inference_steps=2)
    camera = policy.processor.normalizer.denormalize_tensor(torch.tensor([[[1.0, -1.0]]]), "camera")
    torch.testing.assert_close(camera, torch.tensor([[[10.0, -20.0]]]))
    vector = np.zeros(22, dtype=np.float32)
    vector[0], vector[20:] = 0.9, [0.15, -0.3]
    action = decode_vpt_action(vector)
    assert action["forward"] == 1
    np.testing.assert_allclose(action["camera"], [0.15, -0.3])
    frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    index, diagnostics = policy.act(frame, "Unused goal")
    assert 0 <= index < 29
    assert not diagnostics["language_conditioned"]
    assert np.isfinite(diagnostics["vpt_action"]["camera"]).all()
    np.testing.assert_array_equal(policy.previous_image, frame)
    policy.reset()
    assert policy.previous_image is None
