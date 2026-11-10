"""Portable inference for the GB300 VPT ViT recipe (20 keys + camera).

This vision-action policy uses two images and no language or action history.
Export needs the saved experiment configs, checkpoint, and TRAIN statistics.
"""

import argparse
import hashlib
import json
from pathlib import Path

import draccus
import numpy as np
import torch
import yaml

from vla_foundry.data.minecraft.actions import to_discrete29
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import unwrap_state_dict
from vla_foundry.models import create_model
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.model_params import DiffusionPolicyParams

VPT_KEYS = (
    "forward",
    "back",
    "left",
    "right",
    "jump",
    "sneak",
    "sprint",
    "attack",
    "use",
    "drop",
    "inventory",
    *(f"hotbar.{i}" for i in range(1, 10)),
)
VPT_SCHEMA = "vpt20_camera_degrees_v1"


def decode_vpt_action(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (22,) or not np.isfinite(vector).all():
        raise ValueError("Expected 22 finite, denormalized VPT values")
    action = {key: int(value > 0.5) for key, value in zip(VPT_KEYS, vector[:20], strict=True)}
    action["camera"] = vector[20:].clip(-40, 40).tolist()
    return action


def validate_recipe(model, processor):
    if model.get("type") != "diffusion_policy" or model["vision_language_backbone"]["type"] != "vit_backbone":
        raise ValueError("Expected the recipe's native ViT diffusion policy")
    if not model.get("use_flow_matching_scheduler") or model["action_dim"] != 22 or model["proprioception_dim"] != 0:
        raise ValueError("Expected a 22-dimensional flow-matching policy")
    if (
        processor["processor"] != "none"
        or processor["image_size"] != 128
        or processor["action_fields"] != ["keys", "camera"]
        or processor["image_indices"] != [-1, 0]
        or processor["camera_names"] != ["camera"]
        or processor["lowdim_past_timesteps"] != 0
        or processor["lowdim_future_timesteps"] != 14
        or processor["proprioception_fields"]
    ):
        raise ValueError("Processor differs from the two-frame, no-past VPT recipe")
    norm = processor["normalization"]
    if not norm["enabled"] or not norm["centered_norm"] or norm["field_configs"]["keys"]["enabled"]:
        raise ValueError("Expected unnormalized keys and centered camera normalization")
    camera = norm["field_configs"]["camera"]
    if not camera["enabled"] or camera["method"] != "percentile_1_99" or camera["scope"] != "global":
        raise ValueError("Expected global percentile_1_99 camera normalization")


def export_vpt(run: Path, checkpoint: Path, train_stats: Path, output: Path):
    model_cfg = yaml.safe_load((run / "config_model.yaml").read_text())
    proc_cfg = yaml.safe_load((run / "config_processor.yaml").read_text())
    validate_recipe(model_cfg, proc_cfg)
    stats = json.loads(train_stats.read_text())
    if len(stats["keys"]["mean"]) != 20 or len(stats["camera"]["mean"]) != 2:
        raise ValueError("Training statistics do not match the VPT action schema")
    # These are trusted local training checkpoints, which can contain optimizer objects.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(saved["state_dict"])
    model_cfg["resume_from_checkpoint"] = None
    # Decode and strictly load before creating a bundle so incompatibilities fail early.
    model = create_model(draccus.decode(DiffusionPolicyParams, model_cfg), load_pretrained=False)
    model.load_state_dict(state, strict=True)
    output.mkdir(parents=True, exist_ok=False)
    for field in ("dataset_manifest", "val_dataset_manifest", "dataset_statistics", "val_dataset_statistics"):
        proc_cfg[field] = []
    proc_cfg["action_dim"], proc_cfg["proprioception_dim"] = 22, 0
    proc_cfg["image_names"] = ["camera_t-1", "camera_t0"]
    (output / "config_model.yaml").write_text(yaml.safe_dump(model_cfg))
    (output / "config_processor.yaml").write_text(yaml.safe_dump(proc_cfg))
    (output / "stats.json").write_text(json.dumps(stats) + "\n")
    torch.save(state, output / "policy.pt")
    metadata = {
        "schema": VPT_SCHEMA,
        "action_dim": 22,
        "horizon": 15,
        "keys": VPT_KEYS,
        "language_conditioned": False,
        "checkpoint": str(checkpoint),
        "training_step": saved.get("global_step"),
        "train_stats_source": str(train_stats),
        "train_stats_sha256": hashlib.sha256(train_stats.read_bytes()).hexdigest(),
        "policy_sha256": hashlib.sha256((output / "policy.pt").read_bytes()).hexdigest(),
    }
    (output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


class VPTViTPolicy:
    def __init__(self, bundle: str, device: str = "cpu", inference_steps: int = 10):
        self.bundle, self.device = Path(bundle), torch.device(device)
        metadata = json.loads((self.bundle / "policy.json").read_text())
        if metadata["schema"] != VPT_SCHEMA or tuple(metadata["keys"]) != VPT_KEYS:
            raise ValueError("Unsupported VPT policy schema or key order")
        model_cfg = yaml.safe_load((self.bundle / "config_model.yaml").read_text())
        proc_cfg = yaml.safe_load((self.bundle / "config_processor.yaml").read_text())
        validate_recipe(model_cfg, proc_cfg)
        proc_cfg["dataset_statistics"] = [str(self.bundle / "stats.json")]
        self.processor = RoboticsProcessor(draccus.decode(RoboticsDataParams, proc_cfg))
        self.model = create_model(draccus.decode(DiffusionPolicyParams, model_cfg), load_pretrained=False)
        state = torch.load(self.bundle / "policy.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.inference_steps, self.previous_image = inference_steps, None
        self.image_size = self.processor.data_params.image_size

    def reset(self):
        self.previous_image = None

    @torch.inference_mode()
    def act(self, observation, goal: str = ""):
        image = np.asarray(observation)
        previous = image if self.previous_image is None else self.previous_image
        self.previous_image = image.copy()
        batch = {"images": [{"camera_t-1": previous, "camera_t0": image}], "language_instruction": [""], "lowdim": [{}]}
        inputs = self.processor.process_inputs(batch, image_names=["camera_t-1", "camera_t0"])
        inputs = {key: value.to(self.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        predicted = self.model.generate_actions(
            **inputs, actions=torch.zeros(1, 15, 22, device=self.device), num_inference_steps=self.inference_steps
        )
        # Keys remain in 0..1 units; the camera must return to [pitch,yaw] DEGREES.
        camera = self.processor.normalizer.denormalize_tensor(predicted[..., 20:], "camera")
        vector = torch.cat((predicted[..., :20], camera), dim=-1)[0, 0].float().cpu().numpy()
        action = decode_vpt_action(vector)
        index, diagnostics = to_discrete29(action)
        return index, {**diagnostics, "vpt_action": action, "language_conditioned": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_vpt(args.run, args.checkpoint, args.train_stats, args.output)
