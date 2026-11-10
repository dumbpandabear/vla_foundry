"""Load a portable Foundry policy bundle and predict VPT/Craftium actions."""

import json
from pathlib import Path

import draccus
import numpy as np
import torch
import yaml

from vla_foundry.data.minecraft.actions import ACTION_DIM, SCHEMA_VERSION, decode_action, to_discrete29
from vla_foundry.data.minecraft.raw_actions import RAW_SCHEMA
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import load_trainable_state
from vla_foundry.models import create_model
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.model_params import DiffusionPolicyParams


class MinecraftPolicy:
    def __init__(self, bundle: str, device: str = "cpu", inference_steps: int = 10):
        self.bundle = Path(bundle)
        self.device = torch.device(device)
        self.inference_steps = inference_steps
        self.metadata = json.loads((self.bundle / "policy.json").read_text())
        if self.metadata["schema"] not in (SCHEMA_VERSION, RAW_SCHEMA) or self.metadata["action_dim"] != ACTION_DIM:
            raise ValueError("Unsupported Minecraft action schema")
        cfg = yaml.safe_load((self.bundle / "config_model.yaml").read_text())
        if cfg["vision_language_backbone"]["type"] == "vlm_foundry_backbone":
            cfg["vision_language_backbone"]["vlm_experiment_dir"] = str(self.bundle / "backbone")
        params = draccus.decode(DiffusionPolicyParams, cfg)
        compact = self.metadata.get("trainable_only", False)
        self.model = create_model(params, load_pretrained=compact).to(self.device).eval()
        state = torch.load(self.bundle / "policy.pt", map_location="cpu", weights_only=True)
        if compact:
            load_trainable_state(self.model, state)
        else:
            self.model.load_state_dict(state, strict=True)
        data_params = RoboticsDataParams.from_file(str(self.bundle / "config_processor.yaml"))
        self.processor = RoboticsProcessor(data_params)
        self.previous_image = None

    @torch.inference_mode()
    def predict(self, observations, goals) -> np.ndarray:
        """Return [batch, horizon, action_dim] from RGB uint8 and goal strings."""
        if isinstance(goals, str):
            observations, goals = [observations], [goals]
        image_names = self.processor.data_params.image_names
        images = []
        for obs in observations:
            if isinstance(obs, dict):
                images.append(obs)
            elif image_names == ["pov_t0"]:
                images.append({"pov_t0": np.asarray(obs)})
            else:
                raise ValueError("This policy needs both previous and current images")
        batch = {
            "images": images,
            "language_instruction": goals,
            "lowdim": [{} for _ in goals],
        }
        inputs = self.processor.process_inputs(
            batch, image_names=image_names, max_text_seq_len=self.processor.data_params.max_text_seq_len
        )
        model_inputs = {key: value.to(self.device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
        shape = (len(goals), self.metadata["horizon"], ACTION_DIM)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=compact_cuda(self)):
            predicted = self.model.generate_actions(
                **model_inputs, actions=torch.zeros(shape, device=self.device), num_inference_steps=self.inference_steps
            )
        return predicted.float().cpu().numpy()

    def act(self, observation, goal: str) -> tuple[int, dict]:
        """Replan each env step; project the first predicted VPT action to D29."""
        if self.processor.data_params.image_indices == [-1, 0]:
            image = np.asarray(observation)
            observation = {"pov_t-1": image if self.previous_image is None else self.previous_image, "pov_t0": image}
            self.previous_image = image.copy()
        vector = self.predict(observation, goal)[0, 0]
        action = decode_action(vector, camera_scale=self.metadata.get("camera_scale", 180.0))
        index, diagnostics = to_discrete29(action)
        return index, {**diagnostics, "vpt_action": action}


def compact_cuda(policy):
    return policy.device.type == "cuda" and policy.metadata.get("trainable_only", False)
