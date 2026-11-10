"""Restore a compact categorical D29 policy and predict native action IDs."""
import json
from pathlib import Path

import draccus
import torch
import yaml

from vla_foundry.data.minecraft.discrete_actions import DISCRETE_SCHEMA
from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import load_trainable_state
from vla_foundry.models import create_model
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.model_params import MinecraftDiscretePolicyParams


class MinecraftDiscretePolicy:
    def __init__(self, bundle, device='cuda'):
        self.bundle = Path(bundle)
        self.device = torch.device(device)
        self.metadata = json.loads((self.bundle / 'policy.json').read_text())
        if self.metadata['schema'] != DISCRETE_SCHEMA:
            raise ValueError('Unsupported D29 policy schema')
        cfg = yaml.safe_load((self.bundle / 'config_model.yaml').read_text())
        self.model = create_model(draccus.decode(MinecraftDiscretePolicyParams, cfg)).to(device).eval()
        # Bundle provenance is authoritative: it is derived from positive
        # labels in the training audit at export time.
        self.model.supported_action_ids = tuple(self.metadata['supported_ids'])
        load_trainable_state(self.model, torch.load(self.bundle / 'policy.pt',map_location='cpu',weights_only=True))
        self.processor = RoboticsProcessor(RoboticsDataParams.from_file(str(self.bundle / 'config_processor.yaml')))

    @torch.inference_mode()
    def predict(self, observations, goals):
        if isinstance(goals, str):
            observations, goals = [observations], [goals]
        batch = dict(images=observations, language_instruction=goals, lowdim=[{} for _ in goals])
        inputs = self.processor.process_inputs(batch,image_names=self.processor.data_params.image_names,
                                             max_text_seq_len=self.processor.data_params.max_text_seq_len)
        inputs = {k:v.to(self.device) for k,v in inputs.items() if isinstance(v,torch.Tensor)}
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            return self.model.generate_actions(**inputs).cpu().numpy()
