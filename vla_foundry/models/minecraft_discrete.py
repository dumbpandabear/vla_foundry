"""Categorical Minecraft action chunks using Foundry's VLM and transformer."""
import torch
from torch import nn

from vla_foundry.data.minecraft.discrete_actions import SUPPORTED_IDS, action_targets
from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.batch_handlers import BatchHandler
from vla_foundry.models.registry import create_model, register_batch_handler, register_model
from vla_foundry.models.vision_language_backbones import get_vision_language_backbone


class MinecraftDiscretePolicy(BaseModel):
    def __init__(self, params, backbone, transformer):
        super().__init__(params)
        self.vision_language_backbone = backbone
        self.transformer = transformer
        self.condition_encode = nn.Linear(backbone.get_conditioning_embeddings_dim(), transformer.hidden_dim)
        self.tick_queries = nn.Parameter(torch.randn(params.horizon, transformer.hidden_dim) * .02)
        self.output_layer = nn.Linear(transformer.hidden_dim, 29)
        supported = SUPPORTED_IDS if params.supported_action_ids is None else params.supported_action_ids
        if not supported or len(set(supported)) != len(supported) or any(i < 0 or i >= 29 for i in supported):
            raise ValueError("supported_action_ids must contain unique D29 IDs in [0, 29)")
        self.supported_action_ids = tuple(supported)

    def set_grad_checkpointing(self, enable=True):
        self.vision_language_backbone.set_grad_checkpointing(enable)

    def forward(self, input_ids, pixel_values, attention_mask=None, attention_mask_images=None, **kwargs):
        features = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
            attention_mask_images=attention_mask_images, **kwargs).embeddings
        conditioning = self.condition_encode(features)
        queries = self.tick_queries.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        out = self.transformer(inputs_embeds=torch.cat([conditioning, queries], dim=1),
                               output_hidden_states=True, use_cache=False)
        return self.output_layer(out.hidden_states[-1][:, -self.model_params.horizon:])

    @torch.inference_mode()
    def generate_actions(self, **inputs):
        logits = self(**inputs).float()
        # Mask classes that have no positive demonstrations in this policy's
        # training corpus. A native-D29 Craftium run can enable all 29 IDs.
        valid = torch.zeros(29, dtype=torch.bool, device=logits.device)
        valid[list(self.supported_action_ids)] = True
        return logits.masked_fill(~valid, float("-inf")).argmax(dim=-1)


@register_model("minecraft_discrete_policy")
def create_discrete_policy(params, load_pretrained=True):
    return MinecraftDiscretePolicy(params,
        get_vision_language_backbone(params.vision_language_backbone, load_pretrained),
        create_model(params.transformer, load_pretrained))


@register_batch_handler("minecraft_discrete_policy")
class MinecraftDiscreteBatchHandler(BatchHandler):
    def prepare_inputs(self, batch, device, cfg):
        return self._move_to_device({k: v for k, v in batch.items()
                                    if k not in ("actions", "past_mask", "future_mask", "proprioception")}, device)

    def prepare_inputs_and_targets(self, batch, device, cfg):
        actions = batch["actions"].detach().cpu().numpy()
        targets = torch.from_numpy(action_targets(actions)).to(device)
        valid = batch["future_mask"].to(device).bool()
        if batch["past_mask"].any():
            raise ValueError("D29 POC must not receive previous actions")
        targets = targets.masked_fill(~valid, -100)
        return self.prepare_inputs(batch, device, cfg), targets, None

    def compute_loss(self, outputs, targets, loss_fn, cfg, mask=None):
        return loss_fn(input=outputs, target=targets, mask=mask)
