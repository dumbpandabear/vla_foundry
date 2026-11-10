import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText

from vla_foundry.models.registry import register_model
from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from vla_foundry.params.model_params import VLMHFParams


class VLMHF(TransformerBase):
    def __init__(self, model_params: VLMHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        self.pretrained_loaded = load_pretrained
        load_kwargs = {}
        if model_params.pretrained_dtype:
            load_kwargs["dtype"] = getattr(torch, model_params.pretrained_dtype)
        if model_params.attn_implementation:
            load_kwargs["attn_implementation"] = model_params.attn_implementation
        if load_pretrained:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name, trust_remote_code=True, **load_kwargs
            )
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True, **load_kwargs)
        self._limit_hidden_states_to_last_n = None
        self._setup_model_info()

    def _setup_model_info(self):
        """Detect model-specific information like hidden dimensions and expected image size."""
        model = self.model
        config = model.config

        # Detect language model component
        language_model_ref = None
        for attr in ["language_model", "text_model", "llm", "model"]:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                if candidate is not model and hasattr(candidate, "forward"):
                    language_model_ref = candidate
                    break

        # Detect language model hidden dimension
        self._lm_hidden_dim = None
        if language_model_ref is not None:
            lm_config = getattr(language_model_ref, "config", None)
            if lm_config is not None:
                for attr in ["hidden_size", "d_model", "n_embd", "hidden_dim"]:
                    if hasattr(lm_config, attr):
                        self._lm_hidden_dim = getattr(lm_config, attr)
                        break

        # Detect patches per image from vision config
        self._patches_per_image = None
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            img_size = getattr(vision_config, "image_size", None)
            patch_size = getattr(vision_config, "patch_size", 14)
            if isinstance(img_size, int) and isinstance(patch_size, int):
                self._patches_per_image = (img_size // patch_size) ** 2
            elif isinstance(img_size, (list, tuple)) and isinstance(patch_size, int):
                self._patches_per_image = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # SmolVLM/Idefics-family models expect pixel_values as 5D [B, N, C, H, W]
        # (per-sample image count in the second dim). PaliGemma/CLIP/Qwen-VL do not.
        model_type = getattr(config, "model_type", "").lower()
        self._expects_5d_pixel_values = model_type in {"smolvlm", "idefics2", "idefics3"}

    @property
    def lm_hidden_dim(self):
        """Get the language model's hidden dimension (may differ from embedding dim)."""
        if self._lm_hidden_dim is not None:
            return self._lm_hidden_dim
        # Fallback to general hidden_dim
        return self.hidden_dim

    def forward(self, input_ids, pixel_values, attention_mask=None, output_hidden_states=False, **kwargs):
        # SmolVLM/Idefics expect 5D [B, N, C, H, W]; reshape flat 4D from
        # processors like simple_vlm using batch_size from input_ids.
        if (
            pixel_values is not None
            and pixel_values.ndim == 4
            and self._expects_5d_pixel_values
            and input_ids is not None
        ):
            batch_size = input_ids.shape[0]
            num_images = pixel_values.shape[0] // batch_size
            if num_images * batch_size != pixel_values.shape[0]:
                raise ValueError(
                    f"pixel_values batch size ({pixel_values.shape[0]}) is not divisible by "
                    f"input_ids batch size ({batch_size}); cannot infer images-per-sample."
                )
            pixel_values = pixel_values.reshape(batch_size, num_images, *pixel_values.shape[1:])

        out = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        if self._limit_hidden_states_to_last_n is not None and output_hidden_states:
            out.hidden_states = out.hidden_states[-self._limit_hidden_states_to_last_n :]

        return out

    def resize_token_embeddings(self, new_num_tokens: int = None) -> int:
        """Add a new token to the vocabulary and return its ID.

        Args:
            new_num_tokens: The new vocabulary size. If None, adds exactly one token.

        Returns:
            The ID (index) of the newly added token.
        """
        current_size = int(self.model.get_input_embeddings().num_embeddings)

        if new_num_tokens is None:
            new_num_tokens = current_size + 1

        if new_num_tokens > current_size:
            print(f"Resizing token embeddings from {current_size} to {new_num_tokens}")
            self.model.resize_token_embeddings(new_num_tokens, mean_resizing=False)

        # Return the ID of the last token (the newly added one)
        return new_num_tokens - 1

    def enable_lora(self, action_token_id):
        """Adapt attention and the new probe token while freezing the pretrained weights."""
        if not self.model_params.lora_rank:
            return
        if self.model_params.freeze:
            raise ValueError("LoRA requires freeze=false; PEFT freezes the base weights itself")
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=self.model_params.lora_rank,
            lora_alpha=self.model_params.lora_alpha,
            lora_dropout=self.model_params.lora_dropout,
            target_modules=self.model_params.lora_target_modules,
            trainable_token_indices=[action_token_id],
            bias="none",
        )
        self._embedding_dtype = self.model.get_input_embeddings().weight.dtype
        self.model = get_peft_model(self.model, config)
        # PEFT keeps the probe parameter in FP32. Preserve the base activation
        # dtype: Qwen's MoE scatter requires routing weights and logits to match.
        self.model.get_input_embeddings().register_forward_hook(self._cast_embedding_output)
        # Transformers 4.57 Qwen3-VL MoE casts routing probabilities to the
        # hidden-state dtype before scattering into a logits-shaped tensor.
        # Under autocast these dtypes can differ during evaluation. Match the
        # gate output to its input, preserving the router's FP32 softmax and
        # gradients without changing the frozen weights or expert routing.
        for module in self.model.modules():
            if type(module).__name__ == "Qwen3VLMoeTextSparseMoeBlock":
                module.gate.register_forward_hook(self._match_router_output_dtype)
        self.model.print_trainable_parameters()

    def _cast_embedding_output(self, module, args, output):
        return output.to(self._embedding_dtype)

    def _match_router_output_dtype(self, module, args, output):
        return output.to(args[0].dtype)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if enable:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                self.model.gradient_checkpointing_disable()

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)

    def generate(self, input_ids, pixel_values, attention_mask, max_new_tokens=20, **kwargs):
        """Generate text tokens using the VLM HF model"""
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated, pixel_values=pixel_values, attention_mask=attn_mask, **kwargs)
            last_output = outputs.logits[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

        return generated

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        block_types = set()

        # Find text/language model blocks
        for attr in ["language_model", "text_model"]:
            if hasattr(self.model.model, attr):
                for _name, module in getattr(self.model.model, attr).named_modules():
                    if isinstance(module, nn.ModuleList) and len(module) > 0:
                        block_types.add(type(module[0]))

        # Find vision model blocks
        if hasattr(self.model.model, "vision_model") and hasattr(self.model.model.vision_model, "encoder"):
            for _name, module in self.model.model.vision_model.encoder.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    block_types.add(type(module[0]))

        if not block_types:
            raise ValueError("Could not find any model block classes.")

        return tuple(block_types)


@register_model("vlm_hf")
def create_vlm_hf(model_params: VLMHFParams, load_pretrained: bool = True):
    return VLMHF(model_params, load_pretrained=load_pretrained)
