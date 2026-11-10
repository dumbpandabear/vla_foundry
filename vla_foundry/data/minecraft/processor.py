"""Offline byte text processor for the native tiny Minecraft VLM."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


class ByteTokenizer:
    pad_token_id = 0
    vocab_size = 258
    chat_template = None

    def __len__(self):
        return self.vocab_size


class MinecraftByteProcessor:
    """Image placeholder=1; UTF-8 bytes=2..257; padding=0. No downloads."""

    image_token_id = 1
    chat_template = None

    def __init__(self, image_size: int, image_tokens: int):
        self.image_size = image_size
        self.image_tokens = image_tokens
        self.tokenizer = ByteTokenizer()
        self.image_processor = SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.5] * 3)

    def __call__(self, images, text, max_length=None, **kwargs):
        rows, pixels = [], []
        for sample_images, instruction in zip(images, text, strict=True):
            tokens = [1] * (self.image_tokens * len(sample_images))
            # Foundry's plain chat formatting prepends image placeholders.
            instruction = instruction.replace("<image>", "").strip()
            tokens += [b + 2 for b in instruction.encode("utf-8")]
            if max_length is not None:
                if max_length < self.image_tokens * len(sample_images):
                    raise ValueError("Text limit would truncate image tokens")
                tokens = tokens[:max_length]
            rows.append(tokens)
            for image in sample_images:
                if isinstance(image, Image.Image):
                    value = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
                elif isinstance(image, np.ndarray):
                    value = torch.from_numpy(image.copy()).permute(2, 0, 1)
                else:
                    value = image
                if value.dtype != torch.uint8:
                    raise ValueError("Minecraft processor expects RGB uint8 images")
                value = torch.nn.functional.interpolate(
                    value[None].float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
                )[0]
                pixels.append(value / 127.5 - 1)
        ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(rows):
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask, "pixel_values": torch.stack(pixels)}
