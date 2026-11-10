"""Verify compact Qwen restore and BF16 evaluation on real Minecraft examples."""

import argparse
import io
import json
from pathlib import Path

import draccus
import numpy as np
import torch
import yaml
from PIL import Image

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import load_trainable_state
from vla_foundry.inference.minecraft.evaluate import iter_samples
from vla_foundry.models import create_model
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.model_params import DiffusionPolicyParams

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--trace-dtypes", action="store_true")
args = parser.parse_args()
torch.set_num_threads(4)
config = yaml.safe_load(Path("examples/minecraft/qwen3_lora.yaml").read_text())
params = draccus.decode(DiffusionPolicyParams, config["model"])
object.__setattr__(params, "action_dim", 25)
model = create_model(params, load_pretrained=True).to("cuda").eval()
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
assert checkpoint["trainable_only"]
load_trainable_state(model, checkpoint["state_dict"])
del checkpoint
if args.trace_dtypes:

    def trace(name):
        def hook(module, inputs, kwargs, output):
            value = output[0] if isinstance(output, tuple) else output
            input_value = inputs[0] if inputs else kwargs["hidden_states"]
            print("DTYPE", name, input_value.dtype, value.dtype, flush=True)

        return hook

    for name, module in model.named_modules():
        if ".layers.0." in name and name.endswith(
            ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp.gate")
        ):
            module.register_forward_hook(trace(name), with_kwargs=True)
processor = RoboticsProcessor(RoboticsDataParams.from_dict(config["data"]))
images, goals, targets = [], [], []
for sample in iter_samples(Path(config["data"]["val_dataset_manifest"][0])):
    images.append(
        {
            name: np.asarray(Image.open(io.BytesIO(sample[name + ".jpg"])).convert("RGB"))
            for name in ("pov_t-1", "pov_t0")
        }
    )
    goals.append(json.loads(sample["language_instructions.json"])["original"][0])
    with np.load(io.BytesIO(sample["lowdim.npz"])) as arrays:
        targets.append(arrays["vpt_actions"].copy())
    if len(images) == args.batch_size:
        break
inputs = processor.process_inputs(
    {"images": images, "language_instruction": goals, "lowdim": [{"vpt_actions": target} for target in targets]},
    image_names=["pov_t-1", "pov_t0"],
    max_text_seq_len=1024,
)
inputs = {key: value.to("cuda") for key, value in inputs.items() if isinstance(value, torch.Tensor)}
actions = torch.tensor(np.stack(targets), device="cuda")
noise = torch.randn_like(actions)
mask = torch.ones(actions.shape[:2], dtype=torch.bool, device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(**inputs, actions=actions, noise=noise, past_mask=~mask, future_mask=mask)
    loss = (output.float() - (noise - actions)).square().mean()
    assert torch.isfinite(loss)
    predicted = model.generate_actions(**inputs, actions=torch.zeros_like(actions), num_inference_steps=10)
    assert predicted.shape == actions.shape and torch.isfinite(predicted).all()
print(
    "EVAL_AND_GENERATION_PASS",
    float(loss),
    "shape",
    list(predicted.shape),
    "peak_GB",
    torch.cuda.max_memory_allocated() / 1e9,
    flush=True,
)
