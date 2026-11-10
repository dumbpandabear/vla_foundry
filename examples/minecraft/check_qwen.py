import argparse
import io
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import draccus
import numpy as np
import torch
import yaml
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText

from vla_foundry.data.processor.robotics_processor import RoboticsProcessor
from vla_foundry.file_utils import load_trainable_state
from vla_foundry.inference.minecraft.evaluate import iter_samples
from vla_foundry.models import create_model
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.model_params import DiffusionPolicyParams

torch.set_num_threads(2)
parser = argparse.ArgumentParser()
parser.add_argument("--tiny", action="store_true")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--steps", type=int, default=1)
args = parser.parse_args()
c = yaml.safe_load(Path("examples/minecraft/qwen3_lora.yaml").read_text())
if "--tiny" in sys.argv:
    base = Path(c["model"]["vision_language_backbone"]["hf_pretrained"])
    temp = Path(tempfile.mkdtemp(prefix="minecraft-qwen-tiny-"))
    q = json.loads((base / "config.json").read_text())
    q["text_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
    )
    q["vision_config"].update(
        depth=3,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        deepstack_visual_indexes=[0, 1, 2],
    )
    (temp / "config.json").write_text(json.dumps(q))
    toy_base = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(temp), dtype=torch.bfloat16)
    toy_base.save_pretrained(temp)
    del toy_base
    c["model"]["vision_language_backbone"].update(hf_pretrained=str(temp))
    c["model"]["transformer"].update(hidden_dim=32, n_layers=2, n_heads=4)
    c["model"]["vision_language_backbone"]["lora_rank"] = 4
    device = "cpu"
    pretrained = True
else:
    device = "cuda"
    pretrained = True
    print("GPU", torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory, flush=True)
p = draccus.decode(DiffusionPolicyParams, c["model"])
object.__setattr__(p, "action_dim", 25)
model = create_model(p, load_pretrained=pretrained).to(device).train()
model.set_grad_checkpointing()
processor = RoboticsProcessor(RoboticsDataParams.from_dict(c["data"]))
raw = []
for _i, s in enumerate(iter_samples(Path(c["data"]["dataset_manifest"][0]))):
    images = {n: np.asarray(Image.open(io.BytesIO(s[n + ".jpg"])).convert("RGB")) for n in ["pov_t-1", "pov_t0"]}
    with np.load(io.BytesIO(s["lowdim.npz"])) as a:
        actions = a["vpt_actions"].copy()
    raw.append((images, json.loads(s["language_instructions.json"])["original"][0], actions))
    if len(raw) == args.batch_size:
        break
batch = {
    "images": [r[0] for r in raw],
    "language_instruction": [r[1] for r in raw],
    "lowdim": [{"vpt_actions": r[2]} for r in raw],
}
inp = processor.process_inputs(batch, image_names=["pov_t-1", "pov_t0"], max_text_seq_len=1024)
inp = {k: v.to(device) for k, v in inp.items() if isinstance(v, torch.Tensor)}
a = torch.tensor(np.stack([r[2] for r in raw]), device=device)
noise = torch.randn_like(a)
mask = torch.ones(a.shape[:2], dtype=torch.bool, device=device)
print("INPUTS", {k: list(v.shape) for k, v in inp.items()}, flush=True)
optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=1e-4)
started = time.monotonic()
with torch.autocast(device, dtype=torch.bfloat16, enabled=True):
    out = model(**inp, actions=a, noise=noise, past_mask=~mask, future_mask=mask)
    loss = (out.float() - (noise - a)).square().mean()
assert torch.isfinite(loss)
loss.backward()
grad = {n: float(v.grad.norm()) for n, v in model.named_parameters() if v.grad is not None}
assert any("lora_B" in n and g > 0 for n, g in grad.items())
assert any("output_layer" in n and g > 0 for n, g in grad.items())
assert not any(("experts" in n or ".visual." in n) for n in grad)
optimizer.step()
print(
    "PASS",
    float(loss.detach()),
    "trainable",
    sum(v.numel() for v in model.parameters() if v.requires_grad),
    "grad tensors",
    len(grad),
    flush=True,
)
if device == "cuda":
    print("peak_GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)
    print("probe_step_seconds", time.monotonic() - started, flush=True)
    for _step in range(1, args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(**inp, actions=a, noise=noise, past_mask=~mask, future_mask=mask)
            loss = (out.float() - (noise - a)).square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        print("steady_step_seconds", time.monotonic() - started, "batch", args.batch_size, flush=True)
    print("FINAL_PEAK_GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)
if "--tiny" in sys.argv:
    names = {n for n, v in model.named_parameters() if v.requires_grad}
    names.update(n for n, v in model.named_buffers())
    compact = {n: v.detach().clone() for n, v in model.state_dict().items() if n in names}
    model.eval()
    restored = create_model(p, load_pretrained=True).eval()
    load_trainable_state(restored, compact)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        expected = model(**inp, actions=a, noise=noise, past_mask=~mask, future_mask=mask)
    torch.manual_seed(42)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        actual = restored(**inp, actions=a, noise=noise, past_mask=~mask, future_mask=mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("COMPACT_RESTORE_IDENTICAL_PREDICTIONS", flush=True)
elif args.steps > 1:
    # Verify the software renderer on the same kind of node used for deployment.
    subprocess.run(["bash", "examples/minecraft/calibrate_camera.slurm"], check=True, timeout=600)
    print("CRAFTIUM_SAME_NODE_PASS", flush=True)
