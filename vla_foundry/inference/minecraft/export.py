"""Export the checkpoint with lowest validation loss as a portable policy."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
import yaml

from vla_foundry.data.minecraft.actions import ACTION_DIM
from vla_foundry.file_utils import unwrap_state_dict


def export(run: Path, dataset: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Use a new bundle directory: {output}")
    matches = re.findall(r"Validation after checkpoint (\d+): avg_loss=([\d.eE+-]+)", (run / "out.log").read_text())
    if not matches:
        raise ValueError("No completed validation checkpoint found")
    number, loss = min(matches, key=lambda pair: float(pair[1]))
    checkpoint = run / "checkpoints" / f"checkpoint_{number}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = yaml.safe_load((run / "config_model.yaml").read_text())
    output.mkdir(parents=True)
    hf_backbone = model_config["vision_language_backbone"]["type"] == "vlm_backbone"
    if not hf_backbone:
        backbone = Path(model_config["vision_language_backbone"]["vlm_experiment_dir"])
        if not backbone.is_absolute():
            backbone = Path.cwd() / backbone
        (output / "backbone").mkdir()
        shutil.copyfile(backbone / "config_model.yaml", output / "backbone" / "config_model.yaml")
        model_config["vision_language_backbone"]["vlm_experiment_dir"] = "backbone"
    model_config["resume_from_checkpoint"] = None
    (output / "config_model.yaml").write_text(yaml.safe_dump(model_config))
    processor_config = yaml.safe_load((run / "config_processor.yaml").read_text())
    for field in ("dataset_manifest", "dataset_statistics", "val_dataset_manifest", "val_dataset_statistics"):
        processor_config[field] = []
    (output / "config_processor.yaml").write_text(yaml.safe_dump(processor_config))
    torch.save(unwrap_state_dict(state["state_dict"]), output / "policy.pt")
    summary = json.loads((dataset / "dataset.json").read_text())
    metadata = {
        "schema": summary["schema"],
        "action_dim": ACTION_DIM,
        "horizon": summary["horizon"],
        "camera_scale": summary["camera_scale"],
        "image_indices": summary.get("image_indices", [0]),
        "language_conditioned": True,
        "dataset": summary,
        "checkpoint": str(checkpoint),
        "validation_loss": float(loss),
        "training_step": state.get("global_step"),
        "initialization": "pretrained HF VLM with LoRA" if hf_backbone else "native VLM from scratch",
        "trainable_only": state.get("trainable_only", False),
        "base_model": model_config["vision_language_backbone"].get("hf_pretrained"),
        "craftium_action_space": "Dict keys + continuous camera; optional lossy Discrete(29) projection",
        "policy_sha256": hashlib.sha256((output / "policy.pt").read_bytes()).hexdigest(),
    }
    (output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.dataset, args.output)
