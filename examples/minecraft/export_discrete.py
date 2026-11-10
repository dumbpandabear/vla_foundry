"""Export the best validation checkpoint without modifying continuous bundles."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch

from vla_foundry.data.minecraft.actions import D29_NAMES
from vla_foundry.data.minecraft.discrete_actions import DISCRETE_SCHEMA
from vla_foundry.file_utils import unwrap_state_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--checkpoint-number", type=int)
    args = parser.parse_args()
    matches = re.findall(
        r"Validation after checkpoint (\d+): avg_loss=([\d.eE+-]+)", (args.run / "out.log").read_text()
    )
    if not matches:
        raise ValueError("No validated checkpoints")
    if args.checkpoint_number is None:
        number, loss = min(matches, key=lambda pair: float(pair[1]))
    else:
        selected = [pair for pair in matches if int(pair[0]) == args.checkpoint_number]
        if not selected:
            raise ValueError(f"Checkpoint {args.checkpoint_number} has no validation result")
        number, loss = selected[-1]
    checkpoint = args.run / "checkpoints" / f"checkpoint_{number}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    audit = json.loads(args.audit.read_text())
    counts = audit["splits"]["train"]["class_counts"]
    supported_ids = [i for i, name in enumerate(D29_NAMES) if counts.get(name, 0) > 0]
    if not supported_ids:
        raise ValueError("Training audit contains no supported D29 actions")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("config_model.yaml", "config_processor.yaml"):
        shutil.copyfile(args.run / name, args.output / name)
    shutil.copyfile(args.audit, args.output / "label_audit.json")
    torch.save(unwrap_state_dict(state["state_dict"]), args.output / "policy.pt")
    metadata = dict(
        schema=DISCRETE_SCHEMA,
        action_dim=29,
        horizon=15,
        camera_delta=10.0,
        camera_threshold=5.0,
        actions=list(D29_NAMES),
        supported_ids=supported_ids,
        image_size=224,
        image_indices=[-1, 0],
        trainable_only=True,
        checkpoint=str(checkpoint),
        training_step=state.get("global_step"),
        validation_loss=float(loss),
        sha256=hashlib.sha256((args.output / "policy.pt").read_bytes()).hexdigest(),
    )
    (args.output / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
