"""Resolve a valid compact checkpoint for restarting the background POC job."""

import pickle
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
config = yaml.safe_load(config_path.read_text())
run = Path(config["save_path"]) / config["name"]
checkpoints = sorted(
    (run / "checkpoints").glob("checkpoint_*.pt"), key=lambda path: int(path.stem.split("_")[-1]), reverse=True
)
for path in checkpoints:
    import torch

    optimizer_path = path.with_name(path.name.replace("checkpoint_", "optimizer_"))
    if not optimizer_path.exists():
        continue
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        assert checkpoint["checkpoint_num"] == optimizer["checkpoint_num"]
        assert checkpoint["trainable_only"]
    except (OSError, RuntimeError, EOFError, AssertionError, KeyError, pickle.UnpicklingError) as error:
        print(f"Incomplete checkpoint {path}: {error}", file=sys.stderr)
        continue
    config["model"]["resume_from_checkpoint"] = str(path)
    print(f"Resuming from {path}", flush=True)
    break
output_path.write_text(yaml.safe_dump(config, sort_keys=False))
