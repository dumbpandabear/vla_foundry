"""Evaluate generated controls on held-out shards; include a no-op baseline."""

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from vla_foundry.data.minecraft.actions import CAMERA_SCALE
from vla_foundry.inference.minecraft.policy import MinecraftPolicy


def iter_samples(manifest: Path):
    for line in manifest.read_text().splitlines():
        shard = manifest.parent / (json.loads(line)["shard"] + ".tar")
        with tarfile.open(shard) as archive:
            sample = {}
            current = None
            for member in archive:
                key, suffix = member.name.split(".", 1)
                if current is not None and key != current:
                    yield sample
                    sample = {}
                current = key
                sample[suffix] = archive.extractfile(member).read()
            if sample:
                yield sample


def metrics(predicted, targets):
    p, t = predicted[..., :-2] > 0, targets[..., :-2] > 0
    tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
    return {
        "button_accuracy": float((p == t).mean()),
        "button_micro_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "camera_mae_degrees": float(np.abs(predicted[..., -2:] - targets[..., -2:]).mean() * CAMERA_SCALE),
    }


def evaluate(args):
    import torch

    torch.manual_seed(42)
    policy = MinecraftPolicy(str(args.bundle), args.device)
    predictions, targets, images, goals, pending = [], [], [], [], []

    def flush():
        predictions.append(policy.predict(images, goals))
        targets.extend(pending)
        images.clear()
        goals.clear()
        pending.clear()

    for i, sample in enumerate(iter_samples(args.manifest)):
        if i >= args.max_samples:
            break
        images.append(np.asarray(Image.open(io.BytesIO(sample["pov_t0.jpg"])).convert("RGB")))
        goals.append(json.loads(sample["language_instructions.json"])["original"][0])
        with np.load(io.BytesIO(sample["lowdim.npz"])) as arrays:
            pending.append(arrays["vpt_actions"].copy())
        if len(images) == args.batch_size:
            flush()
    if images:
        flush()
    truth, predicted = np.stack(targets), np.concatenate(predictions)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite generated controls")
    baseline = np.full_like(truth, -1)
    baseline[..., -2:] = 0
    result = {
        "samples": len(truth),
        "manifest": str(args.manifest),
        "policy": metrics(predicted, truth),
        "noop_baseline": metrics(baseline, truth),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    evaluate(parser.parse_args())
