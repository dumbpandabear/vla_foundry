"""Offline Minecraft VLA evaluation with instruction and image ablations."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vla_foundry.data.minecraft.actions import BUTTONS
from vla_foundry.inference.minecraft.evaluate import iter_samples
from vla_foundry.inference.minecraft.policy import MinecraftPolicy

MAIN_KEYS = ("forward", "back", "left", "right", "jump", "sneak", "sprint", "attack", "use", "inventory")


def metrics(predicted, target, camera_scale):
    pressed, truth = predicted[..., :-2] > 0, target[..., :-2] > 0
    tp = (pressed & truth).sum(axis=(0, 1))
    fp = (pressed & ~truth).sum(axis=(0, 1))
    fn = (~pressed & truth).sum(axis=(0, 1))
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    camera, target_camera = predicted[..., -2:] * camera_scale, target[..., -2:] * camera_scale
    moving = np.abs(target_camera) >= 1
    return {
        "key_macro_f1": float(np.mean([f1[BUTTONS.index(k)] for k in MAIN_KEYS])),
        "per_key_f1": {key: float(score) for key, score in zip(BUTTONS, f1, strict=True)},
        "key_micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
        "camera_mae_degrees": np.abs(camera - target_camera).mean(axis=(0, 1)).tolist(),
        "camera_direction_accuracy": [
            float(
                (
                    np.sign(camera[..., axis])[moving[..., axis]]
                    == np.sign(target_camera[..., axis])[moving[..., axis]]
                ).mean()
            )
            if moving[..., axis].any()
            else None
            for axis in (0, 1)
        ],
    }


def evaluate(args):
    policy = MinecraftPolicy(str(args.bundle), args.device)
    names = policy.processor.data_params.image_names
    total = sum(json.loads(line)["num_sequences"] for line in args.manifest.read_text().splitlines())
    selected = set(np.linspace(0, total - 1, min(total, args.max_samples), dtype=int).tolist())
    images, goals, targets, recordings = [], [], [], set()
    for index, sample in enumerate(iter_samples(args.manifest)):
        if index not in selected:
            continue
        images.append(
            {name: np.asarray(Image.open(io.BytesIO(sample[name + ".jpg"])).convert("RGB")) for name in names}
        )
        goals.append(json.loads(sample["language_instructions.json"])["original"][0])
        recordings.add(json.loads(sample["metadata.json"])["recording_id"])
        with np.load(io.BytesIO(sample["lowdim.npz"])) as arrays:
            targets.append(arrays["vpt_actions"].copy())
    truth = np.stack(targets)
    camera_scale = policy.metadata.get("camera_scale", 180.0)
    permutation = np.random.default_rng(42).permutation(len(goals))
    shuffled = [goals[index] for index in permutation]
    report = {
        "samples": len(images),
        "recordings": len(recordings),
        "unique_goals": len(set(goals)),
        "manifest": str(args.manifest),
        "camera_units": "[pitch,yaw] degrees per 20Hz tick",
        "shuffled_goal_fraction_changed": float(np.mean([a != b for a, b in zip(goals, shuffled, strict=True)])),
        "action_history_timesteps": 0,
        "cases": {},
    }
    reference = None
    for case in ("policy", "shuffled_instructions", "black_images"):
        torch.manual_seed(42)  # Match diffusion noise across ablations.
        predictions = []
        for start in range(0, len(images), args.batch_size):
            batch_images = images[start : start + args.batch_size]
            if case == "black_images":
                batch_images = [{key: np.zeros_like(image) for key, image in item.items()} for item in batch_images]
            instructions = (shuffled if case == "shuffled_instructions" else goals)[start : start + args.batch_size]
            predictions.append(policy.predict(batch_images, instructions))
        predicted = np.concatenate(predictions)
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite generated actions")
        report["cases"][case] = {
            "tick_0": metrics(predicted[:, :1], truth[:, :1], camera_scale),
            "chunk": metrics(predicted, truth, camera_scale),
        }
        if case == "policy":
            reference = predicted
        else:
            report["cases"][case]["action_mse_change"] = float(np.mean((predicted - reference) ** 2))
        print(f"{case}: {json.dumps(report['cases'][case]['chunk'])}", flush=True)
    noop = np.full_like(truth, -1)
    noop[..., -2:] = 0
    report["cases"]["noop"] = {
        "tick_0": metrics(noop[:, :1], truth[:, :1], camera_scale),
        "chunk": metrics(noop, truth, camera_scale),
    }
    prior_path = args.manifest.parent.parent / "train" / "action_prior.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior["buttons"] != list(BUTTONS):
            raise ValueError("Training action prior uses a different button order")
        rates = np.asarray(prior["press_probability"])
        report["training_action_prior"] = prior
        random_prior = noop.copy()
        random_prior[..., :-2] = np.where(np.random.default_rng(42).random(truth[..., :-2].shape) < rates, 1, -1)
        majority = noop.copy()
        majority[..., :-2] = np.where(rates >= 0.5, 1, -1)
        for name, baseline in (("random_train_prior", random_prior), ("train_majority", majority)):
            report["cases"][name] = {
                "tick_0": metrics(baseline[:, :1], truth[:, :1], camera_scale),
                "chunk": metrics(baseline, truth, camera_scale),
            }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    evaluate(parser.parse_args())
