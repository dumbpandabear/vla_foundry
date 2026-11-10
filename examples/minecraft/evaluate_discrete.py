"""D29 held-out accuracy, macro-F1 and camera metrics with matched ablations."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vla_foundry.data.minecraft.actions import D29_NAMES
from vla_foundry.data.minecraft.discrete_actions import action_targets
from vla_foundry.data.minecraft.streaming import iter_selected_index_samples
from vla_foundry.inference.minecraft.discrete_policy import MinecraftDiscretePolicy
from vla_foundry.inference.minecraft.evaluate import iter_samples


def metrics(predicted, target, supported_ids):
    matrix = np.bincount((target * 29 + predicted).reshape(-1), minlength=29 * 29).reshape(29, 29)
    tp = np.diag(matrix)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(29, dtype=float), where=denominator > 0)
    camera = (target >= 14) & (target <= 17)
    return {
        "accuracy": float((predicted == target).mean()),
        "supported_macro_f1": float(f1[list(supported_ids)].mean()),
        "per_action_f1": dict(zip(D29_NAMES, f1.tolist(), strict=True)),
        "confusion_matrix": matrix.tolist(),
        "camera_macro_f1": float(f1[14:18].mean()),
        "camera_action_accuracy": float((predicted[camera] == target[camera]).mean()) if camera.any() else None,
        "camera_target_ticks": int(camera.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vpt-data-root", type=Path)
    args = parser.parse_args()
    policy = MinecraftDiscretePolicy(args.bundle)
    supported_ids = tuple(int(value) for value in policy.metadata["supported_ids"])
    if not supported_ids or len(set(supported_ids)) != len(supported_ids):
        raise ValueError("Bundle has no valid supported action IDs")
    total = sum(json.loads(line)["num_sequences"] for line in args.manifest.read_text().splitlines())
    selected = set(np.linspace(0, total - 1, min(total, args.max_samples), dtype=int).tolist())
    images, goals, targets, recordings = [], [], [], set()
    if args.vpt_data_root is not None:
        stream = iter_selected_index_samples(
            args.manifest,
            args.vpt_data_root,
            image_size=int(policy.metadata["image_size"]),
            horizon=int(policy.metadata["horizon"]),
            selected=selected,
        )
        for sample in stream:
            images.append(sample["images"])
            goals.append(sample["language_instruction"])
            recordings.add(sample["metadata"]["recording_id"])
            targets.append(action_targets(sample["lowdim"]["vpt_actions"]))
    else:
        for index, sample in enumerate(iter_samples(args.manifest)):
            if index not in selected:
                continue
            images.append(
                {
                    name: np.asarray(Image.open(io.BytesIO(sample[name + ".jpg"])).convert("RGB"))
                    for name in ("pov_t-1", "pov_t0")
                }
            )
            goals.append(json.loads(sample["language_instructions.json"])["original"][0])
            sample_metadata = json.loads(sample["metadata.json"])
            recordings.add(sample_metadata.get("recording_id", sample_metadata.get("run_id")))
            with np.load(io.BytesIO(sample["lowdim.npz"])) as row:
                action_key = "d29_actions" if "d29_actions" in row else "vpt_actions"
                targets.append(action_targets(row[action_key]))
    truth = np.stack(targets)

    permutation = None
    for attempt in range(1000):
        candidate = np.random.default_rng(42 + attempt).permutation(len(goals))
        if all(goals[i] != goals[j] for i, j in enumerate(candidate)):
            permutation = candidate
            break
    if permutation is None:
        raise ValueError("Could not construct distinct shuffled instructions")
    report = {
        "samples": len(images),
        "recordings": len(recordings),
        "manifest": str(args.manifest),
        "supported_action_ids": list(supported_ids),
        "cases": {},
    }
    for case in ("policy", "shuffled_instructions", "black_images"):
        predictions = []
        for start in range(0, len(images), args.batch_size):
            observations = images[start : start + args.batch_size]
            if case == "black_images":
                observations = [{key: np.zeros_like(value) for key, value in row.items()} for row in observations]
            instructions = (
                [goals[index] for index in permutation] if case == "shuffled_instructions" else goals
            )[start : start + args.batch_size]
            predictions.append(policy.predict(observations, instructions))
        predicted = np.concatenate(predictions)
        report["cases"][case] = {
            "tick_0": metrics(predicted[:, :1], truth[:, :1], supported_ids),
            "chunk": metrics(predicted, truth, supported_ids),
        }
        print(case, report["cases"][case]["chunk"]["supported_macro_f1"], flush=True)

    audit = json.loads((args.bundle / "label_audit.json").read_text())
    counts = np.array([audit["splits"]["train"]["class_counts"][name] for name in D29_NAMES])
    majority = np.full_like(truth, counts.argmax())
    random = np.random.default_rng(42).choice(29, size=truth.shape, p=counts / counts.sum())
    for name, predicted in (
        ("train_majority", majority),
        ("random_train_prior", random),
        ("noop", np.zeros_like(truth)),
    ):
        report["cases"][name] = {
            "tick_0": metrics(predicted[:, :1], truth[:, :1], supported_ids),
            "chunk": metrics(predicted, truth, supported_ids),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Evaluation saved", args.output, flush=True)


if __name__ == "__main__":
    main()
