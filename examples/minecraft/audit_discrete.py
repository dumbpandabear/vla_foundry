"""Audit all D29 targets and camera quantization without decoding images."""

import argparse
import io
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vla_foundry.data.minecraft.actions import D29_NAMES
from vla_foundry.data.minecraft.discrete_actions import DISCRETE_SCHEMA, discretize, projected_camera


def validate_split(split, result, dataset_summary):
    expected_examples = int(dataset_summary["samples"][split])
    expected_ticks = expected_examples * int(dataset_summary["horizon"])
    if result["examples"] != expected_examples:
        raise ValueError(
            f"{split} audit/manifest mismatch: audited {result['examples']} examples, expected {expected_examples}"
        )
    if result["ticks"] != expected_ticks:
        raise ValueError(f"{split} tick mismatch: audited {result['ticks']}, expected {expected_ticks}")
    counted_ticks = sum(result["class_counts"].values())
    if counted_ticks != result["ticks"]:
        raise ValueError(f"{split} class-count mismatch: counted {counted_ticks}, audited {result['ticks']}")
    if not np.isfinite(result["camera_quantization_mae"]).all():
        raise ValueError(f"{split} camera audit is nonfinite")


def audit_shard(path):
    counts = np.zeros(29, dtype=np.int64)
    errors, ticks, examples, camera_active = np.zeros(2), 0, 0, 0
    with tarfile.open(path) as archive:
        for member in archive:
            if not member.name.endswith(".lowdim.npz"):
                continue
            with np.load(io.BytesIO(archive.extractfile(member).read())) as row:
                values = row["vpt_actions"]
                labels = discretize(values)
                counts += np.bincount(labels.reshape(-1), minlength=29)
                raw_camera = values[..., -2:] * 40.0
                errors += np.abs(projected_camera(labels) - raw_camera).sum(axis=0)
                camera_active += int((np.abs(raw_camera).max(axis=-1) > 0.01).sum())
                ticks += labels.size
                examples += 1
    return counts, errors, ticks, examples, camera_active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    dataset_summary = json.loads((args.dataset / "dataset.json").read_text())
    report = dict(schema=DISCRETE_SCHEMA, camera_delta=10.0, camera_threshold=5.0, splits={})
    with ProcessPoolExecutor(args.workers) as pool:
        for split in ("train", "val", "test"):
            manifest = args.dataset / split / "manifest.jsonl"
            paths = [
                manifest.parent / (json.loads(line)["shard"] + ".tar")
                for line in manifest.read_text().splitlines()
            ]
            counts, errors = np.zeros(29, dtype=np.int64), np.zeros(2)
            ticks = examples = camera_active = 0
            for c, e, t, n, active in pool.map(audit_shard, paths):
                counts += c
                errors += e
                ticks += t
                examples += n
                camera_active += active
            result = dict(
                examples=examples,
                ticks=ticks,
                class_counts=dict(zip(D29_NAMES, counts.tolist(), strict=True)),
                classes_without_targets=[
                    name for name, count in zip(D29_NAMES, counts, strict=True) if not count
                ],
                camera_quantization_mae=(errors / ticks).tolist(),
                raw_camera_active_ticks=camera_active,
            )
            validate_split(split, result, dataset_summary)
            report["splits"][split] = result
            print(split, "examples", examples, "ticks", ticks, "classes", int((counts > 0).sum()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
