"""Verify a compact Minecraft VPT streaming index and decode representative samples."""

import argparse
import json
from pathlib import Path

from vla_foundry.data.minecraft.actions import ACTION_DIM, D29_NAMES
from vla_foundry.data.minecraft.discrete_actions import DISCRETE_SCHEMA
from vla_foundry.data.minecraft.streaming import decode_recording_samples


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_stream_index(index_root, data_root, decode_per_split=1):
    """Check manifest/audit invariants and decode the first indexed recordings."""
    index_root = Path(index_root)
    summary = load_json(index_root / "dataset.json")
    audit = load_json(index_root / "d29_label_audit.json")
    if summary.get("schema") != "minecraft_vpt_stream_index_v1":
        raise ValueError(f"Unexpected streaming index schema: {summary.get('schema')}")
    if int(summary.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError(f"Unexpected action dimension: {summary.get('action_dim')}")
    if audit.get("schema") != DISCRETE_SCHEMA:
        raise ValueError(f"Unexpected D29 audit schema: {audit.get('schema')}")

    result = {"schema": summary["schema"], "splits": {}}
    for split in ("train", "val", "test"):
        split_root = index_root / split
        manifest_path = split_root / "manifest.jsonl"
        manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        manifest_samples = sum(int(row["num_sequences"]) for row in manifest)
        expected_samples = int(summary["samples"][split])
        if manifest_samples != expected_samples:
            raise ValueError(
                f"{split} manifest count mismatch: manifest={manifest_samples}, summary={expected_samples}"
            )
        class_counts = audit["splits"][split]["class_counts"]
        if list(class_counts) != list(D29_NAMES):
            raise ValueError(f"{split} audit action order does not match D29")
        expected_ticks = expected_samples * int(summary["horizon"])
        if sum(int(value) for value in class_counts.values()) != expected_ticks:
            raise ValueError(f"{split} audit target count does not match {expected_ticks}")

        decoded = 0
        checked_rows = 0
        for shard in manifest:
            shard_path = split_root / shard["shard"]
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            with shard_path.open(errors="replace") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            shard_samples = 0
            for row in rows:
                row_samples = int(row["num_sequences"])
                clip_samples = sum(len(clip["anchors"]) for clip in row["clips"])
                if row_samples != clip_samples:
                    raise ValueError(
                        f"{shard_path}: {row['recording']} count mismatch: {row_samples} != {clip_samples}"
                    )
                shard_samples += row_samples
                checked_rows += 1
                if decoded < decode_per_split:
                    sample = next(
                        decode_recording_samples(
                            row,
                            data_root=data_root,
                            image_size=int(summary["image_size"]),
                            horizon=int(summary["horizon"]),
                            selected_offsets=[0],
                        )
                    )
                    if sample["lowdim"]["vpt_actions"].shape != (int(summary["horizon"]), ACTION_DIM):
                        raise ValueError(f"{split} decoded action shape is invalid")
                    expected_image_shape = (int(summary["image_size"]), int(summary["image_size"]), 3)
                    if any(image.shape != expected_image_shape for image in sample["images"].values()):
                        raise ValueError(f"{split} decoded image shape is invalid")
                    decoded += 1
            if shard_samples != int(shard["num_sequences"]):
                raise ValueError(
                    f"{shard_path} count mismatch: rows={shard_samples}, manifest={shard['num_sequences']}"
                )
            if decoded >= decode_per_split:
                break
        if decoded != decode_per_split:
            raise ValueError(f"{split} decoded {decoded} samples, expected {decode_per_split}")
        result["splits"][split] = {
            "samples": expected_samples,
            "manifest_shards": len(manifest),
            "checked_rows": checked_rows,
            "decoded_samples": decoded,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--decode-per-split", type=int, default=1)
    args = parser.parse_args()
    if args.decode_per_split <= 0:
        parser.error("--decode-per-split must be positive")
    print(
        json.dumps(
            verify_stream_index(args.index, args.data_root, args.decode_per_split),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
