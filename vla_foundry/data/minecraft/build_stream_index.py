"""Build a compact index for training directly from raw VPT clips.

The index stores goals and valid anchor offsets only. Images and action arrays
remain in the source corpus and are decoded by the training dataloader.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from vla_foundry.data.minecraft.actions import ACTION_DIM, BUTTONS, D29_NAMES
from vla_foundry.data.minecraft.discrete_actions import DISCRETE_SCHEMA, discretize, projected_camera
from vla_foundry.data.minecraft.prepare import recording_id, recording_split
from vla_foundry.data.minecraft.raw_actions import RAW_CAMERA_SCALE, RAW_SCHEMA, load_raw_actions, raw_path


def collect_clips(sources, seed):
    groups, seen, split_clips = defaultdict(list), set(), Counter()
    for source in sources:
        with Path(source).open(errors="replace") as stream:
            for line in stream:
                row = json.loads(line)
                name = Path(row["video"]).name
                if name in seen or not row.get("text_goal", "").strip():
                    continue
                seen.add(name)
                recording = recording_id(name)
                split = recording_split(recording, seed)
                groups[recording].append(
                    {
                        "video": name,
                        "text_goal": row["text_goal"],
                        "chunk_index": int(Path(name).stem.rsplit("_chunk_", 1)[1]),
                        "num_frames": int(row["num_frames"]),
                        "split": split,
                    }
                )
                split_clips[split] += 1
                if len(seen) % 10000 == 0:
                    print(f"indexed_annotations={len(seen)} recordings={len(groups)}", flush=True)
    return groups, split_clips


def index_recording(root, recording, clips, horizon, stride):
    action_path = raw_path(root, recording)
    actions, gui, noop, _ = load_raw_actions(action_path)
    labels = discretize(actions)
    coverage = np.zeros(len(actions) + 1, dtype=np.int64)
    indexed_clips, samples = [], 0
    for clip in sorted(clips, key=lambda value: value["chunk_index"]):
        tick_start = clip["chunk_index"] * 64
        usable_frames = min(clip["num_frames"], max(0, len(actions) - tick_start))
        anchors = []
        for anchor in range(stride, usable_frames - horizon + 1, stride):
            first = tick_start + anchor
            if gui[first : first + horizon].any() or noop[first]:
                continue
            anchors.append(anchor)
            coverage[first] += 1
            coverage[first + horizon] -= 1
        if anchors:
            indexed_clips.append(
                {
                    "video": clip["video"],
                    "text_goal": clip["text_goal"],
                    "chunk_index": clip["chunk_index"],
                    "anchors": anchors,
                    "num_sequences": len(anchors),
                }
            )
            samples += len(anchors)
    weights = np.cumsum(coverage[:-1])
    counts = np.bincount(labels, weights=weights, minlength=len(D29_NAMES)).astype(np.int64)
    raw_camera = actions[:, -2:] * RAW_CAMERA_SCALE
    errors = (np.abs(projected_camera(labels) - raw_camera) * weights[:, None]).sum(axis=0)
    camera_active = int(((np.abs(raw_camera).max(axis=-1) > 0.01) * weights).sum())
    return (
        {
            "recording": recording,
            "raw_actions": str(action_path.relative_to(root)),
            "horizon": horizon,
            "num_sequences": samples,
            "clips": indexed_clips,
        },
        counts,
        errors,
        camera_active,
    )


def write_index(args):
    root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    groups, source_clips = collect_clips(args.sources, args.seed)
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (output / split).mkdir()

    manifests = {split: [] for split in ("train", "val", "test")}
    streams, shard_rows, shard_samples = {}, Counter(), Counter()
    samples, clips, recordings = Counter(), Counter(), Counter()
    counts = {split: np.zeros(len(D29_NAMES), dtype=np.int64) for split in manifests}
    errors = {split: np.zeros(2, dtype=np.float64) for split in manifests}
    camera_active = Counter()

    def close_shard(split):
        stream = streams.pop(split, None)
        if stream is None:
            return
        stream.close()
        manifests[split][-1]["num_sequences"] = shard_samples[split]
        shard_rows[split] = 0
        shard_samples[split] = 0

    try:
        for number, recording in enumerate(sorted(groups), 1):
            split = groups[recording][0]["split"]
            row, row_counts, row_errors, row_camera_active = index_recording(
                root, recording, groups[recording], args.horizon, args.stride
            )
            if not row["num_sequences"]:
                continue
            if split not in streams:
                name = f"vpt-index-{split}-{len(manifests[split]):06d}.jsonl"
                streams[split] = (output / split / name).open("w")
                manifests[split].append({"shard": name, "num_sequences": 0})
            streams[split].write(json.dumps(row, separators=(",", ":")) + "\n")
            shard_rows[split] += 1
            shard_samples[split] += row["num_sequences"]
            samples[split] += row["num_sequences"]
            clips[split] += len(row["clips"])
            recordings[split] += 1
            counts[split] += row_counts
            errors[split] += row_errors
            camera_active[split] += row_camera_active
            if shard_rows[split] >= args.recordings_per_shard:
                close_shard(split)
            if number % 100 == 0:
                print(f"indexed_recordings={number}/{len(groups)} samples={dict(samples)}", flush=True)
    finally:
        for split in tuple(streams):
            close_shard(split)

    for split in manifests:
        if not manifests[split] or samples[split] <= 0:
            raise ValueError(f"Empty streaming index split: {split}")
        with (output / split / "manifest.jsonl").open("w") as stream:
            for row in manifests[split]:
                stream.write(json.dumps(row) + "\n")
        (output / split / "stats.json").write_text(
            json.dumps({"vpt_actions": {"mean": [0.0] * ACTION_DIM, "std": [1.0] * ACTION_DIM}}) + "\n"
        )
        (output / split / "preprocessing_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "camera_names": ["pov"],
                    "image_indices": [-1, 0],
                    "past_lowdim_steps": 0,
                    "future_lowdim_steps": args.horizon - 1,
                }
            )
        )

    summary = {
        "schema": "minecraft_vpt_stream_index_v1",
        "source_schema": RAW_SCHEMA,
        "buttons": list(BUTTONS),
        "action_dim": ACTION_DIM,
        "camera_scale": RAW_CAMERA_SCALE,
        "horizon": args.horizon,
        "stride": args.stride,
        "image_size": args.image_size,
        "image_indices": [-1, 0],
        "seed": args.seed,
        "samples": dict(samples),
        "clips": dict(clips),
        "recordings": dict(recordings),
        "source_clips": dict(source_clips),
        "index_shards": {split: len(rows) for split, rows in manifests.items()},
        "data_root": str(root),
        "sources": list(args.sources),
        "split_method": "recording hash: 80% train, 10% val, 10% test",
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = {"schema": DISCRETE_SCHEMA, "camera_delta": 10.0, "camera_threshold": 5.0, "splits": {}}
    for split in manifests:
        ticks = samples[split] * args.horizon
        if int(counts[split].sum()) != ticks:
            raise ValueError(f"{split} target count mismatch: {counts[split].sum()} != {ticks}")
        report["splits"][split] = {
            "examples": samples[split],
            "ticks": ticks,
            "class_counts": dict(zip(D29_NAMES, counts[split].tolist(), strict=True)),
            "classes_without_targets": [
                name for name, count in zip(D29_NAMES, counts[split], strict=True) if not count
            ],
            "camera_quantization_mae": (errors[split] / ticks).tolist(),
            "raw_camera_active_ticks": camera_active[split],
        }
    (output / "d29_label_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    # Keep enough shards for 96-way DDP across many checkpoint windows. Two
    # recordings per shard gives roughly two shards/rank/window on the full
    # corpus while keeping the index compact.
    parser.add_argument("--recordings-per-shard", type=int, default=2)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    if min(arguments.horizon, arguments.stride, arguments.image_size, arguments.recordings_per_shard) <= 0:
        parser.error("Counts and sizes must be positive")
    write_index(arguments)
