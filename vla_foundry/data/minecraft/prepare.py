"""Convert annotated VPT clips into the native Foundry robotics shard format.

Reads video paths from JSONL rather than listing the S3 filesystem. Splits whole
source recordings deterministically so adjacent clips cannot leak across splits.
"""

import argparse
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import av
import numpy as np
import yaml

from vla_foundry.data.minecraft.actions import ACTION_DIM, BUTTONS, CAMERA_SCALE, SCHEMA_VERSION, encode_action


def recording_id(video: str) -> str:
    return Path(video).stem.rsplit("_chunk_", 1)[0]


def recording_split(recording: str, seed: int) -> str:
    fraction = int(hashlib.sha256(f"{seed}:{recording}".encode()).hexdigest()[:8], 16) / 2**32
    return "test" if fraction < 0.1 else "val" if fraction < 0.2 else "train"


def add_bytes(archive, name: str, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def prepare(args):
    root, output = Path(args.data_root), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    limits = dict(zip(("train", "val", "test"), args.max_clips, strict=True))
    counts, samples, reasons, per_recording = Counter(), Counter(), Counter(), Counter()
    seen, groups = set(), {s: set() for s in limits}
    archives, manifests, shard_counts = {}, {}, Counter()
    provenance = (output / "selected_clips.jsonl").open("w")
    try:
        for split in limits:
            (output / split).mkdir()
            manifests[split] = (output / split / "manifest.jsonl").open("w")
        done = False
        for source in args.sources:
            with Path(source).open() as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    video = row["video"]
                    key = Path(video).name
                    if key in seen:
                        continue
                    seen.add(key)
                    recording = recording_id(video)
                    split = recording_split(recording, args.seed)
                    if counts[split] >= limits[split] or per_recording[recording] >= args.max_per_recording:
                        continue
                    if not row.get("text_goal", "").strip():
                        reasons["empty_goal"] += 1
                        continue
                    actions = row.get("actions", [])
                    path = root / "splits" / key
                    if not path.is_file():
                        raise FileNotFoundError(f"{source}:{line_number}: missing referenced clip {path}")
                    with av.open(str(path)) as container:
                        video_stream = container.streams.video[0]
                        fps = float(video_stream.average_rate)
                        if abs(fps - 20.0) > 0.1:
                            raise ValueError(f"Expected 20 FPS, found {fps}: {path}")
                        frames = [frame.to_image() for frame in container.decode(video=0)]
                    if len(actions) > len(frames):
                        raise ValueError(f"More actions than decoded frames: {path}")
                    made = 0
                    for anchor in range(
                        0, min(len(frames), len(actions) - args.action_offset - args.horizon + 1), args.stride
                    ):
                        first = anchor + args.action_offset
                        target = actions[first : first + args.horizon]
                        if any(action.get("_is_null", False) for action in target):
                            reasons["null_window"] += 1
                            continue
                        vectors = np.stack([encode_action(action) for action in target])
                        shard_name = f"shard-{samples[split] // args.shard_samples:06d}.tar"
                        if samples[split] % args.shard_samples == 0:
                            if split in archives:
                                archives[split].close()
                            # Closed on rotation and in the enclosing finally block.
                            archives[split] = tarfile.open(output / split / shard_name, "w")  # noqa: SIM115
                            shard_counts[split] += 1
                        prefix = f"{hashlib.sha256(key.encode()).hexdigest()[:16]}_{anchor:04d}"
                        image = io.BytesIO()
                        frames[anchor].resize((args.image_size, args.image_size)).save(image, "JPEG", quality=90)
                        lowdim = io.BytesIO()
                        np.savez(
                            lowdim,
                            vpt_actions=vectors,
                            past_mask=np.zeros(args.horizon, dtype=bool),
                            future_mask=np.ones(args.horizon, dtype=bool),
                        )
                        metadata = {
                            "recording_id": recording,
                            "video": str(path),
                            "anchor_frame": anchor,
                            "anchor_relative_idx": 0,
                            "action_offset": args.action_offset,
                            "fps": fps,
                            "num_decoded_frames": len(frames),
                            "num_actions": len(actions),
                        }
                        for suffix, data in {
                            "pov_t0.jpg": image.getvalue(),
                            "lowdim.npz": lowdim.getvalue(),
                            "metadata.json": json.dumps(metadata).encode(),
                            "language_instructions.json": json.dumps({"original": [row["text_goal"]]}).encode(),
                        }.items():
                            add_bytes(archives[split], f"{prefix}.{suffix}", data)
                        samples[split] += 1
                        made += 1
                        if samples[split] % args.shard_samples == 0:
                            manifests[split].write(
                                json.dumps({"shard": Path(shard_name).stem, "num_sequences": args.shard_samples}) + "\n"
                            )
                    if made:
                        counts[split] += 1
                        per_recording[recording] += 1
                        groups[split].add(recording)
                        provenance.write(
                            json.dumps(
                                {
                                    "split": split,
                                    "recording": recording,
                                    "video": str(path),
                                    "source": source,
                                    "line": line_number,
                                    "samples": made,
                                }
                            )
                            + "\n"
                        )
                    if sum(counts.values()) % 25 == 0:
                        print(f"clips={dict(counts)} samples={dict(samples)}", flush=True)
                    if all(counts[s] >= limits[s] for s in limits):
                        done = True
                        break
            if done:
                break
        for split in limits:
            if samples[split] % args.shard_samples:
                manifests[split].write(
                    json.dumps(
                        {
                            "shard": f"shard-{shard_counts[split] - 1:06d}",
                            "num_sequences": samples[split] % args.shard_samples,
                        }
                    )
                    + "\n"
                )
            if counts[split] < limits[split]:
                raise ValueError(f"Not enough {split} clips: {counts[split]}/{limits[split]}")
            stats = {"vpt_actions": {"mean": [0.0] * ACTION_DIM, "std": [1.0] * ACTION_DIM}}
            (output / split / "stats.json").write_text(json.dumps(stats))
            (output / split / "preprocessing_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "camera_names": ["pov"],
                        "image_indices": [0],
                        "past_lowdim_steps": 0,
                        "future_lowdim_steps": args.horizon - 1,
                    }
                )
            )
        assert not (
            groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]
        )
        summary = {
            "schema": SCHEMA_VERSION,
            "buttons": list(BUTTONS),
            "camera_scale": CAMERA_SCALE,
            "action_dim": ACTION_DIM,
            "horizon": args.horizon,
            "action_offset": args.action_offset,
            "image_size": args.image_size,
            "seed": args.seed,
            "clips": dict(counts),
            "samples": dict(samples),
            "recordings": {s: len(g) for s, g in groups.items()},
            "skipped": dict(reasons),
            "split_method": "recording hash: 80% train, 10% val, 10% test",
            "note": "POC subset; original train/val labels regrouped to prevent recording overlap",
        }
        (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        provenance.close()
        for archive in archives.values():
            archive.close()
        for manifest in manifests.values():
            manifest.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-clips", nargs=3, type=int, default=[128, 32, 32], metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--max-per-recording", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--action-offset", type=int, default=1, help="Observe frame t, predict controls from t+offset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--shard-samples", type=int, default=128)
    args = parser.parse_args()
    if min(*args.max_clips, args.max_per_recording, args.horizon, args.stride, args.image_size, args.shard_samples) < 1:
        parser.error("Counts, horizon, stride, and sizes must be positive")
    if args.action_offset < 0:
        parser.error("action-offset must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
