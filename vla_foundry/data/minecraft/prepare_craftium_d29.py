"""Convert native Craftium Discrete(29) clips into balanced Foundry shards.

The source manifests contain short, isolated diamond-tech demonstrations with
pre-action video frames and native integer action IDs. This converter resplits
by run ID, samples a bounded number of anchors per action class, and preserves
the native labels in a one-channel ``d29_actions`` field. No VPT projection or
synthetic crafting label is involved.
"""

import argparse
import hashlib
import heapq
import io
import json
import multiprocessing as mp
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import yaml

from vla_foundry.data.minecraft.actions import D29_NAMES
from vla_foundry.data.minecraft.prepare import add_bytes, recording_split

STALE_ROOT = "/checkpoint/jepa_core/koustuvs/experiments/game_data"
CANONICAL_ROOT = "/s3/profiles/scale-ai-gaming-data--use2-az3--x-s3/koustuvs/game_data"


@dataclass(frozen=True)
class Clip:
    video: str
    run_id: str
    goal: str
    actions: tuple[int, ...]


def canonical_video_path(path: str) -> str:
    if path == STALE_ROOT or path.startswith(STALE_ROOT + "/"):
        return CANONICAL_ROOT + path[len(STALE_ROOT) :]
    return path


def candidate_score(seed: int, clip: Clip, anchor: int) -> int:
    value = f"{seed}:{clip.run_id}:{clip.video}:{anchor}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def _retain_candidate(heap, limit, score, clip_id, anchor):
    item = (-score, clip_id, anchor)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score < -heap[0][0]:
        heapq.heapreplace(heap, item)


def select_balanced(sources, per_class, horizon, stride, seed):
    """Select deterministic lowest-hash anchors per split and target class."""
    limits = dict(zip(("train", "val", "test"), per_class, strict=True))
    heaps = {(split, action): [] for split in limits for action in range(len(D29_NAMES))}
    clips, seen = [], set()
    source_rows = Counter()
    for source in sources:
        with Path(source).open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row.get("action_schema") != "craftium_discrete29":
                    raise ValueError(f"Unexpected action schema at {source}:{line_number}")
                if int(row.get("canonical_action_space_n", -1)) != len(D29_NAMES):
                    raise ValueError(f"Unexpected action-space size at {source}:{line_number}")
                video = canonical_video_path(row["video"])
                if video in seen:
                    continue
                seen.add(video)
                actions = tuple(int(value) for value in row["actions"])
                if not actions or min(actions) < 0 or max(actions) >= len(D29_NAMES):
                    raise ValueError(f"Invalid D29 action at {source}:{line_number}")
                run_id = str(row["run_id"])
                clip = Clip(video=video, run_id=run_id, goal=row["text_goal"].strip(), actions=actions)
                if not clip.goal:
                    raise ValueError(f"Empty text goal at {source}:{line_number}")
                clip_id = len(clips)
                clips.append(clip)
                split = recording_split(run_id, seed)
                for anchor in range(1, len(actions) - horizon + 1, stride):
                    action = actions[anchor]
                    _retain_candidate(
                        heaps[(split, action)],
                        limits[split],
                        candidate_score(seed, clip, anchor),
                        clip_id,
                        anchor,
                    )
                source_rows[split] += 1
                if len(clips) % 1000 == 0:
                    print(f"scanned_clips={len(clips)} source_rows={dict(source_rows)}", flush=True)

    selected = defaultdict(lambda: defaultdict(set))
    class_counts = {split: Counter() for split in limits}
    for (split, action), heap in heaps.items():
        if len(heap) < limits[split]:
            raise ValueError(
                f"Insufficient {split} examples for {D29_NAMES[action]}: "
                f"requested {limits[split]}, found {len(heap)}"
            )
        for _, clip_id, anchor in heap:
            selected[split][clip_id].add(anchor)
            class_counts[split][action] += 1
    return clips, selected, class_counts, source_rows


def decode_clip(task):
    split, clip_id, clip, anchors, image_size, horizon = task
    wanted_frames = {index for anchor in anchors for index in (anchor - 1, anchor)}
    encoded = {}
    try:
        with av.open(clip.video) as container:
            fps = float(container.streams.video[0].average_rate)
            for index, frame in enumerate(container.decode(video=0)):
                if index not in wanted_frames:
                    continue
                image = io.BytesIO()
                frame.to_image().resize((image_size, image_size)).save(image, "JPEG", quality=90)
                encoded[index] = image.getvalue()
    except (av.FFmpegError, OSError, EOFError) as exc:
        return split, clip_id, [], {"type": type(exc).__name__, "message": str(exc)}

    missing = wanted_frames - encoded.keys()
    if missing:
        return split, clip_id, [], {"type": "MissingFrames", "message": str(sorted(missing))}
    samples = []
    for anchor in sorted(anchors):
        labels = np.asarray(clip.actions[anchor : anchor + horizon], dtype=np.float32)[:, None]
        if labels.shape != (horizon, 1):
            return split, clip_id, [], {"type": "ShortActions", "message": str(anchor)}
        lowdim = io.BytesIO()
        np.savez(
            lowdim,
            d29_actions=labels,
            past_mask=np.zeros(horizon, dtype=bool),
            future_mask=np.ones(horizon, dtype=bool),
        )
        metadata = {
            "run_id": clip.run_id,
            "video": clip.video,
            "anchor_frame": anchor,
            "anchor_relative_idx": 0,
            "action_schema": "craftium_discrete29",
            "fps": fps,
        }
        samples.append(
            {
                "anchor": anchor,
                "pov_t-1.jpg": encoded[anchor - 1],
                "pov_t0.jpg": encoded[anchor],
                "lowdim.npz": lowdim.getvalue(),
                "metadata.json": json.dumps(metadata).encode(),
                "language_instructions.json": json.dumps({"original": [clip.goal]}).encode(),
                "labels": labels[:, 0].astype(np.int64),
            }
        )
    return split, clip_id, samples, None


def _open_shard(output, split, shard_index):
    stem = f"craftium-d29-{split}-{shard_index:06d}"
    return stem, tarfile.open(output / split / f"{stem}.tar", "w")


def prepare(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    clips, selected, anchor_counts, source_rows = select_balanced(
        args.sources, args.per_class, args.horizon, args.stride, args.seed
    )
    output.mkdir(parents=True)
    for split in ("train", "val", "test"):
        (output / split).mkdir()

    tasks = [
        (split, clip_id, clips[clip_id], tuple(anchors), args.image_size, args.horizon)
        for split in ("train", "val", "test")
        for clip_id, anchors in sorted(selected[split].items())
    ]
    archives, current_stems = {}, {}
    manifests = {split: [] for split in ("train", "val", "test")}
    sample_counts, clip_counts, recording_sets = Counter(), Counter(), defaultdict(set)
    tick_counts = {split: Counter() for split in ("train", "val", "test")}
    failures = Counter()
    provenance = (output / "converted_clips.jsonl").open("w")
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            for number, (split, clip_id, samples, error) in enumerate(pool.map(decode_clip, tasks, chunksize=1), 1):
                clip = clips[clip_id]
                if error:
                    failures[error["type"]] += 1
                    provenance.write(json.dumps({"video": clip.video, "split": split, "error": error}) + "\n")
                    continue
                clip_counts[split] += 1
                recording_sets[split].add(clip.run_id)
                for sample in samples:
                    if sample_counts[split] % args.shard_samples == 0:
                        if split in archives:
                            archives[split].close()
                        stem, archives[split] = _open_shard(output, split, len(manifests[split]))
                        current_stems[split] = stem
                        manifests[split].append({"shard": stem, "num_sequences": 0})
                    digest = hashlib.sha256(f"{clip.video}:{sample['anchor']}".encode()).hexdigest()[:24]
                    for suffix in (
                        "pov_t-1.jpg",
                        "pov_t0.jpg",
                        "lowdim.npz",
                        "metadata.json",
                        "language_instructions.json",
                    ):
                        add_bytes(archives[split], f"{digest}.{suffix}", sample[suffix])
                    manifests[split][-1]["num_sequences"] += 1
                    sample_counts[split] += 1
                    tick_counts[split].update(int(value) for value in sample["labels"])
                provenance.write(
                    json.dumps({"video": clip.video, "run_id": clip.run_id, "split": split, "samples": len(samples)})
                    + "\n"
                )
                if number % 100 == 0:
                    provenance.flush()
                    print(f"decoded_clips={number}/{len(tasks)} samples={dict(sample_counts)}", flush=True)
    finally:
        provenance.close()
        for archive in archives.values():
            archive.close()

    for split in ("train", "val", "test"):
        if sample_counts[split] == 0:
            raise ValueError(f"No converted samples for {split}")
        with (output / split / "manifest.jsonl").open("w") as stream:
            for entry in manifests[split]:
                stream.write(json.dumps(entry) + "\n")
        (output / split / "stats.json").write_text(
            json.dumps({"d29_actions": {"mean": [0.0], "std": [1.0]}}) + "\n"
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

    class_counts = {
        split: {name: tick_counts[split][index] for index, name in enumerate(D29_NAMES)}
        for split in ("train", "val", "test")
    }
    audit = {
        "schema": "craftium_discrete29_native_v1",
        "actions": list(D29_NAMES),
        "splits": {
            split: {
                "samples": sample_counts[split],
                "ticks": sample_counts[split] * args.horizon,
                "class_counts": class_counts[split],
                "classes_without_targets": [name for name, count in class_counts[split].items() if count == 0],
                "balanced_anchor_counts": {
                    D29_NAMES[index]: anchor_counts[split][index] for index in range(len(D29_NAMES))
                },
            }
            for split in ("train", "val", "test")
        },
    }
    (output / "d29_label_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    summary = {
        "schema": "craftium_discrete29_native_v1",
        "action_dim": 1,
        "horizon": args.horizon,
        "image_size": args.image_size,
        "image_indices": [-1, 0],
        "samples": dict(sample_counts),
        "clips": dict(clip_counts),
        "recordings": {split: len(recording_sets[split]) for split in recording_sets},
        "source_rows": dict(source_rows),
        "failures": dict(failures),
        "split_method": "run_id hash: 80% train, 10% val, 10% test",
        "per_class_anchor_limit": dict(zip(("train", "val", "test"), args.per_class, strict=True)),
        "sources": list(args.sources),
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", nargs=3, type=int, default=[5000, 500, 500])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--shard-samples", type=int, default=512)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    if min(*arguments.per_class, arguments.workers, arguments.image_size, arguments.horizon, arguments.stride) <= 0:
        parser.error("Counts and sizes must be positive")
    prepare(arguments)
