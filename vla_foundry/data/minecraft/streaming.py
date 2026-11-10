"""Decode compact-indexed VPT clips directly from source storage."""

import json
from pathlib import Path

import av
import numpy as np

from vla_foundry.data.minecraft.raw_actions import load_raw_actions


def decode_recording_samples(row, data_root, image_size, horizon, selected_offsets=None):
    """Yield robotics samples while loading one raw action log per recording."""
    root = Path(data_root)
    actions, _, _, audit = load_raw_actions(root / row["raw_actions"])
    if int(row["horizon"]) != horizon:
        raise ValueError(f"Index/model horizon mismatch for {row['recording']}: {row['horizon']} != {horizon}")
    selected = None if selected_offsets is None else set(int(value) for value in selected_offsets)
    expected = int(row["num_sequences"]) if selected is None else len(selected)
    yielded = offset = 0
    for clip in row["clips"]:
        all_anchors = [int(value) for value in clip["anchors"]]
        anchors = (
            all_anchors
            if selected is None
            else [anchor for index, anchor in enumerate(all_anchors, offset) if index in selected]
        )
        offset += len(all_anchors)
        if not anchors:
            continue
        wanted = {index for anchor in anchors for index in (anchor - 1, anchor)}
        images = {}
        path = root / "splits" / clip["video"]
        with av.open(str(path)) as container:
            fps = float(container.streams.video[0].average_rate)
            if abs(fps - 20.0) > 0.1:
                raise ValueError(f"Expected 20 FPS, found {fps}: {path}")
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    images[index] = np.asarray(frame.to_image().resize((image_size, image_size))).copy()
        missing = wanted - images.keys()
        if missing:
            raise ValueError(f"Index references missing frames in {path}: {sorted(missing)}")
        tick_start = int(clip["chunk_index"]) * 64
        for anchor in anchors:
            first = tick_start + anchor
            target = actions[first : first + horizon]
            if target.shape != (horizon, actions.shape[-1]):
                raise ValueError(f"Short action target in {path} at {anchor}: {target.shape}")
            yielded += 1
            yield {
                "images": {"pov_t-1": images[anchor - 1], "pov_t0": images[anchor]},
                "lowdim": {"vpt_actions": target},
                "past_mask": np.zeros(horizon, dtype=bool),
                "future_mask": np.ones(horizon, dtype=bool),
                "metadata": {
                    "recording_id": row["recording"],
                    "video": str(path),
                    "anchor_frame": anchor,
                    "original_anchor_tick": first,
                    "anchor_relative_idx": 0,
                    "action_offset": 0,
                    "fps": fps,
                    "num_actions": len(actions),
                    "action_source": audit["raw_path"],
                },
                "intrinsics": {},
                "extrinsics": {},
                "language_instruction": clip["text_goal"],
                "language_instruction_full": {"original": [clip["text_goal"]]},
            }
    if yielded != expected:
        raise ValueError(
            f"Streaming index count mismatch for {row['recording']}: yielded={yielded}, "
            f"expected={expected}"
        )


def index_shards_to_samples(source, data_root, image_size, horizon):
    """WebDataset pipeline stage expanding compact recording-index shards."""
    for shard in source:
        url = shard["url"]
        with open(url, errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    yield from decode_recording_samples(row, data_root, image_size, horizon)
                except Exception as exc:
                    raise RuntimeError(f"Failed streaming {url}:{line_number}: {exc}") from exc


def iter_selected_index_samples(manifest, data_root, image_size, horizon, selected):
    """Decode only selected global sample offsets from a streaming manifest."""
    manifest = Path(manifest)
    selected = sorted(int(value) for value in selected)
    selected_index = global_offset = 0
    for manifest_line in manifest.read_text().splitlines():
        shard = json.loads(manifest_line)
        shard_path = manifest.parent / shard["shard"]
        with shard_path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                count = int(row["num_sequences"])
                end = global_offset + count
                offsets = []
                while selected_index < len(selected) and selected[selected_index] < end:
                    value = selected[selected_index]
                    if value >= global_offset:
                        offsets.append(value - global_offset)
                    selected_index += 1
                if offsets:
                    try:
                        yield from decode_recording_samples(
                            row, data_root, image_size, horizon, selected_offsets=offsets
                        )
                    except Exception as exc:
                        raise RuntimeError(f"Failed streaming {shard_path}:{line_number}: {exc}") from exc
                global_offset = end
                if selected_index == len(selected):
                    return
    if selected_index != len(selected):
        raise ValueError(f"Selected offsets exceed streaming dataset: {selected[selected_index:]}")
