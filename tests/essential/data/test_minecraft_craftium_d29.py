"""Tests for balanced native-D29 Craftium selection."""

import json

from vla_foundry.data.minecraft.actions import D29_NAMES
from vla_foundry.data.minecraft.prepare import recording_split
from vla_foundry.data.minecraft.prepare_craftium_d29 import (
    CANONICAL_ROOT,
    STALE_ROOT,
    canonical_video_path,
    select_balanced,
)


def _run_for_split(split, seed):
    index = 0
    while recording_split(f"run-{index}", seed) != split:
        index += 1
    return f"run-{index}"


def test_canonical_video_path_rewrites_only_stale_prefix():
    assert canonical_video_path(f"{STALE_ROOT}/clips/a.mp4") == f"{CANONICAL_ROOT}/clips/a.mp4"
    assert canonical_video_path("/another/root/a.mp4") == "/another/root/a.mp4"


def test_select_balanced_supports_all_29_native_actions(tmp_path):
    seed = 7
    source = tmp_path / "manifest.jsonl"
    with source.open("w") as stream:
        for split in ("train", "val", "test"):
            run_id = _run_for_split(split, seed)
            actions = [0, *range(29), *range(29)]
            stream.write(
                json.dumps(
                    {
                        "video": f"{STALE_ROOT}/{split}.mp4",
                        "run_id": run_id,
                        "text_goal": f"goal {split}",
                        "action_schema": "craftium_discrete29",
                        "canonical_action_space_n": 29,
                        "actions": actions,
                    }
                )
                + "\n"
            )

    clips, selected, counts, source_rows = select_balanced(
        [source], per_class=[1, 1, 1], horizon=2, stride=1, seed=seed
    )

    assert len(clips) == 3
    assert source_rows == {"train": 1, "val": 1, "test": 1}
    for split in ("train", "val", "test"):
        assert sum(len(anchors) for anchors in selected[split].values()) == len(D29_NAMES)
        assert set(counts[split]) == set(range(29))
