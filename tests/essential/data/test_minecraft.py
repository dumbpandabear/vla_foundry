"""Check action semantics, input normalization, and prepared shard integrity."""

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_foundry.data.minecraft.actions import BUTTONS, decode_action, encode_action, to_discrete29
from vla_foundry.data.minecraft.prepare import recording_id, recording_split
from vla_foundry.data.minecraft.prepare_vla import (
    convert_recording,
    convert_tasks_in_recycled_pools,
    load_resume_state,
    select_clips,
)
from vla_foundry.data.minecraft.processor import MinecraftByteProcessor
from vla_foundry.data.minecraft.raw_actions import load_raw_actions


def test_vpt_roundtrip_preserves_all_controls_and_camera_axes():
    action = {key: i % 2 for i, key in enumerate(BUTTONS)}
    action["camera"] = [-90, 45]
    restored = decode_action(encode_action(action))
    assert restored == action


@pytest.mark.parametrize("camera", ([0, 181], [float("nan"), 0], [1]))
def test_invalid_camera_is_rejected(camera):
    with pytest.raises(ValueError):
        encode_action({"camera": camera})


def test_d29_reports_controls_that_cannot_be_executed_together():
    index, diagnostics = to_discrete29({"forward": 1, "attack": 1, "back": 1, "camera": [0, 0]})
    assert index == 3
    assert diagnostics["name"] == "dig"
    assert diagnostics["dropped_controls"] == ["backward", "forward"]
    index, diagnostics = to_discrete29({"forward": 1, "jump": 1, "attack": 1, "camera": [0, 0]})
    assert index == 28
    assert diagnostics["dropped_controls"] == []


def test_adjacent_clips_share_recording_split():
    a = recording_id("/splits/example-session_chunk_0001.mp4")
    b = recording_id("/splits/example-session_chunk_0042.mp4")
    assert a == b
    assert recording_split(a, 42) == recording_split(b, 42)


def test_full_vla_selection_ignores_subset_caps(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"video": f"same-recording_chunk_{index:04d}.mp4", "text_goal": "Mine stone"}
        for index in range(3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    args = SimpleNamespace(
        sources=[source], seed=42, max_clips=[1, 1, 1], max_per_recording=1, all_clips=True
    )

    groups, counts = select_clips(args)

    assert sum(counts.values()) == 3
    assert sum(len(group) for group in groups.values()) == 3


def test_full_vla_resume_reconstructs_and_validates_completed_prefix(tmp_path):
    groups = {
        "a-recording": [{"split": "train"}, {"split": "train"}],
        "b-recording": [{"split": "val"}],
    }
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "manifest.jsonl").write_text(
            json.dumps({"shard": "a", "num_sequences": 5}) + "\n" if split == "train" else ""
        )
    (tmp_path / "selected_clips.jsonl").write_text(
        json.dumps({"recording": "a-recording", "samples": 2})
        + "\n"
        + json.dumps({"recording": "a-recording", "samples": 3})
        + "\n"
    )
    (tmp_path / "raw_actions_audit.jsonl").write_text("{}\n")

    completed, samples, clips, recordings, skips = load_resume_state(tmp_path, groups)

    assert completed == {"a-recording"}
    assert samples == {"train": 5}
    assert clips == {"train": 2}
    assert recordings == {"train": 1}
    assert skips == {"resume_prior_recordings_without_skip_counters": 1}


def test_full_vla_resume_rejects_manifest_mismatch(tmp_path):
    groups = {"a-recording": [{"split": "train"}]}
    for split in ("train", "val", "test"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "manifest.jsonl").write_text("")
    (tmp_path / "selected_clips.jsonl").write_text(
        json.dumps({"recording": "a-recording", "samples": 1}) + "\n"
    )
    (tmp_path / "raw_actions_audit.jsonl").write_text("{}\n")

    with pytest.raises(ValueError, match="manifest samples"):
        load_resume_state(tmp_path, groups)


def test_full_vla_records_decode_error_in_worker(monkeypatch, tmp_path):
    import vla_foundry.data.minecraft.prepare_vla as module

    actions = np.zeros((64, 25), dtype=np.float32)
    monkeypatch.setattr(
        module,
        "load_raw_actions",
        lambda _path: (actions, np.zeros(64, dtype=bool), np.zeros(64, dtype=bool), {"raw_path": "raw"}),
    )
    monkeypatch.setattr(module, "raw_path", lambda _root, _recording: tmp_path / "raw.jsonl")
    monkeypatch.setattr(module.av, "open", lambda _path: (_ for _ in ()).throw(OSError("corrupt video")))
    (tmp_path / "train").mkdir()
    row = {"video": "a-recording_chunk_0000.mp4", "chunk_index": 0, "split": "train", "text_goal": "Mine"}

    result = convert_recording((tmp_path, tmp_path, "a-recording", [row], 16, 15, 4, 512))

    assert result["samples"] == 0
    assert result["skipped"] == {"decode_error": 1}
    assert result["provenance"][0]["error"]["type"] == "OSError"


def test_full_vla_recycles_whole_worker_pools_without_dropping_tasks(monkeypatch):
    import vla_foundry.data.minecraft.prepare_vla as module

    batch_lengths = []

    class FakePool:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def map(self, _function, batch, chunksize):
            assert chunksize == 1
            batch_lengths.append(len(batch))
            return iter(batch)

    monkeypatch.setattr(module, "ProcessPoolExecutor", FakePool)

    assert list(convert_tasks_in_recycled_pools(list(range(10)), workers=2, tasks_per_worker=3)) == list(range(10))
    assert batch_lengths == [6, 4]


def test_raw_actions_fall_back_to_number_keys_when_hotbar_state_is_missing(tmp_path):
    path = tmp_path / "actions.jsonl"
    base = {
        "mouse": {"buttons": [], "dx": 0, "dy": 0},
        "keyboard": {"keys": []},
        "pitch": 0,
        "yaw": 0,
    }
    rows = [
        {**base, "hotbar": 0},
        {**base, "keyboard": {"keys": ["key.keyboard.2"]}},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    actions, _, _, audit = load_raw_actions(path)

    assert actions[1, BUTTONS.index("hotbar.2")] == 1
    assert audit["missing_hotbar_ticks"] == 1


def test_processor_uint8_normalization_and_goal_tokens():
    processor = MinecraftByteProcessor(28, 4)
    inputs = processor(
        images=[[np.zeros((28, 28, 3), dtype=np.uint8)], [np.full((28, 28, 3), 255, dtype=np.uint8)]],
        text=["<image> Mine", "<image> Walk"],
        max_length=12,
    )
    assert torch.all(inputs["pixel_values"][0] == -1)
    assert torch.all(inputs["pixel_values"][1] == 1)
    assert torch.all(inputs["input_ids"][:, :4] == 1)
    assert not torch.equal(inputs["input_ids"][0], inputs["input_ids"][1])


def test_poc_shards_match_manifest_counts_and_recording_split():
    root = Path("/storage/home/ericli11/minecraft-vla-poc/dataset-v1")
    if not root.exists():
        pytest.skip("Local Minecraft POC data is not available")
    groups = {split: set() for split in ("train", "val", "test")}
    for split in groups:
        for line in (root / split / "manifest.jsonl").read_text().splitlines():
            record = json.loads(line)
            with tarfile.open(root / split / (record["shard"] + ".tar")) as archive:
                samples = 0
                for member in archive:
                    if member.name.endswith("metadata.json"):
                        metadata = json.load(archive.extractfile(member))
                        groups[split].add(metadata["recording_id"])
                        assert recording_split(metadata["recording_id"], 42) == split
                        assert metadata["anchor_frame"] + metadata["action_offset"] + 8 <= metadata["num_actions"]
                        samples += 1
                    elif member.name.endswith("lowdim.npz"):
                        with np.load(io.BytesIO(archive.extractfile(member).read())) as arrays:
                            assert arrays["vpt_actions"].shape == (8, 25)
                            assert np.isfinite(arrays["vpt_actions"]).all()
                            assert arrays["future_mask"].all()
                assert samples == record["num_sequences"]
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
