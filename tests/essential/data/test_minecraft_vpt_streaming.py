"""Check compact-indexed, on-demand VPT decoding."""

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from vla_foundry.data.dataloader import get_datastring_input
from vla_foundry.data.minecraft import build_stream_index
from vla_foundry.data.minecraft.streaming import decode_recording_samples, iter_selected_index_samples


def raw_actions(ticks=64):
    actions = np.full((ticks, 25), -1, dtype=np.float32)
    actions[:, -2:] = 0
    actions[:, 3] = 1
    return actions


def test_index_recording_stores_valid_anchors_and_exact_target_counts(monkeypatch, tmp_path):
    actions = raw_actions()
    gui = np.zeros(len(actions), dtype=bool)
    noop = np.zeros(len(actions), dtype=bool)
    gui[8] = True
    noop[4] = True
    path = tmp_path / "data" / "1.0" / "recording.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    monkeypatch.setattr(build_stream_index, "raw_path", lambda _root, _recording: path)
    monkeypatch.setattr(
        build_stream_index,
        "load_raw_actions",
        lambda _path: (actions, gui, noop, {"raw_path": str(path)}),
    )
    clips = [
        {
            "video": "recording_chunk_0000.mp4",
            "text_goal": "Mine stone",
            "chunk_index": 0,
            "num_frames": 40,
            "split": "train",
        }
    ]

    row, counts, _errors, _camera_active = build_stream_index.index_recording(
        tmp_path, "recording", clips, horizon=15, stride=4
    )

    assert row["clips"][0]["anchors"] == [12, 16, 20, 24]
    assert row["num_sequences"] == 4
    assert counts.sum() == 4 * 15


class FakeContainer:
    def __init__(self, frames):
        self.frames = frames
        self.streams = SimpleNamespace(video=[SimpleNamespace(average_rate=20)])

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def decode(self, video):
        assert video == 0
        for frame in self.frames:
            yield SimpleNamespace(to_image=lambda frame=frame: frame)


def test_stream_decoder_reads_one_action_log_and_selected_frames(monkeypatch, tmp_path):
    actions = raw_actions()
    calls = []
    monkeypatch.setattr(
        "vla_foundry.data.minecraft.streaming.load_raw_actions",
        lambda path: (calls.append(path) or actions, None, None, {"raw_path": str(path)}),
    )
    frames = [Image.fromarray(np.full((12, 16, 3), index, dtype=np.uint8)) for index in range(32)]
    monkeypatch.setattr(
        "vla_foundry.data.minecraft.streaming.av.open", lambda _path: FakeContainer(frames)
    )
    row = {
        "recording": "recording",
        "raw_actions": "data/1.0/recording.jsonl",
        "horizon": 15,
        "num_sequences": 3,
        "clips": [
            {
                "video": "recording_chunk_0000.mp4",
                "text_goal": "Mine stone",
                "chunk_index": 0,
                "anchors": [4, 8, 12],
                "num_sequences": 3,
            }
        ],
    }

    samples = list(decode_recording_samples(row, tmp_path, image_size=8, horizon=15, selected_offsets=[1]))

    assert len(samples) == 1
    assert len(calls) == 1
    assert samples[0]["lowdim"]["vpt_actions"].shape == (15, 25)
    assert samples[0]["images"]["pov_t0"].shape == (8, 8, 3)
    assert samples[0]["metadata"]["anchor_frame"] == 8


def test_selected_index_iterator_skips_unselected_recordings(monkeypatch, tmp_path):
    split = tmp_path / "test"
    split.mkdir()
    rows = [
        {"recording": "a", "num_sequences": 3},
        {"recording": "b", "num_sequences": 4},
    ]
    (split / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (split / "manifest.jsonl").write_text(json.dumps({"shard": "index.jsonl", "num_sequences": 7}) + "\n")
    decoded = []

    def fake_decode(row, _root, _size, _horizon, selected_offsets=None):
        decoded.append((row["recording"], selected_offsets))
        for offset in selected_offsets:
            yield {"recording": row["recording"], "offset": offset}

    monkeypatch.setattr("vla_foundry.data.minecraft.streaming.decode_recording_samples", fake_decode)

    samples = list(iter_selected_index_samples(split / "manifest.jsonl", tmp_path, 8, 15, {4, 6}))

    assert samples == [{"recording": "b", "offset": 1}, {"recording": "b", "offset": 3}]
    assert decoded == [("b", [1, 3])]


def test_datastring_preserves_explicit_stream_index_extension(monkeypatch):
    manifest = [
        {"shard": "index-0.jsonl", "num_sequences": 10},
        {"shard": "index-1.jsonl", "num_sequences": 10},
    ]
    monkeypatch.setattr(
        "vla_foundry.data.dataloader.load_dataset_manifest",
        lambda _path, shard_shuffle_seed=None: manifest,
    )

    datastrings, *_ = get_datastring_input(
        num_samples=20,
        curr_shard_idx_per_dataset=[0],
        shard_shuffle_seed_per_dataset=[None],
        manifest_paths=["/index/train/manifest.jsonl"],
        dataset_weighting=[1.0],
        allow_multiple_epochs=False,
        num_workers_per_gpu=1,
        world_size=1,
    )

    assert datastrings == ["/index/train/{index-0.jsonl,index-1.jsonl}"]
