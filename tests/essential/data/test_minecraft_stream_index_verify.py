"""Tests for the compact VPT stream-index verifier."""

import json

import numpy as np

from vla_foundry.data.minecraft.actions import D29_NAMES
from vla_foundry.data.minecraft.discrete_actions import DISCRETE_SCHEMA
from vla_foundry.data.minecraft.verify_stream_index import verify_stream_index


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def test_verify_stream_index_checks_counts_and_decodes_each_split(monkeypatch, tmp_path):
    horizon, image_size = 15, 8
    samples = {split: 2 for split in ("train", "val", "test")}
    write_json(
        tmp_path / "dataset.json",
        {
            "schema": "minecraft_vpt_stream_index_v1",
            "action_dim": 25,
            "horizon": horizon,
            "image_size": image_size,
            "samples": samples,
        },
    )
    counts = {name: 0 for name in D29_NAMES}
    counts[D29_NAMES[0]] = 2 * horizon
    write_json(
        tmp_path / "d29_label_audit.json",
        {
            "schema": DISCRETE_SCHEMA,
            "splits": {split: {"class_counts": counts} for split in samples},
        },
    )
    for split in samples:
        root = tmp_path / split
        root.mkdir()
        write_json(root / "manifest.jsonl", {"shard": "index.jsonl", "num_sequences": 2})
        write_json(
            root / "index.jsonl",
            {
                "recording": split,
                "horizon": horizon,
                "num_sequences": 2,
                "clips": [{"anchors": [4, 8]}],
            },
        )

    decoded = []

    def fake_decode(row, data_root, image_size, horizon, selected_offsets):
        decoded.append((row["recording"], data_root, selected_offsets))
        yield {
            "images": {
                "pov_t-1": np.zeros((image_size, image_size, 3), dtype=np.uint8),
                "pov_t0": np.zeros((image_size, image_size, 3), dtype=np.uint8),
            },
            "lowdim": {"vpt_actions": np.zeros((horizon, 25), dtype=np.float32)},
        }

    monkeypatch.setattr(
        "vla_foundry.data.minecraft.verify_stream_index.decode_recording_samples",
        fake_decode,
    )

    result = verify_stream_index(tmp_path, "/raw-vpt")

    assert result["splits"]["train"]["samples"] == 2
    assert [item[0] for item in decoded] == ["train", "val", "test"]
    assert all(item[1:] == ("/raw-vpt", [0]) for item in decoded)
