"""Check aggregation of final Craftium integration rollouts."""

import json

import pytest

from examples.minecraft.summarize_craftium_rollouts import summarize


def write_rollout(path, seed, steps=2, nonblank=True):
    path.mkdir()
    (path / "result.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "total_reward": 1.0,
                "seed": seed,
                "nonblank_observations": nonblank,
                "projection": "native categorical D29",
                "action_space": "discrete_29",
            }
        )
    )
    rows = [{"name": name} for name in ("nop", "forward")[:steps]]
    (path / "actions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_rollout_summary_verifies_steps_rendering_and_actions(tmp_path):
    paths = [tmp_path / "a", tmp_path / "b"]
    write_rollout(paths[0], 42)
    write_rollout(paths[1], 43)

    report = summarize(paths, expected_steps=2)

    assert report["verified"]
    assert report["total_steps"] == 4
    assert report["total_reward"] == 2.0
    assert report["action_counts"] == {"nop": 2, "forward": 2}


def test_rollout_summary_rejects_blank_observations(tmp_path):
    path = tmp_path / "blank"
    write_rollout(path, 42, nonblank=False)

    with pytest.raises(ValueError, match="Blank observations"):
        summarize([path], expected_steps=2)
