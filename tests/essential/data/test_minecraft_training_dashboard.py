"""Tests for the self-contained Minecraft training dashboard artifact."""
# ruff: noqa: E501

import json
from argparse import Namespace
from pathlib import Path

from examples.minecraft.build_training_dashboard import (
    collect_snapshot,
    parse_training_log,
    write_dashboard,
)

LOG = """\
2026-11-10,10:00:00 | INFO | Start checkpoint 0
2026-11-10,10:01:00 | INFO | Train Checkpoint: 0 [  5120/188416 (3%)] Loss: 1.800 Data (t): 0.2 Batch (t): 8.0, 32.0000/s, 4.0/s/gpu LR: 0.000001
2026-11-10,10:02:00 | INFO | [VAL] ckpt=1 step=20 avg_loss=1.790000 samples=27904
2026-11-10,10:02:00 | INFO | Start checkpoint 1
2026-11-10,10:03:00 | INFO | Train Checkpoint: 1 [  5120/189440 (3%)] Loss: 1.600 Data (t): 0.2 Batch (t): 8.0, 31.5000/s, 4.0/s/gpu LR: 0.000002
2026-11-10,10:04:00 | INFO | [VAL] ckpt=2 step=40 avg_loss=1.750000 samples=27904
2026-11-10,10:04:00 | INFO | Start checkpoint 2
2026-11-10,10:05:00 | INFO | Train Checkpoint: 2 [  2560/190208 (1%)] Loss: 1.500 Data (t): 0.2 Batch (t): 8.0, 31.0000/s, 4.0/s/gpu LR: 0.000003
"""


def test_parse_training_log_reports_learning_progress_and_eta():
    result = parse_training_log(LOG, global_batch_size=256, total_steps=100)

    assert result["learning"]
    assert result["current_step"] == 50
    assert result["progress"] == 0.5
    assert result["best"]["checkpoint"] == 2
    assert result["improvement_percent"] > 2
    assert result["current_report"]["throughput"] == 31.0
    assert result["eta_hours"] is not None


def test_write_dashboard_embeds_snapshot_and_live_status(tmp_path: Path):
    snapshot = {"title": "Minecraft <D29>", "training": {"current_step": 42}}
    write_dashboard(snapshot, tmp_path)

    html = (tmp_path / "index.html").read_text()
    assert "status.json" in html
    assert "Minecraft <D29>" in html
    assert "ASTRYX core token mapping" in html
    assert "--blue:#0171E3" in html
    assert "Training operations" in html
    assert "Optimizer step" in html
    assert "Cross-entropy loss" in html
    assert "Training loss" in html
    assert "Dataset composition" in html
    assert "coverage-ring" in html
    assert "#9dfc6d" not in html
    assert "#07110d" not in html
    assert json.loads((tmp_path / "status.json").read_text()) == snapshot


def test_collect_snapshot_reads_artifacts_without_slurm(tmp_path: Path):
    root = tmp_path / "root"
    run = root / "runs" / "minecraft-qwen3-d29-full"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "out.log").write_text(LOG)
    (checkpoints / "checkpoint_1.pt").write_bytes(b"model")
    (checkpoints / "optimizer_1.pt").write_bytes(b"optimizer")
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"samples": {"train": 80, "val": 10, "test": 10}}))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "class_counts": {"forward": 10, "craft": 0},
                        "classes_without_targets": ["craft"],
                    }
                }
            }
        )
    )
    args = Namespace(
        root=root,
        run_dir=run,
        dataset_json=dataset,
        audit_json=audit,
        total_steps=100,
        global_batch_size=256,
        source_hours=1.5,
        job_chain=["1"],
        query_slurm=False,
    )

    snapshot = collect_snapshot(args)

    assert snapshot["training"]["current_step"] == 50
    assert snapshot["dataset"]["total_samples"] == 100
    assert snapshot["dataset"]["supported_actions"] == 1
    assert snapshot["checkpoints"][0]["optimizer"] == "optimizer_1.pt"
    assert snapshot["pipeline"]["training"]["state"] == "unknown"
