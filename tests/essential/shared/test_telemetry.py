from __future__ import annotations

import dataclasses

import pytest

from vla_foundry.telemetry import get_or_create_wandb_run_id, resolve_wandb_mode


def test_auto_wandb_mode_falls_back_offline_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_wandb_mode("auto") == "offline"


def test_auto_wandb_mode_uses_online_with_api_key(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert resolve_wandb_mode("auto") == "online"


def test_wandb_run_id_persists_across_stages(tmp_path):
    first = get_or_create_wandb_run_id(tmp_path)
    second = get_or_create_wandb_run_id(tmp_path)
    assert first == second
    assert (tmp_path / ".wandb_run_id").read_text().strip() == first


def test_explicit_wandb_run_id_cannot_split_persisted_run(tmp_path):
    get_or_create_wandb_run_id(tmp_path, "first-stage")
    with pytest.raises(ValueError, match="differs from persisted ID"):
        get_or_create_wandb_run_id(tmp_path, "different-stage")


def test_training_config_remains_dataclass_serializable():
    from vla_foundry.params.train_experiment_params import TrainExperimentParams

    field_names = {field.name for field in dataclasses.fields(TrainExperimentParams)}
    assert {
        "wandb_mode",
        "wandb_run_id",
        "wandb_group",
        "wandb_job_type",
        "wandb_notes",
    } <= field_names
