"""Weights & Biases lifecycle helpers for resumable cluster training."""

from __future__ import annotations

import dataclasses
import logging
import netrc
import os
from pathlib import Path
from typing import Any


def has_wandb_credentials() -> bool:
    """Return whether non-interactive W&B authentication is available."""
    if os.getenv("WANDB_API_KEY"):
        return True
    try:
        return netrc.netrc().authenticators("api.wandb.ai") is not None
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False


def resolve_wandb_mode(requested: str) -> str:
    """Resolve ``auto`` without ever prompting inside an unattended job."""
    if requested != "auto":
        return requested
    return "online" if has_wandb_credentials() else "offline"


def get_or_create_wandb_run_id(experiment_path: str | Path, explicit_id: str | None = None) -> str:
    """Persist one run ID so dependency-chained Slurm stages resume one run."""
    import wandb

    run_id_path = Path(experiment_path) / ".wandb_run_id"
    persisted_id = run_id_path.read_text().strip() if run_id_path.exists() else None
    if explicit_id and persisted_id and explicit_id != persisted_id:
        raise ValueError(
            f"Configured W&B run ID {explicit_id!r} differs from persisted ID {persisted_id!r} in {run_id_path}"
        )
    run_id = explicit_id or persisted_id or wandb.util.generate_id()
    if not persisted_id:
        run_id_path.write_text(run_id + "\n")
        run_id_path.chmod(0o600)
    return run_id


def initialize_wandb(
    cfg: Any,
    *,
    experiment_name: str,
    experiment_path: str | Path,
    global_step: int,
    checkpoint_num: int,
    total_steps: int,
    total_parameters: int,
    trainable_parameters: int,
):
    """Initialize a resumable W&B run and declare its metric schema."""
    import wandb

    mode = resolve_wandb_mode(cfg.wandb_mode)
    run_id = get_or_create_wandb_run_id(experiment_path, cfg.wandb_run_id)
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    tags = list(dict.fromkeys([*cfg.wandb_tags, "vla", cfg.model.type, *( ["slurm"] if slurm_job_id else [])]))
    logging.info(
        "Initializing W&B telemetry: project=%s mode=%s run_id=%s resume=%s",
        cfg.wandb_project_name,
        mode,
        run_id,
        mode == "online",
    )
    run = wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project_name,
        name=experiment_name,
        id=run_id,
        group=cfg.wandb_group,
        job_type=cfg.wandb_job_type,
        notes=cfg.wandb_notes,
        tags=tags,
        mode=mode,
        resume="allow" if mode == "online" else None,
        dir=str(experiment_path),
        config={
            "training": dataclasses.asdict(cfg),
            "runtime": {
                "slurm_job_id": slurm_job_id,
                "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
                "hostname": os.uname().nodename,
                "world_size": cfg.distributed.world_size,
                "start_global_step": global_step,
                "start_checkpoint": checkpoint_num,
                "total_steps": total_steps,
            },
        },
        save_code=True,
    )
    run.define_metric("global_step")
    run.define_metric("samples", step_metric="global_step")
    run.define_metric("tokens", step_metric="global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("val/*", step_metric="global_step")
    run.define_metric("checkpoint/*", step_metric="global_step")
    run.summary["model/total_parameters"] = total_parameters
    run.summary["model/trainable_parameters"] = trainable_parameters
    run.summary["model/trainable_fraction"] = trainable_parameters / max(total_parameters, 1)
    run.summary["training/total_steps"] = total_steps
    run.summary["training/total_samples"] = cfg.total_train_samples
    return run
