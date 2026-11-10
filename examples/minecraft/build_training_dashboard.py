#!/usr/bin/env python3
# ruff: noqa: E501
"""Build a self-contained live dashboard for the full Minecraft D29 run.

The generated ``index.html`` embeds the latest snapshot so it renders as a
standalone MetaMate Artifact.  When it is served beside ``status.json``, it
polls that file and updates without a page reload.  The builder intentionally
uses only the Python standard library so it can also run on login nodes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

VAL_RE = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}).*?"
    r"\[VAL\] ckpt=(?P<checkpoint>\d+) step=(?P<step>\d+) "
    r"avg_loss=(?P<loss>[0-9.]+) samples=(?P<samples>\d+)"
)
TRAIN_RE = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2},\d{2}:\d{2}:\d{2}).*?"
    r"Train Checkpoint: (?P<shard>\d+) \[\s*(?P<done>\d+)/(?P<total>\d+) "
    r"\(\s*(?P<percent>\d+)%\)\] Loss: (?P<loss>[0-9.]+).*?"
    r"(?P<throughput>[0-9.]+)/s.*?LR: (?P<lr>[0-9.]+)"
)
START_RE = re.compile(r"Start checkpoint (?P<checkpoint>\d+)")
CHECKPOINT_RE = re.compile(r"checkpoint_(?P<checkpoint>\d+)\.pt$")
ERROR_RE = re.compile(
    r"Traceback|CUDA out of memory|non.?finite|NCCL.*(?:error|abort)|FAILED|SIGSEGV",
    re.IGNORECASE,
)

DEFAULT_ROOT = Path("/storage/home/ericli11/minecraft-vla-poc")
DEFAULT_RUN = DEFAULT_ROOT / "runs/minecraft-qwen3-d29-full"
DEFAULT_DATASET = Path("/checkpoint/data/ericli11/minecraft-vla-poc/dataset-v2-full/dataset.json")
DEFAULT_AUDIT = DEFAULT_ROOT / "d29-label-audit-full.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "metamate-training-dashboard"
DEFAULT_JOB_CHAIN = [
    "16394334",
    "16394335",
    "16394336",
    "16394338",
    "16394339",
    "16394340",
    "16394341",
    "16394342",
    "16394343",
    "16394344",
    "16394345",
]


def _timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d,%H:%M:%S").replace(tzinfo=UTC)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _size_label(size: int) -> str:
    value = float(size)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or suffix == "TB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.1f} {suffix}"
        value /= 1024
    raise AssertionError("unreachable")


def _command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout if result.returncode == 0 else ""


def parse_training_log(text: str, *, global_batch_size: int, total_steps: int) -> dict[str, Any]:
    validations = [
        {
            "checkpoint": int(match["checkpoint"]),
            "step": int(match["step"]),
            "loss": float(match["loss"]),
            "samples": int(match["samples"]),
            "time": _timestamp(match["time"]).isoformat().replace("+00:00", "Z"),
        }
        for match in VAL_RE.finditer(text)
    ]
    reports_by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for match in TRAIN_RE.finditer(text):
        reports_by_shard[int(match["shard"])].append(
            {
                "time": _timestamp(match["time"]).isoformat().replace("+00:00", "Z"),
                "done": int(match["done"]),
                "total": int(match["total"]),
                "percent": int(match["percent"]),
                "loss": float(match["loss"]),
                "throughput": float(match["throughput"]),
                "lr": float(match["lr"]),
            }
        )
    shard_losses = []
    for shard in sorted(reports_by_shard):
        reports = reports_by_shard[shard]
        shard_losses.append(
            {
                "shard": shard,
                "mean_loss": sum(report["loss"] for report in reports) / len(reports),
                "latest_loss": reports[-1]["loss"],
                "reports": len(reports),
                "complete": bool(validations and shard < validations[-1]["checkpoint"]),
            }
        )

    starts = [int(match["checkpoint"]) for match in START_RE.finditer(text)]
    current_shard = starts[-1] if starts else (validations[-1]["checkpoint"] if validations else 0)
    current_report = reports_by_shard.get(current_shard, [])[-1] if reports_by_shard.get(current_shard) else None
    validated_step = validations[-1]["step"] if validations else 0
    current_step = validated_step
    if current_report:
        current_step += current_report["done"] // global_batch_size
    current_step = min(current_step, total_steps)

    first_loss = validations[0]["loss"] if validations else None
    best = min(validations, key=lambda point: point["loss"]) if validations else None
    latest = validations[-1] if validations else None
    improvement = ((first_loss - best["loss"]) / first_loss * 100) if first_loss and best else 0.0
    recent_delta = latest["loss"] - validations[-2]["loss"] if len(validations) > 1 else None
    learning = bool(first_loss and best and best["loss"] <= first_loss - 0.02)

    eta_hours = None
    eta_at = None
    if len(validations) >= 2:
        start = datetime.fromisoformat(validations[0]["time"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(validations[-1]["time"].replace("Z", "+00:00"))
        elapsed_hours = (end - start).total_seconds() / 3600
        completed_steps = validations[-1]["step"] - validations[0]["step"]
        if elapsed_hours > 0 and completed_steps > 0:
            steps_per_hour = completed_steps / elapsed_hours
            eta_hours = max(0.0, (total_steps - current_step) / steps_per_hour)
            eta_at = (datetime.now(UTC) + timedelta(hours=eta_hours)).isoformat().replace("+00:00", "Z")

    return {
        "validations": validations,
        "shard_losses": shard_losses,
        "current_shard": current_shard,
        "current_report": current_report,
        "current_step": current_step,
        "total_steps": total_steps,
        "progress": current_step / total_steps if total_steps else 0.0,
        "best": best,
        "latest_validation": latest,
        "improvement_percent": improvement,
        "recent_validation_delta": recent_delta,
        "learning": learning,
        "eta_hours": eta_hours,
        "eta_at": eta_at,
        "error_count": len(ERROR_RE.findall(text)),
    }


def query_jobs(job_ids: list[str]) -> list[dict[str, str]]:
    output = _command(
        [
            "sacct",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,Elapsed,Start,End,ExitCode,NodeList",
            "-n",
            "-P",
        ]
    )
    records = {
        fields[0]: fields
        for line in output.splitlines()
        if (fields := line.split("|")) and fields[0] in job_ids
    }
    jobs: list[dict[str, str]] = []
    for index, job_id in enumerate(job_ids):
        fields = records.get(job_id, [])
        jobs.append(
            {
                "stage": "quality + Craftium" if index == len(job_ids) - 1 else f"training {index + 1}",
                "job_id": job_id,
                "state": fields[1] if len(fields) > 1 else "UNKNOWN",
                "elapsed": fields[2] if len(fields) > 2 else "—",
                "start": fields[3] if len(fields) > 3 else "—",
                "end": fields[4] if len(fields) > 4 else "—",
                "exit_code": fields[5] if len(fields) > 5 else "—",
                "node": fields[6] if len(fields) > 6 else "—",
            }
        )
    return jobs


def checkpoint_artifacts(checkpoint_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in checkpoint_dir.glob("checkpoint_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        checkpoint = int(match["checkpoint"])
        optimizer = checkpoint_dir / f"optimizer_{checkpoint}.pt"
        artifacts.append(
            {
                "checkpoint": checkpoint,
                "model": path.name,
                "model_bytes": path.stat().st_size,
                "model_size": _size_label(path.stat().st_size),
                "optimizer": optimizer.name if optimizer.is_file() else None,
                "optimizer_bytes": optimizer.stat().st_size if optimizer.is_file() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )
    return sorted(artifacts, key=lambda item: item["checkpoint"])


def artifact_state(root: Path) -> dict[str, Any]:
    bundle = root / "policy-qwen3-d29-full"
    evaluation = root / "test-qwen3-d29-full.json"
    gate = root / "d29-full-quality-gate.json"
    craftium = root / "craftium-d29-full-summary.json"
    return {
        "export": {"state": "complete" if (bundle / "policy.json").is_file() else "pending", "path": str(bundle)},
        "evaluation": {"state": "complete" if evaluation.is_file() else "pending", "path": str(evaluation)},
        "quality_gate": {"state": "complete" if gate.is_file() else "pending", "path": str(gate)},
        "craftium": {"state": "complete" if craftium.is_file() else "pending", "path": str(craftium)},
    }


def collect_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    out_log = args.run_dir / "out.log"
    log_text = out_log.read_text(errors="replace") if out_log.is_file() else ""
    training = parse_training_log(
        log_text,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
    )
    dataset = _read_json(args.dataset_json) or {}
    audit = _read_json(args.audit_json) or {}
    train_audit = (audit.get("splits") or {}).get("train") or {}
    class_counts = train_audit.get("class_counts") or {}
    supported_actions = sum(int(value) > 0 for value in class_counts.values())
    missing_actions = list(train_audit.get("classes_without_targets") or [])
    jobs = query_jobs(args.job_chain) if args.query_slurm else []
    active_job = next((job for job in jobs if job["state"] == "RUNNING"), None)
    checkpoints = checkpoint_artifacts(args.run_dir / "checkpoints")

    wandb_files = sorted((args.run_dir / "wandb").glob("*/run-*.wandb"))
    wandb_file = wandb_files[-1] if wandb_files else None
    telemetry = {
        "mode": "offline" if wandb_file else "unavailable",
        "run_id": (args.run_dir / ".wandb_run_id").read_text().strip()
        if (args.run_dir / ".wandb_run_id").is_file()
        else "—",
        "bytes": wandb_file.stat().st_size if wandb_file else 0,
        "size": _size_label(wandb_file.stat().st_size) if wandb_file else "0 B",
        "modified": datetime.fromtimestamp(wandb_file.stat().st_mtime, UTC)
        .isoformat()
        .replace("+00:00", "Z")
        if wandb_file
        else None,
    }
    stages = artifact_state(args.root)
    stages = {
        "dataset": {"state": "complete" if args.dataset_json.is_file() else "missing", "path": str(args.dataset_json)},
        "training": {
            "state": "running" if active_job else (jobs[0]["state"].lower() if jobs else "unknown"),
            "path": str(args.run_dir),
        },
        **stages,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "title": "Minecraft Discrete-29 VLA",
        "subtitle": "Qwen3-VL-30B-A3B · full 1,951-hour corpus · 8× H200",
        "run": {
            "name": args.run_dir.name,
            "job_id": active_job["job_id"] if active_job else args.job_chain[0],
            "state": active_job["state"] if active_job else (jobs[0]["state"] if jobs else "UNKNOWN"),
            "elapsed": active_job["elapsed"] if active_job else (jobs[0]["elapsed"] if jobs else "—"),
            "node": active_job["node"] if active_job else (jobs[0]["node"] if jobs else "—"),
            "gpu_count": 8,
            "gpu_type": "H200",
            "action_space": "Discrete 29",
            "model_parameters": 31_096_346_509,
        },
        "training": training,
        "dataset": {
            "source_hours": args.source_hours,
            "samples": dataset.get("samples", {}),
            "total_samples": sum((dataset.get("samples") or {}).values()),
            "recordings": dataset.get("recordings", {}),
            "total_recordings": sum((dataset.get("recordings") or {}).values()),
            "clips": dataset.get("clips", {}),
            "total_clips": sum((dataset.get("clips") or {}).values()),
            "supported_actions": supported_actions,
            "total_actions": 29,
            "missing_actions": missing_actions,
        },
        "checkpoints": checkpoints,
        "jobs": jobs,
        "telemetry": telemetry,
        "pipeline": stages,
        "paths": {
            "run": str(args.run_dir),
            "training_log": str(out_log),
            "dataset": str(args.dataset_json),
            "audit": str(args.audit_json),
        },
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark"><title>Minecraft D29 VLA · Training</title>
<style>
:root{--bg:#07110d;--panel:#0d1b15;--panel2:#11231b;--line:#21382d;--text:#ecf8f0;--muted:#90a99a;--lime:#9dfc6d;--mint:#47e6a1;--amber:#ffc45c;--red:#ff776d;--blue:#70b7ff;--shadow:0 18px 60px #0008}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 18% 0,#173222 0,transparent 36%),radial-gradient(circle at 92% 8%,#102e2a 0,transparent 27%),var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;min-height:100vh}.shell{max-width:1440px;margin:auto;padding:34px 28px 60px}.top{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:24px}.eyebrow{color:var(--lime);font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px}h1{font-size:clamp(28px,4vw,52px);line-height:1;margin:0 0 10px;letter-spacing:-.045em}.subtitle{color:var(--muted);font-size:15px}.live{display:flex;align-items:center;gap:9px;border:1px solid var(--line);background:#0b1812dd;padding:10px 13px;border-radius:999px;white-space:nowrap}.dot{width:8px;height:8px;border-radius:50%;background:var(--lime);box-shadow:0 0 14px var(--lime);animation:pulse 1.8s infinite}@keyframes pulse{50%{opacity:.35}}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}.card{background:linear-gradient(145deg,#102019e8,#0a1711e8);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);padding:20px;min-width:0}.hero{grid-column:span 12;display:grid;grid-template-columns:1.4fr .8fr;gap:24px}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:22px}.metric{border-left:2px solid var(--line);padding-left:13px}.label{color:var(--muted);font-size:11px;letter-spacing:.1em;text-transform:uppercase}.value{font-size:24px;font-weight:750;letter-spacing:-.03em;margin-top:4px}.value.good{color:var(--lime)}.progress-head{display:flex;justify-content:space-between;align-items:end;margin-bottom:8px}.progress-track{height:12px;background:#05100b;border:1px solid var(--line);border-radius:999px;overflow:hidden}.progress-fill{height:100%;background:linear-gradient(90deg,var(--mint),var(--lime));box-shadow:0 0 24px #70f78b88}.progress-caption{display:flex;justify-content:space-between;color:var(--muted);font:12px ui-monospace,SFMono-Regular,Menlo,monospace;margin-top:8px}.chart-card{grid-column:span 8}.side-card{grid-column:span 4}.section-title{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:14px}.section-title h2{font-size:16px;margin:0}.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;padding:5px 9px;color:var(--muted);font-size:11px}.chip.good{color:var(--lime);border-color:#427a48}.chip.warn{color:var(--amber);border-color:#7c6437}.chart{width:100%;height:270px;display:block}.axis{stroke:#294136;stroke-width:1}.gridline{stroke:#1d3028;stroke-width:1}.loss-line{fill:none;stroke:var(--lime);stroke-width:3;stroke-linecap:round;stroke-linejoin:round;filter:drop-shadow(0 0 6px #9dfc6d66)}.loss-dot{fill:#07110d;stroke:var(--lime);stroke-width:2}.best-dot{fill:var(--lime)}.axis-text{fill:#789184;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}.tooltip-text{fill:var(--text);font:700 11px ui-monospace,SFMono-Regular,Menlo,monospace}.pipeline{display:grid;gap:9px}.stage{display:grid;grid-template-columns:16px 1fr auto;gap:10px;align-items:center;padding:10px 0;border-bottom:1px solid #1b3026}.stage:last-child{border:0}.stage-dot{width:10px;height:10px;border-radius:3px;background:#34463d}.stage.complete .stage-dot{background:var(--lime)}.stage.running .stage-dot{background:var(--blue);box-shadow:0 0 10px #70b7ff88}.stage-state{color:var(--muted);font-size:11px;text-transform:uppercase}.data-card{grid-column:span 4}.check-card{grid-column:span 4}.ops-card{grid-column:span 4}.big{font-size:36px;font-weight:800;letter-spacing:-.05em}.muted{color:var(--muted)}.bar-list{display:grid;gap:12px;margin-top:18px}.bar-label{display:flex;justify-content:space-between;margin-bottom:5px;font-size:12px}.mini-track{height:6px;background:#08140f;border-radius:99px;overflow:hidden}.mini-fill{height:100%;background:var(--mint)}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;color:var(--muted);font-weight:500;padding:8px 7px;border-bottom:1px solid var(--line)}td{padding:9px 7px;border-bottom:1px solid #172a21;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.table-card{grid-column:span 7;overflow:auto}.jobs-card{grid-column:span 5;overflow:auto}.best-row{background:#18342166}.status-running{color:var(--blue)}.status-complete,.status-completed{color:var(--lime)}.status-pending{color:var(--muted)}.footer{display:flex;justify-content:space-between;gap:16px;color:var(--muted);font-size:11px;margin-top:18px}.path{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%}@media(max-width:920px){.hero{grid-template-columns:1fr}.chart-card,.side-card,.data-card,.check-card,.ops-card,.table-card,.jobs-card{grid-column:span 12}.metric-row{grid-template-columns:repeat(2,1fr)}.top{flex-direction:column}}@media(max-width:560px){.shell{padding:22px 14px 40px}.metric-row{grid-template-columns:1fr}.card{padding:16px}}
</style><style>
/* ASTRYX core token mapping: neutral surfaces, blue accent, 4px spacing grid. */
:root{color-scheme:light;--bg:#F1F4F7;--panel:#FFFFFF;--panel2:#FFFFFF;--line:#05365919;--text:#0A1317;--muted:#4E606F;--lime:#0064E0;--mint:#6B1EFD;--amber:#D66100;--red:#D31130;--blue:#0171E3;--shadow:0 1px 1px rgba(0,0,0,.08),0 2px 8px rgba(0,0,0,.08)}
body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.4286}
.shell{max-width:1320px;padding:24px 28px 48px}.appbar{height:56px;display:flex;align-items:center;justify-content:space-between;margin:-24px -28px 28px;padding:0 28px;background:var(--panel);border-bottom:1px solid var(--line)}.brand{display:flex;align-items:center;gap:10px}.brand-mark{display:grid;place-items:center;width:28px;height:28px;border-radius:8px;background:var(--blue);color:#fff;font-weight:700}.brand-copy{display:grid;line-height:1.2}.brand-copy strong{font-size:14px;font-weight:600}.brand-copy small{font-size:11px;color:var(--muted)}
.top{align-items:end;margin-bottom:24px}.eyebrow{color:var(--blue);font-family:inherit;font-size:12px;font-weight:600;letter-spacing:0;text-transform:none;margin-bottom:8px}h1{font-size:29px;line-height:1.24;letter-spacing:-.015em;font-weight:600;margin-bottom:6px}.subtitle{font-size:14px}.live{background:var(--panel);border-color:var(--line);padding:7px 11px;border-radius:9999px}.dot{background:var(--blue);box-shadow:none;animation:none}
.grid{gap:16px}.card{background:var(--panel);border-color:var(--line);border-radius:12px;box-shadow:var(--shadow);padding:20px}.hero{gap:28px}.metric{border-left-color:var(--line)}.label{font-size:12px;letter-spacing:0;text-transform:none;font-weight:500}.value{font-size:24px;font-weight:600;letter-spacing:-.01em}.value.good{color:var(--blue)}.big{font-size:35px;font-weight:600;letter-spacing:-.02em}.section-title h2{font-size:17px;font-weight:600}.chip{border-radius:9999px;background:var(--bg);font-size:12px}.chip.good{color:var(--blue);border-color:#0064E055;background:#0082FB14}.chip.warn{color:#A13F04;border-color:#EB6E0055;background:#F2790214}
.progress-track,.mini-track{background:#CCD3DB;border-color:var(--line)}.progress-fill{background:linear-gradient(90deg,#0171E3,#6B1EFD);box-shadow:none}.progress-caption{font-family:"SF Mono",Monaco,Consolas,monospace}.axis{stroke:#CCD3DB}.gridline{stroke:#E5E9ED}.loss-line{stroke:#0171E3;filter:none}.loss-dot{fill:var(--panel);stroke:#0171E3}.best-dot{fill:#0171E3}.axis-text{fill:var(--muted)}.stage{border-bottom-color:var(--line)}.stage-dot{background:#AFB9C4}.stage.complete .stage-dot{background:#0171E3}.stage.running .stage-dot{background:#6B1EFD;box-shadow:none}.mini-fill{background:#0171E3}th{border-bottom-color:var(--line)}td{border-bottom-color:var(--line)}.best-row{background:#0082FB14}.status-running{color:#6B1EFD}.status-complete,.status-completed{color:#0064E0}.footer{border-top:1px solid var(--line);padding-top:16px}.path{font-family:"SF Mono",Monaco,Consolas,monospace}
.kpi-grid{grid-column:span 12;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.kpi-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);padding:18px 20px;min-width:0}.kpi-head{display:flex;justify-content:space-between;align-items:center;color:var(--muted);font-size:12px}.kpi-icon{display:grid;place-items:center;width:30px;height:30px;border-radius:8px;background:#0082FB14;color:var(--blue);font-weight:700}.kpi-value{font-size:29px;line-height:1.2;font-weight:600;letter-spacing:-.02em;margin:14px 0 4px}.kpi-foot{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:12px}.kpi-foot strong{color:var(--blue);font-weight:600}.kpi-progress{height:4px;border-radius:9999px;background:#CCD3DB;margin-top:14px;overflow:hidden}.kpi-progress>span{display:block;height:100%;background:var(--blue)}
.chart-card{grid-column:span 8}.train-chart-card{grid-column:span 4}.chart-header{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:8px}.chart-subtitle{color:var(--muted);font-size:12px;margin-top:3px}.chart-legend{display:flex;align-items:center;gap:14px;color:var(--muted);font-size:12px;white-space:nowrap}.legend-key{display:inline-flex;align-items:center;gap:6px}.legend-line{width:18px;height:2px;background:var(--blue)}.legend-line.dashed{height:0;background:none;border-top:2px dashed #8494A3}.chart{height:320px}.loss-area{fill:url(#loss-area)}.loss-line{stroke-width:2.5}.baseline-line{stroke:#8494A3;stroke-width:1.5;stroke-dasharray:5 5}.axis-label{fill:var(--muted);font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.point-label{fill:var(--text);font:600 11px "SF Mono",Monaco,Consolas,monospace}.train-line{fill:none;stroke:#6B1EFD;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}.train-dot{fill:var(--panel);stroke:#6B1EFD;stroke-width:2}
.pipeline-card{grid-column:span 12}.pipeline-strip{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));margin-top:8px}.pipe-stage{position:relative;padding:18px 12px 6px;text-align:center;color:var(--muted);font-size:12px}.pipe-stage:before{content:"";position:absolute;top:7px;left:0;right:0;height:2px;background:#CCD3DB}.pipe-stage:first-child:before{left:50%}.pipe-stage:last-child:before{right:50%}.pipe-node{position:absolute;z-index:1;top:1px;left:calc(50% - 7px);width:14px;height:14px;border:3px solid var(--panel);border-radius:50%;background:#AFB9C4;box-shadow:0 0 0 1px #CCD3DB}.pipe-stage.complete .pipe-node{background:var(--blue);box-shadow:0 0 0 1px var(--blue)}.pipe-stage.running .pipe-node{background:#6B1EFD;box-shadow:0 0 0 1px #6B1EFD}.pipe-stage.complete:before{background:var(--blue)}.pipe-name{display:block;color:var(--text);font-weight:500;margin-top:5px}.pipe-state{display:block;margin-top:2px;text-transform:capitalize}
.dataset-card{grid-column:span 5}.coverage-card{grid-column:span 3}.runtime-card{grid-column:span 4}.split-stack{display:flex;height:18px;border-radius:5px;overflow:hidden;margin:20px 0 16px;background:#CCD3DB}.split-stack span:nth-child(1){background:#0171E3}.split-stack span:nth-child(2){background:#6B1EFD}.split-stack span:nth-child(3){background:#EB6E00}.split-legend{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.split-item{display:grid;grid-template-columns:8px 1fr;column-gap:7px;align-items:center}.split-swatch{width:8px;height:8px;border-radius:2px}.split-item:nth-child(1) .split-swatch{background:#0171E3}.split-item:nth-child(2) .split-swatch{background:#6B1EFD}.split-item:nth-child(3) .split-swatch{background:#EB6E00}.split-item strong{font-size:13px;font-weight:600}.split-item small{grid-column:2;color:var(--muted)}.coverage-wrap{display:flex;align-items:center;justify-content:center;gap:20px;padding:10px 0}.coverage-ring{--coverage:0deg;position:relative;width:116px;height:116px;flex:0 0 auto;border-radius:50%;background:conic-gradient(var(--blue) var(--coverage),#CCD3DB 0)}.coverage-ring:after{content:"";position:absolute;inset:12px;border-radius:50%;background:var(--panel)}.coverage-value{position:absolute;z-index:1;inset:0;display:grid;place-items:center;text-align:center;font-size:24px;font-weight:600}.coverage-value small{display:block;font-size:11px;color:var(--muted);font-weight:400}.coverage-note{color:var(--muted);font-size:12px;line-height:1.5}.detail-list{display:grid;margin-top:4px}.detail-row{display:flex;justify-content:space-between;gap:16px;padding:9px 0;border-bottom:1px solid var(--line)}.detail-row:last-child{border:0}.detail-row span{color:var(--muted)}.detail-row strong{font-weight:500;text-align:right}.table-card{grid-column:span 7}.jobs-card{grid-column:span 5}
@media(max-width:920px){.kpi-grid{grid-template-columns:repeat(2,1fr)}.chart-card,.train-chart-card,.pipeline-card,.dataset-card,.coverage-card,.runtime-card{grid-column:span 12}.pipeline-strip{grid-template-columns:repeat(3,1fr);row-gap:14px}.split-legend{grid-template-columns:repeat(3,1fr)}}
@media(max-width:560px){.kpi-grid{grid-template-columns:1fr}.pipeline-strip{grid-template-columns:1fr}.pipe-stage:before{display:none}.split-legend{grid-template-columns:1fr}.coverage-wrap{justify-content:flex-start}}
@media(prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#111112;--panel:#1F1F22;--panel2:#28292C;--line:#F2F4F619;--text:#DFE2E5;--muted:#AAAFB5;--lime:#3E9EFB;--mint:#9081FF;--blue:#2694FE;--amber:#FD9537;--red:#F5394F;--shadow:0 1px 1px rgba(0,0,0,.2),0 2px 8px rgba(0,0,0,.2)}.progress-track,.mini-track{background:#5A5E66}.gridline{stroke:#333338}.axis{stroke:#494D53}.chip.good{color:#AFD7FF;background:#0171E322}.chip.warn{color:#FDB876;background:#F2790222}.best-row{background:#0171E319}}
@media(max-width:560px){.shell{padding:16px 14px 36px}.appbar{margin:-16px -14px 22px;padding:0 14px}}
</style></head><body><main class="shell"><header class="appbar"><div class="brand"><span class="brand-mark">V</span><span class="brand-copy"><strong>VLA Foundry</strong><small>Training operations</small></span></div><div class="live"><span class="dot" id="live-dot"></span><span id="live-label">Loading run</span></div></header><header class="top"><div><div class="eyebrow">Full-corpus training run</div><h1 id="title">Minecraft Discrete-29 VLA</h1><div class="subtitle" id="subtitle"></div></div></header><div class="grid" id="dashboard"></div><footer class="footer"><span id="generated"></span><span>Auto-refreshes every 15 seconds when served with status.json</span></footer></main>
<script id="initial-data" type="application/json">__INITIAL_DATA__</script><script>
const initial=JSON.parse(document.getElementById('initial-data').textContent);let state=initial;
const $=s=>document.querySelector(s),esc=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=n=>new Intl.NumberFormat('en-US',{notation:n>=1e6?'compact':'standard',maximumFractionDigits:1}).format(n||0),pct=n=>(100*(n||0)).toFixed(1)+'%',loss=n=>Number(n).toFixed(6),ago=iso=>{if(!iso)return'—';const s=Math.max(0,(Date.now()-Date.parse(iso))/1000);return s<60?Math.round(s)+'s ago':s<3600?Math.round(s/60)+'m ago':Math.round(s/3600)+'h ago'};
function validationChart(points){if(!points.length)return'<div class="muted">Waiting for validation data</div>';const w=860,h=320,p={l:68,r:24,t:24,b:54},vals=points.map(d=>d.loss),rawLo=Math.min(...vals),rawHi=Math.max(...vals),pad=Math.max(.012,(rawHi-rawLo)*.14),lo=rawLo-pad,hi=rawHi+pad,xMax=Math.max(1000,Math.ceil(points.at(-1).step/1000)*1000),x=v=>p.l+v*(w-p.l-p.r)/xMax,y=v=>p.t+(hi-v)*(h-p.t-p.b)/(hi-lo),path=points.map((d,i)=>(i?'L':'M')+x(d.step).toFixed(1)+' '+y(d.loss).toFixed(1)).join(' '),area=`${path} L ${x(points.at(-1).step).toFixed(1)} ${h-p.b} L ${x(points[0].step).toFixed(1)} ${h-p.b} Z`,baseline=y(points[0].loss);let grid='';for(let i=0;i<6;i++){const v=lo+(hi-lo)*i/5,yy=y(v);grid+=`<line class="gridline" x1="${p.l}" y1="${yy}" x2="${w-p.r}" y2="${yy}"/><text class="axis-text" text-anchor="end" x="${p.l-10}" y="${yy+4}">${v.toFixed(3)}</text>`}let ticks='';for(let i=0;i<6;i++){const v=xMax*i/5,xx=x(v);ticks+=`<line class="axis" x1="${xx}" y1="${h-p.b}" x2="${xx}" y2="${h-p.b+5}"/><text class="axis-text" text-anchor="middle" x="${xx}" y="${h-p.b+21}">${v>=1000?(v/1000).toFixed(v%1000?1:0)+'k':Math.round(v)}</text>`}const dots=points.map(d=>`<circle class="loss-dot ${d.loss===rawLo?'best-dot':''}" cx="${x(d.step)}" cy="${y(d.loss)}" r="4.5"><title>Checkpoint ${d.checkpoint} · step ${d.step.toLocaleString()} · loss ${loss(d.loss)}</title></circle>`).join(''),last=points.at(-1);return`<svg viewBox="0 0 ${w} ${h}" class="chart" role="img" aria-label="Validation loss by optimizer step"><defs><linearGradient id="loss-area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0171E3" stop-opacity=".22"/><stop offset="1" stop-color="#0171E3" stop-opacity=".02"/></linearGradient></defs>${grid}<line class="axis" x1="${p.l}" y1="${h-p.b}" x2="${w-p.r}" y2="${h-p.b}"/><line class="axis" x1="${p.l}" y1="${p.t}" x2="${p.l}" y2="${h-p.b}"/>${ticks}<line class="baseline-line" x1="${p.l}" y1="${baseline}" x2="${w-p.r}" y2="${baseline}"/><path class="loss-area" d="${area}"/><path class="loss-line" d="${path}"/>${dots}<text class="point-label" text-anchor="end" x="${x(last.step)-8}" y="${y(last.loss)-10}">${loss(last.loss)}</text><text class="axis-label" text-anchor="middle" x="${(p.l+w-p.r)/2}" y="${h-5}">Optimizer step</text><text class="axis-label" text-anchor="middle" transform="translate(15 ${(p.t+h-p.b)/2}) rotate(-90)">Cross-entropy loss</text></svg>`}
function trainingChart(points){if(!points.length)return'<div class="muted">Waiting for training data</div>';const w=420,h=320,p={l:58,r:18,t:24,b:52},vals=points.map(d=>d.mean_loss),rawLo=Math.min(...vals),rawHi=Math.max(...vals),pad=Math.max(.012,(rawHi-rawLo)*.15),lo=rawLo-pad,hi=rawHi+pad,x=i=>p.l+i*(w-p.l-p.r)/Math.max(1,points.length-1),y=v=>p.t+(hi-v)*(h-p.t-p.b)/(hi-lo),path=points.map((d,i)=>(i?'L':'M')+x(i)+' '+y(d.mean_loss)).join(' ');let grid='';for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);grid+=`<line class="gridline" x1="${p.l}" y1="${yy}" x2="${w-p.r}" y2="${yy}"/><text class="axis-text" text-anchor="end" x="${p.l-8}" y="${yy+4}">${v.toFixed(2)}</text>`}const dots=points.map((d,i)=>`<circle class="train-dot" cx="${x(i)}" cy="${y(d.mean_loss)}" r="4"><title>Shard ${d.shard+1} · mean loss ${d.mean_loss.toFixed(4)}</title></circle><text class="axis-text" text-anchor="middle" x="${x(i)}" y="${h-p.b+20}">C${d.shard+1}</text>`).join('');return`<svg viewBox="0 0 ${w} ${h}" class="chart" role="img" aria-label="Mean training loss by shard">${grid}<line class="axis" x1="${p.l}" y1="${h-p.b}" x2="${w-p.r}" y2="${h-p.b}"/><line class="axis" x1="${p.l}" y1="${p.t}" x2="${p.l}" y2="${h-p.b}"/><path class="train-line" d="${path}"/>${dots}<text class="axis-label" text-anchor="middle" x="${(p.l+w-p.r)/2}" y="${h-5}">Training shard</text></svg>`}
function pipelineStrip(p){return Object.entries(p).map(([name,item])=>`<div class="pipe-stage ${esc(item.state)}"><span class="pipe-node"></span><span class="pipe-name">${esc(name.replaceAll('_',' '))}</span><span class="pipe-state">${esc(item.state)}</span></div>`).join('')}
function render(d){state=d;const t=d.training,r=d.run,ds=d.dataset,b=t.best||{},cur=t.current_report||{},isFresh=Date.now()-Date.parse(d.generated_at)<90000;$('#title').textContent=d.title;$('#subtitle').textContent=d.subtitle;$('#live-label').textContent=`${r.state} · job ${r.job_id}`;$('#live-dot').style.background=isFresh?'var(--blue)':'var(--amber)';$('#generated').textContent=`Snapshot ${new Date(d.generated_at).toLocaleString()} · telemetry ${ago(d.telemetry.modified)}`;const eta=t.eta_hours==null?'Calculating':`${t.eta_hours.toFixed(0)}h`,validations=t.validations||[],latest=validations.at(-1)||{},shards=t.shard_losses||[],latestShard=shards.at(-1)||{},splits=['train','val','test'].map(name=>({name,value:Number(ds.samples[name]||0)})),coverage=ds.total_actions?ds.supported_actions/ds.total_actions:0,recent=t.recent_validation_delta,trend=recent==null?'—':`${recent<=0?'↓':'↑'} ${Math.abs(recent).toFixed(4)}`;$('#dashboard').innerHTML=`
<section class="kpi-grid"><article class="kpi-card"><div class="kpi-head"><span>Training progress</span><span class="kpi-icon">%</span></div><div class="kpi-value">${pct(t.progress)}</div><div class="kpi-foot"><span>Step ${fmt(t.current_step)}</span><span>of ${fmt(t.total_steps)}</span></div><div class="kpi-progress"><span style="width:${pct(t.progress)}"></span></div></article><article class="kpi-card"><div class="kpi-head"><span>Best validation loss</span><span class="kpi-icon">↓</span></div><div class="kpi-value">${b.loss==null?'—':loss(b.loss)}</div><div class="kpi-foot"><strong>${t.improvement_percent.toFixed(2)}% lower</strong><span>latest Δ ${trend}</span></div></article><article class="kpi-card"><div class="kpi-head"><span>Training throughput</span><span class="kpi-icon">↗</span></div><div class="kpi-value">${cur.throughput==null?'—':cur.throughput.toFixed(1)} <small>/s</small></div><div class="kpi-foot"><span>${r.gpu_count}× ${r.gpu_type}</span><span>${esc(r.node)}</span></div></article><article class="kpi-card"><div class="kpi-head"><span>Estimated remaining</span><span class="kpi-icon">◷</span></div><div class="kpi-value">${eta}</div><div class="kpi-foot"><span>${esc(r.elapsed)} elapsed</span><strong>${t.error_count?'Needs attention':'Healthy'}</strong></div></article></section>
<section class="card chart-card"><div class="chart-header"><div><div class="section-title"><h2>Validation loss</h2></div><div class="chart-subtitle">Held-out cross-entropy measured against optimizer step</div></div><div class="chart-legend"><span class="legend-key"><i class="legend-line"></i>Validation</span><span class="legend-key"><i class="legend-line dashed"></i>C1 baseline</span></div></div>${validationChart(validations)}</section>
<section class="card train-chart-card"><div class="chart-header"><div><div class="section-title"><h2>Training loss</h2></div><div class="chart-subtitle">Mean reported loss by shard</div></div><span class="chip">latest ${latestShard.mean_loss==null?'—':latestShard.mean_loss.toFixed(3)}</span></div>${trainingChart(shards)}</section>
<section class="card pipeline-card"><div class="section-title"><h2>End-to-end pipeline</h2><span class="chip ${t.learning?'good':'warn'}">${t.learning?'Learning confirmed':'Collecting evidence'}</span></div><div class="pipeline-strip">${pipelineStrip(d.pipeline)}</div></section>
<section class="card dataset-card"><div class="section-title"><h2>Dataset composition</h2><span class="chip">${fmt(ds.source_hours)} hours</span></div><div class="value">${fmt(ds.total_samples)} examples</div><div class="split-stack">${splits.map(s=>`<span style="width:${pct(s.value/ds.total_samples)}"></span>`).join('')}</div><div class="split-legend">${splits.map(s=>`<div class="split-item"><i class="split-swatch"></i><strong>${esc(s.name)}</strong><small>${fmt(s.value)} · ${pct(s.value/ds.total_samples)}</small></div>`).join('')}</div></section>
<section class="card coverage-card"><div class="section-title"><h2>Action coverage</h2></div><div class="coverage-wrap"><div class="coverage-ring" style="--coverage:${coverage*360}deg"><div class="coverage-value">${ds.supported_actions}<small>of ${ds.total_actions}</small></div></div><div class="coverage-note">Observed classes in raw VPT supervision.<br><strong>${ds.total_actions-ds.supported_actions} crafting actions</strong> remain unsupervised.</div></div></section>
<section class="card runtime-card"><div class="section-title"><h2>Runtime & telemetry</h2><span class="chip ${t.error_count?'warn':'good'}">${t.error_count} fatal signals</span></div><div class="detail-list"><div class="detail-row"><span>Current checkpoint</span><strong>C${t.current_shard+1} · ${cur.percent??0}%</strong></div><div class="detail-row"><span>Learning rate</span><strong>${cur.lr==null?'—':cur.lr.toExponential(2)}</strong></div><div class="detail-row"><span>W&B telemetry</span><strong>${esc(d.telemetry.size)} · ${esc(d.telemetry.mode)}</strong></div><div class="detail-row"><span>Saved states</span><strong>${d.checkpoints.length} model + optimizer</strong></div><div class="detail-row"><span>Latest validation</span><strong>${latest.loss==null?'pending':loss(latest.loss)}</strong></div></div></section>
<section class="card table-card"><div class="section-title"><h2>Validation checkpoints</h2><span class="chip">lower is better</span></div><table><thead><tr><th>Checkpoint</th><th>Step</th><th>Loss</th><th>Δ previous</th><th>Samples</th></tr></thead><tbody>${validations.map((v,i)=>`<tr class="${v.checkpoint===b.checkpoint?'best-row':''}"><td>C${v.checkpoint}${v.checkpoint===b.checkpoint?' ★':''}</td><td>${fmt(v.step)}</td><td>${loss(v.loss)}</td><td>${i?((v.loss-validations[i-1].loss)>=0?'+':'')+(v.loss-validations[i-1].loss).toFixed(6):'—'}</td><td>${fmt(v.samples)}</td></tr>`).join('')}</tbody></table></section>
<section class="card jobs-card"><div class="section-title"><h2>Slurm pipeline</h2><span class="chip">${d.jobs.length} jobs</span></div><table><thead><tr><th>Stage</th><th>Job</th><th>State</th><th>Elapsed</th></tr></thead><tbody>${d.jobs.map(j=>`<tr><td>${esc(j.stage)}</td><td>${esc(j.job_id)}</td><td class="status-${esc(j.state.toLowerCase())}">${esc(j.state)}</td><td>${esc(j.elapsed)}</td></tr>`).join('')}</tbody></table></section>`}
render(initial);async function refresh(){try{const res=await fetch(`status.json?t=${Date.now()}`,{cache:'no-store'});if(res.ok)render(await res.json())}catch(_){}}setInterval(refresh,15000);
</script></body></html>'''


def write_dashboard(snapshot: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    compact = json.dumps(snapshot, separators=(",", ":")).replace("</", "<\\/")
    (output / "status.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    html = HTML_TEMPLATE.replace("__INITIAL_DATA__", compact)
    # The ASTRYX stylesheet replaced the original prototype theme. Emit only
    # the final design system CSS so artifact sanitizers never fall back to the
    # superseded green prototype if they normalize adjacent style elements.
    html = re.sub(
        r"<style>.*?</style><style>(.*?)</style>",
        lambda match: f"<style>{match.group(1)}</style>",
        html,
        count=1,
        flags=re.DOTALL,
    )
    (output / "index.html").write_text(html)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-steps", type=int, default=59_406)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--source-hours", type=float, default=1_951)
    parser.add_argument("--job-chain", nargs="+", default=DEFAULT_JOB_CHAIN)
    parser.add_argument("--no-slurm", action="store_false", dest="query_slurm")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.total_steps <= 0 or args.global_batch_size <= 0 or args.interval <= 0:
        raise SystemExit("total steps, global batch size, and interval must be positive")
    while True:
        snapshot = collect_snapshot(args)
        write_dashboard(snapshot, args.output)
        print(
            f"dashboard={args.output / 'index.html'} step={snapshot['training']['current_step']} "
            f"best={((snapshot['training']['best'] or {}).get('loss'))}",
            flush=True,
        )
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
