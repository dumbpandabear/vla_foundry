# Minecraft full-training telemetry

The full Discrete-29 configuration logs to the `minecraft-vla-foundry` W&B
project under the `minecraft-discrete29-full` group. One run ID is persisted in
the experiment directory, so dependency-chained Slurm stages resume the same
online run instead of creating unrelated dashboards.

Logged signals include training and validation loss, learning rate, unclipped
gradient norm, sample/token progress, checkpoint progress, throughput, data and
compute timing, GPU memory, parameter counts, and Slurm/runtime metadata. W&B's
system monitor supplies host, CPU, GPU utilization, and device telemetry.

`wandb_mode: auto` is non-interactive: it uses online mode when
`WANDB_API_KEY` or a W&B entry in `~/.netrc` is available, otherwise it records
offline data beneath the training run directory. Configure credentials before
the first GPU stage to get a live dashboard:

```bash
cd /storage/home/ericli11/Projects/vla_foundry
.venv/bin/python -m wandb login
```

Offline stages can be uploaded later, preserving the persistent run ID:

```bash
bash examples/minecraft/sync_wandb_full.sh
```

## MetaMate Artifact dashboard

Generate a self-contained HTML dashboard from the authoritative training log,
checkpoints, Slurm chain, dataset manifest, label audit, and W&B file:

```bash
.venv/bin/python examples/minecraft/build_training_dashboard.py
```

The artifact is written to
`/storage/home/ericli11/minecraft-vla-poc/metamate-training-dashboard/index.html`
with a machine-readable `status.json` beside it. The HTML embeds its latest
snapshot for direct upload to MetaMate Artifacts. When served beside
`status.json`, it refreshes every 15 seconds. A lightweight watcher can keep
the snapshot current for a full Slurm stage:

```bash
sbatch examples/minecraft/run_training_dashboard.slurm
```

For a multi-day run, submit a dependency chain of daily refresh jobs (seven
days by default):

```bash
bash examples/minecraft/submit_training_dashboard.sh
```
