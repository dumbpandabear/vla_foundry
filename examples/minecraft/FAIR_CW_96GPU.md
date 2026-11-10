# Minecraft Discrete-29: FAIR-CW 96-GPU handoff

This recipe continues the 21-observed-action foundation model on the complete
VPT-derived Minecraft corpus. It uses 24 Richmond GB300 nodes with 4 GPUs each, exports the
best validation checkpoint, and runs the held-out offline evaluation. The
separate native-Craftium fine-tune adds supervision for actions 18–25 after the
foundation run.

## What transfers easily

| Item | Size | Transport |
| --- | ---: | --- |
| Source, configs, tests, launchers | small | Git/private branch or source archive |
| Compact trainable-weights checkpoint 8 | 102,544,797 bytes | shared home or artifact copy |
| Compressed checkpoint 8 | 94,755,262 bytes | shared home or artifact copy |
| Compact streaming index, manifests, and label audit | small | build once beside the raw corpus |
| Optimizer checkpoint 8 | 205,072,533 bytes | optional artifact copy only |
| Qwen3-VL-30B-A3B-Thinking base model | 62 GB | shared `/checkpoint` or model cache |
| Prepared VPT WebDataset | about 423 GB | not required; optional fallback only |
| Balanced native-D29 Craftium dataset | 3.7 GB | shared `/checkpoint` or artifact copy |

Do not commit the model or optimizer to normal Git. Although the zstd file is
just below GitHub's single-file ceiling, it would permanently add about 95 MB
to repository history. The source remote is public, so use a private branch,
Git bundle, or an approved internal artifact store for this work.

## Source branch

Check out `minecraft-vla-d29-fair-cw`. It contains the streaming index builder,
raw VPT dataloader, Discrete-29 model and export path, telemetry, tests, and the
Richmond launchers. A Git bundle is also produced as a fallback when the branch
cannot be fetched from the remote.

## ARM64 environment

Richmond (`fair-cw-use2-1`) is `aarch64`. Do not copy or reuse the x86_64
`.venv` from FAIR-SC. Check out the source first, then create a Python 3.12
environment on FAIR-CW itself:

```bash
uname -m
uv sync
uv pip install -e .
.venv/bin/python -c \
  'import platform, torch; print(platform.machine(), torch.__version__, torch.version.cuda)'
```

The output must report `aarch64`, a CUDA-enabled PyTorch build, and the CUDA
version expected by the FAIR-CW nodes. If the locked PyTorch wheel is not
available for ARM64, use FAIR-CW's supported CUDA/PyTorch environment and run
`uv pip install -e . --no-deps`; do not install an x86 wheel or CPU-only torch.
The `.pt` checkpoint contains tensors and is portable across CPU architectures.

## Current artifact

The portable weights-only checkpoint is:

```text
/storage/home/ericli11/minecraft-vla-poc/transfer/checkpoint_8.pt.zst
sha256 56ac38a3351469239780eaafdbf6eed3ce4f867e4d80901d27dbdf09dac140a2
```

The decompressed checkpoint must have:

```text
sha256 0f4c94c1aca16326160b8b90006d24ceff72f9107705825158da12a0e7b3356e
```

Copy the compressed checkpoint to FAIR-CW with
`cloud_corp hpc copy` or another approved small-file transfer, then run:

```bash
zstd -d -f \
  /storage/home/ericli11/minecraft-vla-poc/transfer/checkpoint_8.pt.zst \
  -o /storage/home/ericli11/minecraft-vla-poc/transfer/checkpoint_8.pt
sha256sum /storage/home/ericli11/minecraft-vla-poc/transfer/checkpoint_8.pt
```

The checkpoint is only about 95 MB; it is not necessary to involve the bulk
data-transfer service for it.

## Build the compact streaming index

Richmond already has the source corpus at:

```text
/s3/profiles/data-foundations-use2-1/minecraft_vpt
```

Its `data/`, `splits/`, and newest `json_data/train_qwen35_9b_04_29_26.jsonl`
and validation files match the loader's expected layout. Do not transfer or
regenerate the 423 GB prepared WebDataset. Build a compact metadata index once;
the trainer then reads action JSONL files and decodes only the required MP4
frames directly from the resident corpus:

```bash
VLA_REPO_ROOT="$PWD" \
VLA_SOURCE=/s3/profiles/data-foundations-use2-1/minecraft_vpt \
VLA_INDEX_OUTPUT=/s3/profiles/data-foundations-use2-1/minecraft_vpt/vla_foundry_stream_index \
bash examples/minecraft/build_vpt_stream_index_richmond.sh
```

The index stores goals, valid frame anchors, manifests, statistics, and the D29
label audit, but no images or action arrays. Its two-recording shards provide
enough independent shards for 96-way DDP and 80 checkpoint windows. The builder
fails rather than replacing an existing output directory. If the normal profile
is not writable, set `VLA_INDEX_OUTPUT` to the corresponding `-rw` profile and
use that same path as `VLA_DATA` below.

The old materialized pipeline remains available as
`prepare_vla_full_richmond.sh`, but it is only a fallback for a measured
streaming-throughput problem.

## Training recipe

The launcher defaults reproduce the model/data recipe with a 96-GPU batch:

- 24 nodes x 4 GB300 GPUs, DDP, BF16 autocast
- Qwen3-VL-30B-A3B-Thinking, first 2 VLM layers, rank-16 LoRA
- 6-layer, width-384, 6-head action transformer
- two 224x224 frames, 15-tick categorical D29 prediction horizon
- global batch 384, per-GPU microbatch 4, no accumulation
- AdamW, LR `5e-5`, cosine decay, 200-step warm restart
- inverse-frequency clipped action weights from the dataset audit
- one full pass over the exact training-example count recorded by the index
- validation and automatic best-checkpoint export

Checkpoint 8 is loaded weights-only. This deliberately avoids coupling the
new 96-GPU run to the old 8-GPU optimizer/scheduler and shard cursor; it trains
one clean full-corpus pass from the learned checkpoint. This is safer than an
exact resume when changing global batch size.

Create a 24-node allocation using FAIR-CW's account/QoS names, then run:

```bash
cd /path/to/vla_foundry
zstd -d -f "$HOME/minecraft-vla-poc/transfer/checkpoint_8.pt.zst" \
  -o "$HOME/minecraft-vla-poc/transfer/checkpoint_8.pt"

VLA_ROOT="$HOME/minecraft-vla-poc" \
VLA_SOURCE=/s3/profiles/data-foundations-use2-1/minecraft_vpt \
VLA_DATA=/s3/profiles/data-foundations-use2-1/minecraft_vpt/vla_foundry_stream_index \
VLA_BASE_MODEL=/path/to/Qwen3-VL-30B-A3B-Thinking \
VLA_RESUME="$HOME/minecraft-vla-poc/transfer/checkpoint_8.pt" \
VLA_REPO_ROOT="$PWD" \
bash examples/minecraft/launch_discrete_full_fair_cw96.sh
```

Richmond exposes four GB300 GPUs per Slurm node. Required Slurm shape is
`--nodes=24 --ntasks-per-node=1 --gpus-per-node=4` with QoS
`g3_ocp_high` when that association is available. Confirm the account rather
than guessing it:

```bash
sacctmgr show user name="$USER" format=user,defaultaccount -p
sacctmgr show assoc where user="$USER" format=account%20,qos%130
```

Use one launcher task per node; each launcher starts four local torchrun
workers. Do not start 96 Slurm tasks and then run torchrun again.

Run the multinode preflight inside the allocation before training:

```bash
VLA_ROOT="$HOME/minecraft-vla-poc" \
VLA_SOURCE=/s3/profiles/data-foundations-use2-1/minecraft_vpt \
VLA_DATA=/s3/profiles/data-foundations-use2-1/minecraft_vpt/vla_foundry_stream_index \
VLA_BASE_MODEL=/path/to/Qwen3-VL-30B-A3B-Thinking \
VLA_RESUME="$HOME/minecraft-vla-poc/transfer/checkpoint_8.pt" \
VLA_REPO_ROOT="$PWD" \
bash examples/minecraft/preflight_fair_cw96.sh
```

It checks all 24 nodes independently for four CUDA devices and visibility of
the repository, raw VPT corpus, checkpoint, model, and all index manifests.

Before reserving all 24 nodes, run a one-node throughput/correctness smoke by
overriding the expected world size and training budget:

```bash
VLA_EXPECTED_WORLD_SIZE=4 VLA_TOTAL_SAMPLES=4096 \
VLA_TOTAL_VAL_SAMPLES=1024 VLA_NUM_CHECKPOINTS=1 \
VLA_GLOBAL_BATCH_SIZE=16 VLA_NAME=minecraft-vpt-stream-smoke \
VLA_SOURCE=/s3/profiles/data-foundations-use2-1/minecraft_vpt \
VLA_DATA=/s3/profiles/data-foundations-use2-1/minecraft_vpt/vla_foundry_stream_index \
VLA_BASE_MODEL=/path/to/Qwen3-VL-30B-A3B-Thinking \
VLA_RESUME="$HOME/minecraft-vla-poc/transfer/checkpoint_8.pt" \
VLA_REPO_ROOT="$PWD" \
bash examples/minecraft/launch_discrete_full_fair_cw96.sh
```

Confirm all four ranks make steady progress and storage read throughput is
acceptable before submitting the 96-GPU job.

The launcher validates every input, computes the exact dataset sample count
and class weights, trains, exports a policy bundle, and evaluates the held-out
test split. Override `VLA_GPUS_PER_NODE`, `VLA_EXPECTED_WORLD_SIZE`, batch, and
paths through environment variables if FAIR-CW differs.

## Native all-29 fine-tune

The completed balanced dataset is at:

```text
/checkpoint/data/ericli11/minecraft-vla-poc/dataset-craftium-d29-balanced
```

It is 3.7 GB and contains 145,000 train, 14,500 validation, and 14,500 test
sequences. The split is by run ID (no episode leakage), conversion reported no
failures, and every one of the 29 actions occurs in every split. Copy this
directory if FAIR-CW does not share the current `/checkpoint` namespace.

After the foundation launcher succeeds, use the same 96-GPU allocation shape:

```bash
VLA_ROOT="$HOME/minecraft-vla-poc" \
VLA_ALL29_DATA=/path/to/dataset-craftium-d29-balanced \
VLA_BASE_MODEL=/path/to/Qwen3-VL-30B-A3B-Thinking \
VLA_FOUNDATION_RUN="$HOME/minecraft-vla-poc/runs/minecraft-qwen3-d29-full-fair-cw96" \
VLA_REPO_ROOT="$PWD" \
bash examples/minecraft/launch_discrete_all29_fair_cw96.sh
```

This stage selects the foundation checkpoint with the lowest validation loss,
loads it weights-only, trains 10 balanced-D29 epochs at LR `1e-5`, exports a
29-action bundle, evaluates it, and enforces the all-29 offline quality gate.
The resulting policy bundle is small enough to copy back for Craftium rollout
verification; the eight environment rollouts still require the Craftium
runtime rather than training infrastructure alone.

## Acceptance checks

Training is usable only when all of these hold:

1. Validation loss improves from the starting checkpoint and does not diverge.
2. Offline policy macro-F1 beats random-prior, shuffled-language, and
   black-image ablations.
3. Exported `policy.json` lists the expected supported action IDs.
4. The exported policy completes a nonblank native-D29 Craftium rollout.
5. After native all-29 fine-tuning, the eight crafting/smelting actions pass
   the per-action offline gate and the eight isolated Craftium task rollouts.

The VPT corpus contains targets for only 21 of 29 semantic actions. Do not
claim an all-29 policy from this stage alone; actions 18–25 require the native
Craftium dataset and `qwen3_discrete29_all29.yaml` fine-tune.
