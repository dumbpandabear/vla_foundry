# Minecraft VLA proof of concept

For the current supervised VLA training run, see [TRAINING.md](TRAINING.md),
`qwen3_lora.yaml`, and `train_qwen.slurm`. It adapts the locally downloaded
Qwen3-VL-30B-A3B-Thinking with LoRA and a Foundry flow-matching action head.
It uses instruction labels and raw floating point VPT actions. Craftium is a
downstream benchmark. The initial integration experiment is documented below.

This uses VLA Foundry's **native training entrypoint, robotics WebDataset
pipeline, ViT/VLM backbone, and diffusion policy**. The initial experiment uses
a small model initialized from scratch, since Hugging Face model downloads are
blocked from this host. It is an integration POC, not a demonstrated Minecraft
player. A pretrained backbone and larger dataset are the next training stage.

## Data

The April 29 Qwen train/validation JSONL files contain instructions, video paths,
and VPT controls. The converter opens those video paths directly; directory
listings from the mounted S3 filesystem can stop after 10,000 entries.

The POC selects 128 train, 32 validation, and 32 test clips with at most four
clips per source recording. Whole recordings are assigned to deterministic
80/10/10 partitions. This deliberately regroups the source train/validation
records to avoid adjacent clips appearing in different partitions. There was
no separate test manifest in `json_data/`. The selected examples are an early
stream subset, not a representative sample of all 1,951 hours.

Each sample has one RGB observation, a text goal, and eight future controls at
20 FPS. The initial alignment is observation frame t to action t+1; this is
explicit in `--action-offset` and saved in metadata. Sampled annotated actions
match the prefix of `data_split.jsonl`, with trailing actions omitted. The
converter never pads missing action labels and excludes null-action windows.
Exact VPT observation/action timing still needs an upstream provenance audit
before scaling the run.

Controls are 23 binary buttons encoded as -1/+1 plus [pitch, yaw] degrees
divided by 180. These are already normalized; Foundry normalization is disabled.
The identity `stats.json` only describes this schema for Foundry configuration,
not empirical population statistics. Diffusion flow matching uses MSE over the
action vector; inference thresholds button outputs at zero.

```bash
python -m vla_foundry.data.minecraft.prepare \
  --data-root /bbfs/s3/profiles/jepa--use2-az3--x-s3/data/minecraft_vpt \
  --sources /bbfs/s3/profiles/jepa--use2-az3--x-s3/data/minecraft_vpt/json_data/train_qwen35_9b_04_29_26.jsonl \
            /bbfs/s3/profiles/jepa--use2-az3--x-s3/data/minecraft_vpt/json_data/val_qwen35_9b_04_29_26.jsonl \
  --output /storage/home/ericli11/minecraft-vla-poc/dataset-v1
```

Use a new output directory when regenerating. Prepared shards, checkpoints,
and exported policies live under persistent home rather than modifying source
data. `selected_clips.jsonl` records source file, line, recording, and split.

## Training

The local `.venv` inherits the existing Python 3.12/PyTorch installation at
`/engshare/conda/xlformers_llama4_flagship_conda-mtlsfixed`. Missing Foundry
dependencies are installed only in `.venv`; the shared environment is not
changed. This host has PyTorch 2.12 nightly, rather than Foundry's locked 2.7.0.
For a fresh installation elsewhere, use the repository's supported `uv sync`
environment and install the private Craftium wheel for rollout.

From the VLA Foundry repository root:

```bash
sbatch --qos=h200_dev --gpus=1 --cpus-per-task=8 --mem=32G \
  --time=00:30:00 --job-name=minecraft-vla-poc \
  --output=/storage/home/ericli11/minecraft-vla-poc/train-%j.log \
  examples/minecraft/train.slurm
```

`poc.yaml` runs 1,024 optimizer steps (4,096 samples including repeated passes),
saves four checkpoints, validates each, then exports the checkpoint with lowest
validation denoising loss. Generated-action test metrics include button F1,
button accuracy, camera MAE, and a no-op baseline. Test data does not select the
checkpoint. For another run, change the dataset paths, run name, and output
bundle paths in the configuration and launcher.

## Craftium import

The exported directory includes model weights, model/processor configurations,
backbone architecture, and the versioned action schema. It has no dependency
on source videos or training manifests at inference. Keep the Foundry checkout
with the Minecraft processor extension available on `PYTHONPATH`.

The current Craftium harness uses Discrete(29). VPT permits simultaneous keys
and continuous camera motion, so the adapter is explicitly lossy. It logs
dropped controls, selects slots before camera before movement/interactions,
and replans every environment step. VPT inventory clicks do not supervise
Craftium's direct craft/smelt primitives; this policy cannot claim to learn
those actions from these labels. A later D29-specific Craftium fine-tuning
dataset or a compatible Dict action environment is needed for that coverage.

On a GPU node with the private Craftium wheel installed:

```bash
export PYTHONPATH=/storage/home/ericli11/Projects/vla_foundry
export SDL_VIDEODRIVER=offscreen
VLA_ENV=/storage/home/ericli11/Projects/vla_foundry/.venv
export LD_LIBRARY_PATH="$VLA_ENV/lib/python3.12/site-packages/craftium.libs:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="/etc/glvnd/egl_vendor.d:/usr/share/glvnd/egl_vendor.d:$VLA_ENV/lib/python3.12/site-packages/craftium.libs/glvnd/egl_vendor.d"
"$VLA_ENV/bin/python" \
  /storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium/scripts/run_minecraft_vla.py \
  --bundle /storage/home/ericli11/minecraft-vla-poc/policy-v1 \
  --task mine_cobblestone --goal 'Mine stone blocks using a pickaxe.' \
  --steps 64 --output /storage/home/ericli11/minecraft-vla-poc/rollout-v1
```

This creates an isolated Craftium environment and records actions, reward, a
video, and a result JSON. A successful import/rollout is separate from task
success; zero reward means the policy did not solve the task during that run.

## CPU rendering and the GB300 checkpoint

`render_cpu.slurm` tests software rendering on a CPU worker. The private wheel's
bundled `libEGL_mesa.so.0` requires a missing `libgallium-26.1.1.so`. The launchers
use an EGL vendor JSON pointing to the **absolute system Mesa library path**;
the standard Mesa JSON uses a relative library name and selects the broken
bundled library when `craftium.libs` is on `LD_LIBRARY_PATH`.

```bash
sbatch --account=data --qos=cpu_lowest --cpus-per-task=4 --mem=8G \
  --time=00:05:00 --output=/storage/home/ericli11/minecraft-vla-poc/cpu-render-%j.log \
  examples/minecraft/render_cpu.slurm
sbatch --account=data --qos=cpu_lowest --cpus-per-task=4 --mem=12G \
  --time=00:10:00 --output=/storage/home/ericli11/minecraft-vla-poc/rollout-cpu-%j.log \
  examples/minecraft/rollout_cpu.slurm
```

The supplied GB300 recipe is a separate **vision-action** model: its ViT has
no text encoder. Its 20 keys are in 0/1 units, camera targets use train-only
percentile normalization, and it expects frames t-1 and t with no past actions.
It is incompatible with the 25-dimensional byte-language baseline above.
`vpt_vit.py` exports and loads that recipe with explicit schema checks, restores
camera degrees, thresholds keys at 0.5, and preserves the previous RGB frame.
The harness renders at 640×360 and the saved processor resizes to 128×128.
The initial history frame repeats the reset observation.

Run this export on the cluster holding the actual checkpoint and train stats:

```bash
python -m vla_foundry.inference.minecraft.vpt_vit \
  --run /path/to/best/experiment \
  --checkpoint /path/to/best/experiment/checkpoints/checkpoint_N.pt \
  --train-stats /checkpoint/data-readiness/ericli11/vla_poc/data/vpt10_128/train/shards/stats.json \
  --output /path/to/portable-vpt-policy
```

Transfer the resulting bundle plus this checkout's Minecraft inference code to
the x86 cluster. The CPU launcher accepts `VLA_BUNDLE=/path/to/portable-vpt-policy`
in its environment. We cannot export or verify the real GB300 checkpoint until
its weights and saved model/processor configs are available here. The portable
loader is exercised with a synthetic native ViT checkpoint in `test_vpt_port.py`;
that test checks import mechanics and camera units, not trained performance.
