# Minecraft Qwen3-VL VLA POC

The 100,000-example POC completed in job 15718347. Checkpoint 5 was exported;
test macro-F1 is 0.2086 versus a 0.1223 training-frequency baseline. Three
2,000-step Craftium rollouts returned zero reward. Full results and artifacts:
`/storage/home/ericli11/minecraft-vla-poc/RESULTS.md`.

Train offline on human VPT actions, then benchmark the exported policy in
Craftium. The previous native model trained from scratch is retained as an
integration baseline; its queued larger run was canceled when Qwen became
available.

## Data

`/storage/home/ericli11/minecraft-vla-poc/dataset-v2/` contains native Foundry
WebDataset shards and provenance records:

| Split | Recordings | Training examples |
|---|---:|---:|
| Train | 685 | 183,645 |
| Validation | 68 | 17,423 |
| Test | 66 | 18,389 |

Selection uses 20,000 / 2,000 / 2,000 clips, capped at 32 per recording. This
bounded subset represents roughly 21 hours before filtering, not the entire
approximately 1,951-hour corpus. Source ordering affects selection; it is not
uniform sampling across the entire corpus.

Each example has RGB frames t-1 and t, squashed to 224×224, a language annotation,
and 15 future ticks t..t+14 at 20 Hz. No previous actions are supplied. Targets
contain 23 buttons encoded as -1/+1 and floating point pitch/yaw degrees / 40.
Camera labels come from raw mouse dy/dx × 360/2400, clipped at ±40°, rather than
rounded annotation arrays. Whole recordings stay in one split (hash, seed 42).
GUI windows, no-op anchors and incomplete future windows are excluded. Initial
stuck attack is suppressed and hotbar changes become one-tick presses.

## Model and fine-tuning

Config: `examples/minecraft/qwen3_lora.yaml`.

- Base: `/checkpoint/data/ericli11/models/Qwen3-VL-30B-A3B-Thinking`.
- Foundry `vlm_backbone` loads the pretrained MoE VLM in BF16 with SDPA.
- Qwen's own image processor, tokenizer and chat template handle the inputs.
  Image processing is bounded at 224×224 area, giving 49 merged tokens per frame.
- A new action probe token is appended; its final two layer representations
  condition the action head. Text generation and reasoning are not training targets.
- PEFT 0.18.1 LoRA rank 16, alpha 32, dropout 0.05 adapts attention q/k/v/o
  projections. The new probe embedding is also trainable. Vision, expert/router,
  original embedding and other base weights remain frozen.
- The action transformer is width 384, six layers/six heads; it predicts a
  `[batch,15,25]` action chunk using native Foundry flow matching.
- AdamW lr 1e-4, cosine schedule, 200 warmup optimizer steps, BF16 autocast,
  gradient checkpointing, gradient clipping 1.0. Microbatch 32, effective batch 32.
  Data-loader workers are disabled for this POC to avoid OpenMP failures after fork.
- Initial POC budget: 100,000 examples presented (3,125 optimizer steps), five
  checkpoints. This is an initial fine-tuning experiment, not a converged policy.
- Validation denoising loss chooses the checkpoint; held-out test actions do not
  participate in training or checkpoint selection.
- `validate_on_resume: true` validates the restored checkpoint before consuming
  more training examples, recovering the validation stage after an interruption.

Compact checkpoints contain all trainable tensors and persistent buffers plus
separate optimizer state. Restoring requires the original pretrained model and
checks that every trainable parameter is supplied. The exported policy references
the local base model; it does not duplicate 60+ GB of frozen weights. To move
clusters, copy the base model as well and update the model/processor paths.

## Launch and artifacts

From `/storage/home/ericli11/Projects/vla_foundry`:

```bash
sbatch --account=data --qos=h200_dev --gres=gpu:1 --cpus-per-task=8 \
  --mem=180G --time=00:45:00 \
  --output=/storage/home/ericli11/minecraft-vla-poc/qwen-check-%j.log \
  examples/minecraft/check_qwen.slurm
sbatch --account=data --qos=h200_dev --gres=gpu:1 --cpus-per-task=12 \
  --mem=180G --time=24:00:00 --dependency=afterok:CHECK_JOB_ID \
  --output=/storage/home/ericli11/minecraft-vla-poc/train-qwen-%j.log \
  examples/minecraft/train_qwen.slurm
```

The training script resumes any valid existing checkpoint, runs training, exports the best validation checkpoint,
evaluates 512 examples spread across the test stream (with instruction/image
ablations), then runs three 2,000-step Craftium rollouts at seeds 42/43/44.
Shell error handling stops the pipeline when any required stage fails.

Artifacts under `/storage/home/ericli11/minecraft-vla-poc/`:

- `runs/minecraft-qwen3-lora-poc/`: logs, resolved config, adapter/head checkpoints.
- `policy-qwen3-lora-poc/`: exported adapter/head weights, configs, provenance.
- `test-qwen3-lora-poc.json`: per-key/macro F1 and camera metrics, ablations/no-op.
- `dataset-v2/train/action_prior.json`: training-only button frequencies for
  random-frequency and constant-majority baselines in the same offline report.
- `rollout-qwen3-<job>-seed<seed>/`: action log, reward, and rollout video.

## Craftium benchmark

Render the VoxeLibre task at 320×180, squash to the training image size, use
frameskip 3 (roughly 18.5 Hz), and execute eight predicted ticks before replanning.
Continuous dictionary controls preserve simultaneous movement/dig/camera commands.
Camera commands use Craftium's native degree interface; VPT yaw is negated because
Luanti positive yaw turns left. Unsupported controls are filtered against the
actual environment action space. This mapping does not supply inventory/crafting
skills that are absent from the VPT labels.

`calibrate_camera.slurm` independently verifies known ±5° native camera commands
against reported pitch/yaw. Rollouts log applied camera commands and player angles.
Offline accuracy is not a claim of successful play; benchmark rewards are reported
as observed, including zero reward.

## Checks

All 13 downloaded safetensors shards passed header/length checks (62,141,508,064
bytes of tensor payload). The reduced Qwen MoE check passed action forward/backward,
confirmed adapter/head gradients with no expert/vision gradients, and restored a
compact state with identical predictions. A native training/validation/checkpoint
smoke test also completed, and the CPU Craftium camera check measured exactly
±5° for both axes. The full checkpoint passed forward/backward at microbatch 32: approximately
65.8 GB peak GPU memory and 7.04 seconds per steady training step on one H200.
Craftium rendering is checked separately from training. SDL must explicitly
select Mesa’s software EGL device on GPU workers; device selection is performed
by `inference/minecraft/renderer.py` before starting the environment.
The bundled SDL2 lists only eight EGL devices, which omits Mesa on an eight-GPU
worker. `software_egl.c` filters SDL's device list to software devices in this
case; the small loader is compiled with `cc` and cached under
`~/.cache/minecraft-vla/renderer/`. Standard EGL calls still use GLVND.
GPU-worker calibration job 15715177 completed with nonblank 320×180 frames and
exact ±5° pitch/yaw responses.
The continuous adapter also has a separate smoke runner:
`examples/minecraft/run_craftium.slurm` accepts the arguments for
`scripts/run_minecraft_vla.py` (bundle, device, steps, seed, action mode, output).
Job 15715345 completed 32 continuous-control steps with the reduced Qwen
integration model, including chunk replanning, applied-action logs and a video.
Its zero reward is only an integration result, not a measurement of the full
pretrained Qwen policy being trained.
The first full Qwen checkpoint (20,128 examples / step 629) passed compact restore,
BF16 evaluation and ten-step flow generation at batch 32 in job 15718337, with
77.7 GB peak memory. Transformers 4.57's MoE router requires logits and routing
probabilities to share a dtype; the HF wrapper matches gate outputs to their input
dtype to prevent autocast evaluation failures. Frozen weights are unchanged.
An interrupted run can resume from the latest valid paired model/optimizer checkpoint. No performance claims are based on these connectivity checks.
