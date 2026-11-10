# Qwen3-VL categorical Craftium D29 POC

Config: `qwen3_discrete29.yaml`; launcher: `train_discrete.slurm`.

This experiment starts from the original local Qwen3-VL-30B-A3B-Thinking
weights with fresh rank-16 attention LoRA, a learned action token, and a fresh
six-layer, width-384, six-head Foundry transformer. Fifteen learned tick queries
produce `[batch,15,29]` categorical logits. Cross-entropy trains the targets;
there is no diffusion scheduler or numeric regression on class IDs. Labels and
previous actions never enter the model forward pass. The original continuous
policy and its run are preserved.

## Labels

Reuses the existing `dataset-v2` train/val/test WebDataset shards without
duplicating images. Each example supplies adjacent 224×224 RGB frames and
an instruction. VPT targets are mapped to native D29 class IDs by the versioned
`craftium_d29_nearest_camera_v1` conversion:

1. Hotbar presses select the first pressed slot.
2. Otherwise, camera movement ≥5° selects one look action on the dominant axis.
   A native Craftium look step is 10°; smaller movements round to zero.
   Minecraft-positive yaw means `look_right`; the native D29 wrapper handles
   Luanti's opposite numeric yaw sign.
3. Otherwise, use maps to place.
4. Otherwise, forward/jump/attack map to the available movement/dig combinations.
   Forward+dig alone maps to dig, because the vocabulary has no forward+dig action.

This is a lossy projection: simultaneous look+move, strafing/backward movement,
small mouse motion, and large camera magnitudes cannot be represented exactly.
Reported D29 accuracy measures imitation of these converted labels, not exact
reproduction of all original human controls or successful gameplay.

The complete label audit is
`/storage/home/ericli11/minecraft-vla-poc/d29-label-audit.json`:

- Train: 183,645 examples / 2,754,675 ticks; all 21 labelable actions are present.
- Validation: 17,423 examples; test: 18,389 examples.
- Four look actions account for 334,428 train ticks, about 12.1%.
- Camera quantization MAE before learning: pitch 0.6814°, yaw 1.0681° per tick.
- Eight craft/smelt primitives have no positive labels. They are retained in the
  29-logit head but disabled during inference until suitable demonstrations exist.

## Training and evaluation

- One H200, 12 CPUs, 180 GB RAM, twelve-hour allocation.
- Initial budget: 100,000 examples presented / 3,125 optimizer steps, batch 32.
  This is approximately 0.54 passes over the available training examples.
- AdamW lr 1e-4, cosine decay, 200 warmup steps, weight decay 0.01.
- BF16, gradient checkpointing, gradient clip 1.0, no FSDP, no augmentation.
- Standard unweighted cross-entropy; z-loss disabled. Frequency imbalance is
  visible in the audit and per-action metrics; no test frequencies fit training.
- Five checkpoints, each evaluated on 2,048 validation examples. Lowest validation
  cross-entropy selects the export without looking at test outcomes.
- Evaluation uses 2,048 evenly spread test examples. Reports tick-0 and full-chunk
  accuracy, macro-F1 over the 21 supported actions, per-action F1, confusion
  matrices, camera-action accuracy/F1, majority/random-frequency/no-op baselines,
  and shuffled-instruction/black-image ablations.

Expected paths under `/storage/home/ericli11/minecraft-vla-poc/`:

- `runs/minecraft-qwen3-d29-poc/`: checkpoints, optimizer states and training log.
- `policy-qwen3-d29-poc/`: compact exported adapters/head and label provenance.
- `test-qwen3-d29-poc.json`: offline evaluation after training.
- `train-d29-JOB_ID.log`: Slurm output.

Run `sbatch examples/minecraft/train_discrete.slurm` from Foundry. The launcher
trains, exports, then evaluates. It requires the completed label audit.
To benchmark the exported policy in Craftium, use the existing software-rendering
launcher with `--action-mode discrete_29 --replan-ticks 4 --steps 2000`, supplying
the D29 bundle and a fresh output directory. The adapter sends native integer
action IDs directly to the environment. Benchmark rewards are separate from
offline action accuracy.
