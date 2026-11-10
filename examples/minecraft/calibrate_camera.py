"""Check native Craftium degree camera commands before the policy benchmark."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vla_foundry.inference.minecraft.renderer import select_software_egl_device

sys.path.insert(0, "/storage/home/ericli11/Projects/jepa-internal/app/agent-harness/craftium")
from utils import create_e2e_env

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
select_software_egl_device()
env = create_e2e_env(
    obs_width=320,
    obs_height=180,
    minimal_task="mine_cobblestone",
    lua_seed=42,
    frameskip=3,
    max_timesteps=20,
    action_mode="continuous",
    run_dir_prefix=str(args.output / "world"),
)
env.unwrapped.max_retries = 1
report = []
try:
    obs, info = env.reset(seed=42)
    render_std = float(np.asarray(obs).std())
    if render_std <= 2:
        raise RuntimeError("Craftium returned a blank observation")
    Image.fromarray(np.asarray(obs)).save(args.output / "initial.png")
    print(f"Nonblank Craftium observation: RGB std={render_std:.2f}", flush=True)
    for camera in ([0.0, 5.0], [0.0, -5.0], [5.0, 0.0], [-5.0, 0.0]):
        before = [float(info["player_pitch"]), float(info["player_yaw"])]
        action = {k: 0 for k in env.action_space.spaces if k != "camera"}
        action["camera"] = np.array(camera, dtype=np.float32)
        obs, _, _, _, info = env.step(action)
        after = [float(info["player_pitch"]), float(info["player_yaw"])]
        delta = [(b - a + 180) % 360 - 180 for a, b in zip(before, after, strict=True)]
        report.append({"command_degrees": camera, "before": before, "after": after, "delta_degrees": delta})
        axis = 0 if camera[0] else 1
        assert abs(abs(delta[axis]) - 5) < 0.5, report[-1]
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
finally:
    env.close()
