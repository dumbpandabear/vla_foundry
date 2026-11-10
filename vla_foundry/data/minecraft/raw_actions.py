"""Decode raw VPT controls without integer camera truncation."""

import json
from pathlib import Path

import numpy as np

from vla_foundry.data.minecraft.actions import BUTTONS

RAW_SCHEMA = "vpt_buttons_float_camera_v2"
RAW_CAMERA_SCALE = 40.0
KEY_CODES = {
    "ESC": "escape",
    "back": "s",
    "drop": "q",
    "forward": "w",
    "inventory": "e",
    "jump": "space",
    "left": "a",
    "right": "d",
    "sneak": "left.shift",
    "sprint": "left.control",
    "swapHands": "f",
    **{f"hotbar.{i}": str(i) for i in range(1, 10)},
}


def raw_path(root: Path, recording: str):
    versions = ["10.0"] + sorted(p.name for p in (root / "data").iterdir() if p.name != "10.0")
    for version in versions:
        path = root / "data" / version / (recording + ".jsonl")
        if path.is_file():
            return path
    raise FileNotFoundError(f"No raw VPT action log for {recording}")


def load_raw_actions(path: Path):
    """Return vectors, GUI/no-op flags and camera timing audit from a recording."""
    vectors, invalid = [], []
    previous_hotbar, stuck_attack = None, None
    previous = None
    timing_errors, small_moves, missing_hotbar = [], 0, 0
    with path.open(errors="replace") as stream:
        for line in stream:
            row = json.loads(line)
            mouse = row["mouse"]
            keys = set(row["keyboard"]["keys"])
            vector = np.full(len(BUTTONS) + 2, -1, dtype=np.float32)
            for index, key in enumerate(BUTTONS):
                vector[index] = 1 if "key.keyboard." + KEY_CODES.get(key, "__unknown__") in keys else -1
            buttons = set(mouse["buttons"])
            if stuck_attack is None:
                stuck_attack = 0 in buttons
            if 0 not in buttons:
                stuck_attack = False
            for key, pressed in (
                ("attack", 0 in buttons and not stuck_attack),
                ("use", 1 in buttons),
                ("pickItem", 2 in buttons),
            ):
                vector[BUTTONS.index(key)] = 1 if pressed else -1
            # Some otherwise valid VPT logs omit the absolute selected-slot
            # field on individual ticks.  In that case retain the number-key
            # controls decoded above instead of aborting the whole recording.
            # Do not invent a slot transition or change the last known state.
            if "hotbar" in row:
                hotbar = int(row["hotbar"])
                for slot in range(9):
                    vector[BUTTONS.index(f"hotbar.{slot + 1}")] = (
                        1 if (previous_hotbar is not None and hotbar != previous_hotbar and hotbar == slot) else -1
                    )
                previous_hotbar = hotbar
            else:
                missing_hotbar += 1
            camera = np.asarray([mouse["dy"], mouse["dx"]], dtype=np.float32) * (360.0 / 2400.0)
            if not np.isfinite(camera).all():
                raise ValueError(f"Nonfinite raw camera: {path}")
            small_moves += int(np.any((np.abs(camera) > 0) & (np.abs(camera) < 1)))
            vector[-2:] = camera.clip(-RAW_CAMERA_SCALE, RAW_CAMERA_SCALE) / RAW_CAMERA_SCALE
            gui = bool(row.get("isGuiOpen", False))
            invalid.append(gui)
            vectors.append(vector)
            if previous is not None and not gui and not previous["gui"]:
                actual = np.asarray([row["pitch"] - previous["pitch"], row["yaw"] - previous["yaw"]])
                actual[1] = (actual[1] + 180) % 360 - 180
                if abs(previous["pitch"] + previous["camera"][0]) < 89:
                    timing_errors.append(float(np.max(np.abs(actual - previous["camera"]))))
            previous = {"pitch": row["pitch"], "yaw": row["yaw"], "camera": camera, "gui": gui}
    vectors = np.stack(vectors)
    noop = (vectors[:, :-2] < 0).all(axis=-1) & (vectors[:, -2:] == 0).all(axis=-1)
    return (
        vectors,
        np.asarray(invalid),
        noop,
        {
            "raw_path": str(path),
            "ticks": len(vectors),
            "missing_hotbar_ticks": missing_hotbar,
            "subdegree_camera_ticks": small_moves,
            "camera_next_state_error_p50_degrees": float(np.median(timing_errors)) if timing_errors else None,
            "camera_next_state_error_p95_degrees": float(np.percentile(timing_errors, 95)) if timing_errors else None,
        },
    )
