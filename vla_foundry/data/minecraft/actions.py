"""Versioned VPT action vector and explicit Craftium projections."""

import numpy as np

BUTTONS = (
    "ESC",
    "back",
    "drop",
    "forward",
    *(f"hotbar.{i}" for i in range(1, 10)),
    "inventory",
    "jump",
    "left",
    "right",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
)
ACTION_DIM = len(BUTTONS) + 2
CAMERA_SCALE = 180.0
SCHEMA_VERSION = "vpt_buttons_camera_v1"
D29_NAMES = (
    "nop",
    "forward",
    "jump",
    "dig",
    "place",
    *(f"slot_{i}" for i in range(1, 10)),
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "craft_planks",
    "craft_crafting_table",
    "craft_stick",
    "craft_wooden_pickaxe",
    "craft_stone_pickaxe",
    "craft_furnace",
    "smelt_iron_ingot",
    "craft_iron_pickaxe",
    "forward_jump",
    "jump_dig",
    "forward_jump_dig",
)


def encode_action(action: dict) -> np.ndarray:
    """Encode buttons as -1/+1 and camera [pitch, yaw] degrees / 180."""
    camera = np.asarray(action.get("camera", [0, 0]), dtype=np.float32)
    if camera.shape != (2,) or not np.isfinite(camera).all() or np.abs(camera).max() > CAMERA_SCALE:
        raise ValueError(f"Invalid camera degrees: {camera}")
    buttons = []
    for key in BUTTONS:
        value = action.get(key, 0)
        if value not in (0, 1, False, True):
            raise ValueError(f"Invalid binary control {key}: {value}")
        buttons.append(1.0 if value else -1.0)
    return np.concatenate((np.asarray(buttons, dtype=np.float32), camera / CAMERA_SCALE))


def decode_action(vector, threshold: float = 0.0, camera_scale: float = CAMERA_SCALE) -> dict:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (ACTION_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected {ACTION_DIM} finite action values")
    result = {key: int(value > threshold) for key, value in zip(BUTTONS, vector[:-2], strict=True)}
    result["camera"] = (vector[-2:].clip(-1, 1) * camera_scale).tolist()
    return result


def to_craftium_dict(action: dict, camera_limit: float = 10.0) -> dict:
    """Map controls; callers must filter against the actual env's supported keys."""
    names = {"back": "backward", "attack": "dig", "use": "place"}
    supported = {"forward", "jump", "left", "right", "sneak", "sprint", "inventory", "pickItem"}
    result = {names.get(key, key): int(action.get(key, 0)) for key in names.keys() | supported}
    result.update({f"slot_{i}": int(action.get(f"hotbar.{i}", 0)) for i in range(1, 10)})
    result["camera"] = np.asarray(action["camera"], dtype=np.float32).clip(-camera_limit, camera_limit)
    return result


def to_discrete29(action: dict, camera_deadzone: float = 2.0) -> tuple[int, dict]:
    """Lossy projection to one D29 step, with diagnostics for dropped controls.

    Slot selection takes priority, then camera, then movement/interactions.
    Craft/smelt primitives have no supervised VPT labels and are never invented.
    """
    mapped = to_craftium_dict(action)
    active = {k for k, v in mapped.items() if k != "camera" and v}
    chosen = set()
    camera = mapped["camera"]
    slots = sorted(k for k in active if k.startswith("slot_"))
    if slots:
        name = slots[0]
        chosen.add(name)
    elif np.abs(camera).max() >= camera_deadzone:
        pitch, yaw = camera
        if abs(yaw) >= abs(pitch):
            name = "look_right" if yaw > 0 else "look_left"
        else:
            name = "look_down" if pitch > 0 else "look_up"
    elif "place" in active:
        name = "place"
        chosen.add("place")
    else:
        combo = active & {"forward", "jump", "dig"}
        name = {
            frozenset(): "nop",
            frozenset({"forward"}): "forward",
            frozenset({"jump"}): "jump",
            frozenset({"dig"}): "dig",
            frozenset({"forward", "jump"}): "forward_jump",
            frozenset({"jump", "dig"}): "jump_dig",
            frozenset({"forward", "jump", "dig"}): "forward_jump_dig",
            frozenset({"forward", "dig"}): "dig",
        }[frozenset(combo)]
        chosen = combo if combo != {"forward", "dig"} else {"dig"}
    unsupported = sorted(k for k in ("ESC", "drop", "swapHands") if action.get(k))
    return D29_NAMES.index(name), {"name": name, "dropped_controls": sorted(active - chosen) + unsupported}
