"""Versioned VPT-to-Craftium D29 supervision; no synthetic crafting labels."""
import numpy as np

from vla_foundry.data.minecraft.actions import BUTTONS, D29_NAMES

DISCRETE_SCHEMA = "craftium_d29_nearest_camera_v1"
CAMERA_DELTA = 10.0
SUPPORTED_IDS = tuple(i for i in range(29) if not 18 <= i <= 25)


def discretize(actions, camera_scale=40.0, camera_delta=CAMERA_DELTA):
    """Map [...,25] normalized VPT controls to [...] class IDs.

    Priority: hotbar, camera, place, then movement/dig combinations. Camera
    turns below half a native turn are rounded to zero. Larger turns use one
    native turn on the dominant axis. Simultaneous unsupported controls are
    lost, as required by the one-action-per-tick interface.
    """
    values = np.asarray(actions, dtype=np.float32)
    if values.shape[-1] != 25 or not np.isfinite(values).all():
        raise ValueError("Expected finite VPT action vectors of length 25")
    pressed = values[..., :23] > 0

    def at(name):
        return pressed[..., BUTTONS.index(name)]

    combo = at("forward").astype(np.int64) + 2 * at("jump") + 4 * at("attack")
    labels = np.asarray([0, 1, 2, 26, 3, 3, 27, 28], dtype=np.int64)[combo]
    labels = np.where(at("use"), 4, labels)
    camera = values[..., -2:].clip(-1, 1) * camera_scale
    pitch, yaw = camera[..., 0], camera[..., 1]
    yaw_dominant = np.abs(yaw) >= np.abs(pitch)
    look = np.where(yaw_dominant, np.where(yaw > 0, 15, 14), np.where(pitch > 0, 17, 16))
    labels = np.where(np.abs(camera).max(axis=-1) >= camera_delta / 2, look, labels)
    slots = np.stack([at(f"hotbar.{i}") for i in range(1, 10)], axis=-1)
    labels = np.where(slots.any(axis=-1), slots.argmax(axis=-1) + 5, labels)
    return labels.astype(np.int64)


def action_targets(actions):
    """Return D29 targets from raw VPT controls or native D29 demonstrations.

    Existing VPT datasets store 25 floating-point controls per tick. Craftium
    demonstrations may instead store one integer D29 ID per tick in a trailing
    singleton dimension so they can supervise semantic craft/smelt primitives
    that have no equivalent VPT button.
    """
    values = np.asarray(actions)
    if values.ndim < 1:
        raise ValueError("Expected actions with a trailing feature dimension")
    if values.shape[-1] == 25:
        return discretize(values)
    if values.shape[-1] != 1:
        raise ValueError("Expected raw VPT actions [...,25] or native D29 labels [...,1]")
    labels = values[..., 0]
    if not np.isfinite(labels).all():
        raise ValueError("Expected finite native D29 labels")
    rounded = np.rint(labels)
    if not np.array_equal(labels, rounded) or (rounded < 0).any() or (rounded >= len(D29_NAMES)).any():
        raise ValueError("Native D29 labels must be integer IDs in [0, 29)")
    return rounded.astype(np.int64)


def projected_camera(labels, camera_delta=CAMERA_DELTA):
    """Camera degrees in Minecraft sign convention, for quantization audits."""
    labels = np.asarray(labels)
    pitch = np.where(labels == 17, camera_delta, np.where(labels == 16, -camera_delta, 0.))
    yaw = np.where(labels == 15, camera_delta, np.where(labels == 14, -camera_delta, 0.))
    return np.stack([pitch, yaw], axis=-1)
