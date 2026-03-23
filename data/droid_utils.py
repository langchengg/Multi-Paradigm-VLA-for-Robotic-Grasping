"""
Shared DROID + Franka action utilities.

These helpers keep Notebook 2 and Notebook 3 aligned on the same:
- Franka delta-pose action interface
- DROID action conversion assumptions
- Hugging Face sample parsing helpers
"""

import io
import json

import numpy as np
from PIL import Image


FRANKA_ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
TRANSLATION_STEP_M = 0.03
ROTATION_STEP_RAD = 0.05
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSE_VALUE = 1.0
ACTION_MIN = -1.0
ACTION_MAX = 1.0
DROID_DEFAULT_FPS = 15.0

__all__ = [
    "ACTION_MAX",
    "ACTION_MIN",
    "DROID_DEFAULT_FPS",
    "FRANKA_ACTION_KEYS",
    "GRIPPER_CLOSE_VALUE",
    "GRIPPER_OPEN_VALUE",
    "ROTATION_STEP_RAD",
    "TRANSLATION_STEP_M",
    "droid_action_to_franka_action",
    "droid_cartesian_velocity_to_franka_action",
    "ensure_franka_action_7d",
    "gripper_value_to_binary_command",
    "image_to_uint8_array",
    "load_droid_task_lookup",
    "sample_get",
]


def ensure_franka_action_7d(
    action,
    source_name="<unknown>",
    action_min=ACTION_MIN,
    action_max=ACTION_MAX,
):
    """Convert actions to the repo's normalized 7-DOF Franka delta-pose interface."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 7:
        return np.clip(action, action_min, action_max)
    if action.shape[0] == 4:
        return np.array(
            [action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]],
            dtype=np.float32,
        )
    raise ValueError(
        f"Unsupported action dimension {action.shape[0]} in {source_name}. "
        "Expected 7-DOF Franka actions or legacy 4-DOF actions."
    )


def gripper_value_to_binary_command(
    value,
    gripper_open_value=GRIPPER_OPEN_VALUE,
    gripper_close_value=GRIPPER_CLOSE_VALUE,
):
    """Collapse continuous gripper annotations to this repo's open/close sign convention."""
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    if 0.0 <= scalar <= 1.0:
        return gripper_close_value if scalar >= 0.5 else gripper_open_value
    return gripper_close_value if scalar > 0.0 else gripper_open_value


def droid_cartesian_velocity_to_franka_action(
    cartesian_velocity,
    *,
    gripper_position=None,
    gripper_velocity=None,
    droid_fps=DROID_DEFAULT_FPS,
    translation_step_m=TRANSLATION_STEP_M,
    rotation_step_rad=ROTATION_STEP_RAD,
    action_min=ACTION_MIN,
    action_max=ACTION_MAX,
    gripper_open_value=GRIPPER_OPEN_VALUE,
    gripper_close_value=GRIPPER_CLOSE_VALUE,
    source_name="<unknown>",
):
    """
    Convert DROID Cartesian velocity commands into the normalized Franka delta-pose action.

    DROID stores real Franka commands at a fixed control rate. We integrate one control
    period, then normalize by the MuJoCo Franka controller's per-step translation and
    rotation scales used throughout this repo.
    """
    velocity = np.asarray(cartesian_velocity, dtype=np.float32).reshape(-1)
    if velocity.shape[0] < 6:
        raise ValueError(
            f"Unsupported DROID action dimension {velocity.shape[0]} in {source_name}. "
            "Expected at least 6 Cartesian velocity values."
        )

    delta_xyz = velocity[:3] / droid_fps
    delta_rpy = velocity[3:6] / droid_fps
    if gripper_position is not None:
        gripper = gripper_value_to_binary_command(
            gripper_position,
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
    elif gripper_velocity is not None:
        gripper = gripper_value_to_binary_command(
            gripper_velocity,
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
    else:
        gripper = gripper_open_value

    normalized = np.array(
        [
            delta_xyz[0] / translation_step_m,
            delta_xyz[1] / translation_step_m,
            delta_xyz[2] / translation_step_m,
            delta_rpy[0] / rotation_step_rad,
            delta_rpy[1] / rotation_step_rad,
            delta_rpy[2] / rotation_step_rad,
            gripper,
        ],
        dtype=np.float32,
    )
    return ensure_franka_action_7d(
        normalized,
        source_name=source_name,
        action_min=action_min,
        action_max=action_max,
    )


def droid_action_to_franka_action(
    action,
    *,
    translation_step_m=TRANSLATION_STEP_M,
    rotation_step_rad=ROTATION_STEP_RAD,
    action_min=ACTION_MIN,
    action_max=ACTION_MAX,
    gripper_open_value=GRIPPER_OPEN_VALUE,
    gripper_close_value=GRIPPER_CLOSE_VALUE,
    source_name="<unknown>",
):
    """
    Adapt flattened DROID actions to this repo's normalized Franka delta-pose interface.

    The Hugging Face DROID conversions expose a flattened action vector. When XYZ / RPY
    values are already near [-1, 1], we keep them as normalized controls; otherwise we
    treat them as metric / radian deltas and scale them into the shared Franka interface.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 6:
        raise ValueError(
            f"Unsupported DROID flattened action dimension {action.shape[0]} in {source_name}."
        )

    out = np.zeros(7, dtype=np.float32)
    out[:6] = action[:6]
    if action.shape[0] >= 7:
        out[6] = gripper_value_to_binary_command(
            action[6],
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
    else:
        out[6] = gripper_open_value

    if np.max(np.abs(out[:3])) > 1.0:
        out[:3] = out[:3] / translation_step_m
    if np.max(np.abs(out[3:6])) > 1.0:
        out[3:6] = out[3:6] / rotation_step_rad
    return ensure_franka_action_7d(
        out,
        source_name=source_name,
        action_min=action_min,
        action_max=action_max,
    )


def sample_get(sample, *paths):
    """Retrieve a possibly nested value from a Hugging Face sample dict."""
    for path in paths:
        if isinstance(sample, dict) and path in sample and sample[path] is not None:
            return sample[path]
        value = sample
        found = True
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break
        if found and value is not None:
            return value
    return None


def image_to_uint8_array(image, source_name):
    """Normalize PIL / Hugging Face image payloads to HWC uint8 arrays."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            arr = np.array(Image.open(io.BytesIO(image["bytes"])).convert("RGB"), dtype=np.uint8)
        elif image.get("path"):
            arr = np.array(Image.open(image["path"]).convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported image payload in {source_name}: {image.keys()}")
    elif hasattr(image, "to_pil"):
        arr = np.array(image.to_pil().convert("RGB"), dtype=np.uint8)
    elif hasattr(image, "numpy"):
        arr = np.asarray(image.numpy())
    elif hasattr(image, "asnumpy"):
        arr = np.asarray(image.asnumpy())
    else:
        arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims in {source_name}, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_droid_task_lookup(repo_id):
    """Map DROID task indices to natural-language instructions when annotations are missing."""
    from huggingface_hub import hf_hub_download

    try:
        tasks_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="meta/tasks.jsonl",
        )
    except Exception:
        return {}

    task_lookup = {}
    with open(tasks_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                task_lookup[int(record["task_index"])] = record["task"]
    return task_lookup
