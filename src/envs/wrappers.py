from __future__ import annotations

import numpy as np


def flatten_robot_state(robot_state: dict, object_state: dict | None = None) -> np.ndarray:
    """Flatten structured state into the tensor used by policies.

    Synthetic MuJoCo observations include object state, so the benchmark policy
    conditions on robot proprioception plus target pose. CLIP image/text
    conditioning is still shared by all decoders, but the low-level controller
    gets enough state to learn reliable Cartesian servoing from finite data.
    """
    parts = [
        np.asarray(robot_state["eef_pos"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["eef_quat"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["gripper"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["qpos"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["qvel"], dtype=np.float32).reshape(-1),
    ]
    if object_state is not None:
        target_pos = np.asarray(object_state["target_pos"], dtype=np.float32).reshape(-1)
        eef_pos = np.asarray(robot_state["eef_pos"], dtype=np.float32).reshape(-1)
        parts.extend([
            target_pos,
            np.asarray(object_state["target_quat"], dtype=np.float32).reshape(-1),
            target_pos - eef_pos,
        ])
    return np.concatenate(parts, axis=0).astype(np.float32)
