from __future__ import annotations

import numpy as np


def flatten_robot_state(robot_state: dict) -> np.ndarray:
    """Flatten the structured robot state into the tensor used by policies."""
    parts = [
        np.asarray(robot_state["eef_pos"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["eef_quat"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["gripper"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["qpos"], dtype=np.float32).reshape(-1),
        np.asarray(robot_state["qvel"], dtype=np.float32).reshape(-1),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)

