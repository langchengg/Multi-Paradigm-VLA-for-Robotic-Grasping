from __future__ import annotations

import numpy as np


def ensure_7d_action(action, source_name: str = "action") -> np.ndarray:
    """Return this benchmark's normalized 7D Franka delta-pose action."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 7:
        return np.clip(action, -1.0, 1.0)
    if action.shape[0] == 4:
        return np.array(
            [action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]],
            dtype=np.float32,
        )
    raise ValueError(f"{source_name} must have 7 values, got shape {action.shape}")


def scripted_grasp_policy(obs: dict, phase: int, phase_step: int, target_pos) -> tuple[np.ndarray, int, int]:
    """Scripted expert for collecting synthetic Franka grasp demonstrations."""
    gripper_pos = np.asarray(obs["robot_state"]["eef_pos"], dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float32)
    action = np.zeros(7, dtype=np.float32)

    if phase == 0:
        goal = target_pos.copy()
        goal[2] += 0.08
        direction = goal - gripper_pos
        action[:3] = direction * np.array([7.5, 7.5, 6.0], dtype=np.float32)
        action[6] = -1.0
        if np.linalg.norm(direction) < 0.015 or phase_step >= 45:
            return np.clip(action, -1.0, 1.0), 1, 0
        return np.clip(action, -1.0, 1.0), 0, phase_step + 1

    if phase == 1:
        goal = target_pos.copy()
        goal[2] -= 0.055
        direction = goal - gripper_pos
        xy_error = np.linalg.norm(direction[:2])
        action[:3] = direction * np.array([4.5, 4.5, 4.0], dtype=np.float32)
        action[6] = -1.0
        at_grasp_depth = gripper_pos[2] <= target_pos[2] + 0.015
        if (at_grasp_depth and xy_error < 0.055) or phase_step >= 100:
            return np.clip(action, -1.0, 1.0), 2, 0
        return np.clip(action, -1.0, 1.0), 1, phase_step + 1

    if phase == 2:
        goal = target_pos.copy()
        goal[2] -= 0.055
        direction = goal - gripper_pos
        action[:3] = direction * np.array([1.5, 1.5, 1.0], dtype=np.float32)
        action[6] = 1.0
        gripper_width = float(np.asarray(obs["robot_state"].get("gripper", [0.08]))[0])
        if (phase_step >= 12 and gripper_width < 0.045) or phase_step >= 45:
            return np.clip(action, -1.0, 1.0), 3, 0
        return np.clip(action, -1.0, 1.0), 2, phase_step + 1

    if phase == 3:
        goal = target_pos.copy()
        goal[2] = 0.34
        direction = goal - gripper_pos
        action[:3] = direction * np.array([1.2, 1.2, 1.5], dtype=np.float32)
        action[6] = 1.0
        if phase_step >= 60 or gripper_pos[2] > 0.32:
            return np.clip(action, -1.0, 1.0), 4, 0
        return np.clip(action, -1.0, 1.0), 3, phase_step + 1

    goal = target_pos.copy()
    goal[2] = 0.56
    direction = goal - gripper_pos
    action[:3] = direction * np.array([1.5, 1.5, 2.5], dtype=np.float32)
    action[6] = 1.0
    return np.clip(action, -1.0, 1.0), 4, phase_step + 1


class ScriptedGraspController:
    """Stateful scripted expert used by dataset generation and action-MSE evaluation."""

    def __init__(self):
        self.phase = 0
        self.phase_step = 0
        self.target_pos = None

    def reset(self, obs: dict) -> None:
        self.phase = 0
        self.phase_step = 0
        self.target_pos = np.asarray(obs["object_state"]["target_pos"], dtype=np.float32).copy()

    def act(self, obs: dict) -> np.ndarray:
        if self.target_pos is None:
            self.reset(obs)
        action, self.phase, self.phase_step = scripted_grasp_policy(
            obs,
            self.phase,
            self.phase_step,
            self.target_pos,
        )
        return action
