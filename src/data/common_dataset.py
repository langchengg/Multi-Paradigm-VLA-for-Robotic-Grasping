from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.hdf5_utils import list_episodes


def _decode_instruction(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class VLADataset(Dataset):
    """Unified dataset API for synthetic and optional LIBERO action chunks."""

    def __init__(self, hdf5_path, horizon: int = 16, dataset_name: str = "synthetic"):
        self.hdf5_path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.hdf5_path}. Generate synthetic demos first."
            )
        self.horizon = int(horizon)
        self.dataset_name = dataset_name
        self._index = []
        self._episode_lengths = {}
        self._action_dim = None
        self._robot_state_dim = None

        with h5py.File(self.hdf5_path, "r") as h5:
            for ep in list_episodes(h5):
                length = int(h5[ep]["actions"].shape[0])
                self._episode_lengths[ep] = length
                for t in range(length):
                    self._index.append((ep, t))
            if self._index:
                first_ep = self._index[0][0]
                self._action_dim = int(h5[first_ep]["actions"].shape[-1])
                self._robot_state_dim = int(
                    self._flatten_robot_state(
                        h5[first_ep]["robot_state"],
                        0,
                        h5[first_ep].get("object_state"),
                    ).shape[0]
                )

    @property
    def action_dim(self) -> int:
        return int(self._action_dim or 0)

    @property
    def robot_state_dim(self) -> int:
        return int(self._robot_state_dim or 0)

    def __len__(self):
        return len(self._index)

    def _flatten_robot_state(self, robot_group, t: int, object_group=None) -> np.ndarray:
        parts = [
            np.asarray(robot_group["eef_pos"][t], dtype=np.float32).reshape(-1),
            np.asarray(robot_group["eef_quat"][t], dtype=np.float32).reshape(-1),
            np.asarray(robot_group["gripper"][t], dtype=np.float32).reshape(-1),
            np.asarray(robot_group["qpos"][t], dtype=np.float32).reshape(-1),
            np.asarray(robot_group["qvel"][t], dtype=np.float32).reshape(-1),
        ]
        if object_group is not None:
            target_pos = np.asarray(object_group["target_pos"][t], dtype=np.float32).reshape(-1)
            eef_pos = np.asarray(robot_group["eef_pos"][t], dtype=np.float32).reshape(-1)
            parts.extend([
                target_pos,
                np.asarray(object_group["target_quat"][t], dtype=np.float32).reshape(-1),
                target_pos - eef_pos,
            ])
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _action_chunk(self, actions: np.ndarray, t: int) -> np.ndarray:
        chunk = actions[t:t + self.horizon]
        if chunk.shape[0] < self.horizon:
            pad = np.repeat(chunk[-1:],
                            self.horizon - chunk.shape[0],
                            axis=0) if len(chunk) else np.zeros((self.horizon, actions.shape[-1]))
            chunk = np.concatenate([chunk, pad], axis=0)
        return chunk.astype(np.float32)

    def __getitem__(self, idx):
        ep, t = self._index[idx]
        with h5py.File(self.hdf5_path, "r") as h5:
            group = h5[ep]
            image = np.asarray(group["images"]["agentview"][t], dtype=np.float32)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected image [H,W,3], found {image.shape}")
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

            robot_state = torch.from_numpy(
                self._flatten_robot_state(group["robot_state"], t, group.get("object_state"))
            )
            actions = np.asarray(group["actions"], dtype=np.float32)
            action_chunk = torch.from_numpy(self._action_chunk(actions, t))

            instruction = _decode_instruction(group["instruction"][()])
            success = bool(group["success"][()])
            dataset_name = group.attrs.get("dataset_name", self.dataset_name)

        return {
            "image": image,
            "instruction": instruction,
            "robot_state": robot_state,
            "action_chunk": action_chunk,
            "success": success,
            "dataset_name": dataset_name,
        }


def collate_vla_batch(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "instruction": [item["instruction"] for item in batch],
        "robot_state": torch.stack([item["robot_state"] for item in batch], dim=0),
        "action_chunk": torch.stack([item["action_chunk"] for item in batch], dim=0),
        "success": torch.tensor([item["success"] for item in batch], dtype=torch.bool),
        "dataset_name": [item["dataset_name"] for item in batch],
    }
