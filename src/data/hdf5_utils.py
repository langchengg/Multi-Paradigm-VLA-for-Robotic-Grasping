from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def list_episodes(h5: h5py.File) -> list[str]:
    return sorted(name for name in h5.keys() if name.startswith("episode_"))


def write_episode(path, episode_index: int, episode: dict) -> None:
    """Append one episode to the benchmark HDF5 schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as h5:
        name = f"episode_{episode_index:06d}"
        if name in h5:
            del h5[name]
        group = h5.create_group(name)

        images = group.create_group("images")
        images.create_dataset(
            "agentview",
            data=np.asarray(episode["images"], dtype=np.uint8),
            compression="gzip",
        )
        group.create_dataset("actions", data=np.asarray(episode["actions"], dtype=np.float32))
        group.create_dataset("reward", data=np.asarray(episode["reward"], dtype=np.float32))
        group.create_dataset("success", data=np.asarray(bool(episode["success"]), dtype=np.bool_))
        group.create_dataset("instruction", data=np.asarray(episode["instruction"], dtype=STRING_DTYPE))

        robot = group.create_group("robot_state")
        for key in ["eef_pos", "eef_quat", "gripper", "qpos", "qvel"]:
            robot.create_dataset(key, data=np.asarray(episode["robot_state"][key], dtype=np.float32))

        obj = group.create_group("object_state")
        for key in ["target_pos", "target_quat"]:
            obj.create_dataset(key, data=np.asarray(episode["object_state"][key], dtype=np.float32))

        group.attrs["dataset_name"] = str(episode.get("dataset_name", "synthetic"))
        group.attrs["target_name"] = str(episode.get("target_name", ""))


def read_episode(h5: h5py.File, episode_name: str) -> dict:
    group = h5[episode_name]
    instruction = group["instruction"][()]
    if isinstance(instruction, bytes):
        instruction = instruction.decode("utf-8")
    return {
        "images": group["images"]["agentview"],
        "actions": group["actions"],
        "reward": group["reward"],
        "success": bool(group["success"][()]),
        "instruction": str(instruction),
        "robot_state": group["robot_state"],
        "object_state": group["object_state"],
        "dataset_name": group.attrs.get("dataset_name", "synthetic"),
    }
