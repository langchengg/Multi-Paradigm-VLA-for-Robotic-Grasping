import numpy as np

from src.data.common_dataset import VLADataset, collate_vla_batch
from src.data.hdf5_utils import write_episode


def _make_episode(length=5, image_size=16):
    return {
        "images": np.zeros((length, image_size, image_size, 3), dtype=np.uint8),
        "actions": np.zeros((length, 7), dtype=np.float32),
        "reward": np.zeros((length,), dtype=np.float32),
        "robot_state": {
            "eef_pos": np.zeros((length, 3), dtype=np.float32),
            "eef_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (length, 1)),
            "gripper": np.zeros((length, 1), dtype=np.float32),
            "qpos": np.zeros((length, 9), dtype=np.float32),
            "qvel": np.zeros((length, 9), dtype=np.float32),
        },
        "object_state": {
            "target_pos": np.zeros((length, 3), dtype=np.float32),
            "target_quat": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (length, 1)),
        },
        "instruction": "pick up the red cube",
        "success": True,
        "target_name": "red_cube",
        "dataset_name": "synthetic",
    }


def test_vla_dataset_shapes(tmp_path):
    path = tmp_path / "demo.hdf5"
    write_episode(path, 0, _make_episode())
    dataset = VLADataset(path, horizon=4)
    item = dataset[0]
    assert len(dataset) == 5
    assert dataset.action_dim == 7
    assert dataset.robot_state_dim == 26
    assert item["image"].shape == (3, 16, 16)
    assert item["action_chunk"].shape == (4, 7)
    assert item["robot_state"].shape == (26,)
    assert item["instruction"] == "pick up the red cube"


def test_vla_collate_shapes(tmp_path):
    path = tmp_path / "demo.hdf5"
    write_episode(path, 0, _make_episode())
    dataset = VLADataset(path, horizon=4)
    batch = collate_vla_batch([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 16, 16)
    assert batch["action_chunk"].shape == (2, 4, 7)
    assert batch["robot_state"].shape == (2, 26)
    assert batch["instruction"] == ["pick up the red cube", "pick up the red cube"]

