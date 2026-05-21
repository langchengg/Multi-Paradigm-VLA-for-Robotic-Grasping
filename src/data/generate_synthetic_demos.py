from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.hdf5_utils import write_episode
from src.envs.controllers import ScriptedGraspController
from src.envs.synthetic_franka_grasp_env import SyntheticFrankaGraspEnv
from src.utils.seeding import seed_everything


def _append_state(buffers: dict, obs: dict) -> None:
    buffers["images"].append(obs["image"])
    for key in ["eef_pos", "eef_quat", "gripper", "qpos", "qvel"]:
        buffers["robot_state"][key].append(obs["robot_state"][key])
    for key in ["target_pos", "target_quat"]:
        buffers["object_state"][key].append(obs["object_state"][key])


def generate_synthetic_demos(
    output_path="data/synthetic_demos.hdf5",
    num_episodes: int = 50,
    image_size: int = 128,
    camera_name: str = "frontview",
    max_steps: int = 150,
    seed: int = 0,
    noise_std: float = 0.02,
    require_success: bool = False,
    max_attempts: int | None = None,
) -> dict:
    seed_everything(seed)
    env = SyntheticFrankaGraspEnv(image_size=image_size, camera_name=camera_name)
    env._max_steps = max_steps
    controller = ScriptedGraspController()
    output_path = Path(output_path)
    if output_path.exists():
        output_path.unlink()

    success_count = 0
    try:
        episode_idx = 0
        attempts = 0
        max_attempts = max_attempts or (num_episodes * (10 if require_success else 1))
        while episode_idx < num_episodes and attempts < max_attempts:
            attempts += 1
            obs = env.reset(randomize=True)
            env._max_steps = max_steps
            controller.reset(obs)
            instruction = obs["instruction"]
            target_name = obs["object_state"]["target_name"]
            buffers = {
                "images": [],
                "actions": [],
                "reward": [],
                "robot_state": {key: [] for key in ["eef_pos", "eef_quat", "gripper", "qpos", "qvel"]},
                "object_state": {key: [] for key in ["target_pos", "target_quat"]},
            }
            info = {"success": False}

            for _step in range(max_steps):
                _append_state(buffers, obs)
                action = controller.act(obs)
                if noise_std > 0:
                    action[:6] += np.random.normal(0.0, noise_std, size=6).astype(np.float32)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                buffers["actions"].append(action)
                obs, reward, done, info = env.step(action)
                buffers["reward"].append(float(reward))
                if done:
                    break

            success = bool(info.get("success", False))
            if require_success and not success:
                print(
                    f"[generate] attempt={attempts} rejected "
                    f"steps={len(buffers['actions'])} success=False"
                )
                continue

            success_count += int(success)
            episode = {
                "images": np.asarray(buffers["images"], dtype=np.uint8),
                "actions": np.asarray(buffers["actions"], dtype=np.float32),
                "reward": np.asarray(buffers["reward"], dtype=np.float32),
                "robot_state": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in buffers["robot_state"].items()
                },
                "object_state": {
                    key: np.asarray(values, dtype=np.float32)
                    for key, values in buffers["object_state"].items()
                },
                "instruction": instruction,
                "success": success,
                "target_name": target_name,
                "dataset_name": "synthetic",
            }
            write_episode(output_path, episode_idx, episode)
            print(
                f"[generate] episode={episode_idx + 1}/{num_episodes} attempt={attempts} "
                f"steps={len(buffers['actions'])} success={success}"
            )
            episode_idx += 1
    finally:
        env.close()

    stats = {
        "output_path": str(output_path),
        "num_episodes": episode_idx,
        "attempts": attempts,
        "success_count": success_count,
        "success_rate": success_count / max(episode_idx, 1),
    }
    print(f"[generate] wrote {output_path} success_rate={stats['success_rate']:.3f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HDF5 synthetic Franka grasp demos.")
    parser.add_argument("--output", dest="output_path", default="data/synthetic_demos.hdf5")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--camera_name", default="frontview")
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--require_success", action="store_true")
    parser.add_argument("--max_attempts", type=int, default=None)
    args = parser.parse_args()
    generate_synthetic_demos(**vars(args))


if __name__ == "__main__":
    main()
