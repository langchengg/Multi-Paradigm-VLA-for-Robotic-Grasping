"""
Expert Demo Collection
======================
Collect expert demonstrations using a scripted policy.
Saves trajectories as .npz files for VLA training.
"""

import os
import sys
import numpy as np
from pathlib import Path


def scripted_grasp_policy(obs, phase, phase_step, target_pos):
    """
    Simple scripted grasping policy with 4 phases:
    1. Approach: Move above the target object
    2. Descend: Lower to grasp position
    3. Grasp: Close fingers
    4. Lift: Raise the grasped object

    Args:
        obs: current observation dict
        phase: current phase (0-3)
        phase_step: steps within current phase
        target_pos: target object position

    Returns:
        action: (4,) action [dx, dy, dz, gripper_cmd]
        new_phase: updated phase
        new_phase_step: updated step count
    """
    gripper_pos = obs["gripper_pos"]

    if phase == 0:
        # Phase 0: Move above target
        goal = target_pos.copy()
        goal[2] += 0.15  # 15cm above
        direction = goal - gripper_pos
        action = np.zeros(4)
        action[:3] = direction * 5.0  # P-gain
        action[3] = -1.0  # open gripper

        if np.linalg.norm(direction) < 0.02 or phase_step > 30:
            return action, 1, 0
        return action, 0, phase_step + 1

    elif phase == 1:
        # Phase 1: Descend to object
        goal = target_pos.copy()
        goal[2] += 0.02  # 2cm above (contact height)
        direction = goal - gripper_pos
        action = np.zeros(4)
        action[:3] = direction * 4.0
        action[3] = -1.0  # open gripper

        if np.linalg.norm(direction) < 0.015 or phase_step > 25:
            return action, 2, 0
        return action, 1, phase_step + 1

    elif phase == 2:
        # Phase 2: Close gripper
        action = np.zeros(4)
        action[3] = 1.0  # close gripper

        if phase_step > 10:
            return action, 3, 0
        return action, 2, phase_step + 1

    else:
        # Phase 3: Lift
        goal = gripper_pos.copy()
        goal[2] = 0.6  # lift target height
        direction = goal - gripper_pos
        action = np.zeros(4)
        action[:3] = direction * 3.0
        action[3] = 1.0  # keep gripper closed

        return action, 3, phase_step + 1


def collect_demos(env, num_demos=50, save_dir="data/demos", add_noise=True,
                  noise_std=0.03, verbose=True):
    """
    Collect expert demonstrations.

    Args:
        env: SimpleGraspEnv or RobosuiteVLAWrapper instance
        num_demos: number of demonstrations to collect
        save_dir: directory to save .npz files
        add_noise: whether to add Gaussian noise to actions
        noise_std: noise standard deviation
        verbose: print progress

    Returns:
        demos: list of trajectory dicts
        stats: collection statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    demos = []
    success_count = 0

    target_objects = getattr(env, "OBJECTS", ["red_cube", "blue_cube", "green_cube"])

    for i in range(num_demos):
        target = np.random.choice(target_objects)
        obs = env.reset(target_object=target)

        trajectory = {
            "images": [],
            "instructions": [],
            "actions": [],
            "rewards": [],
            "gripper_positions": [],
            "target_positions": [],
        }

        phase = 0
        phase_step = 0
        target_pos = obs["target_pos"].copy()

        for step in range(150):
            # Record observation
            trajectory["images"].append(obs["image"])
            trajectory["instructions"].append(obs["instruction"])
            trajectory["gripper_positions"].append(obs["gripper_pos"].copy())
            trajectory["target_positions"].append(obs["target_pos"].copy())

            # Get expert action
            action, phase, phase_step = scripted_grasp_policy(
                obs, phase, phase_step, target_pos
            )

            # Add noise for data augmentation
            if add_noise:
                action[:3] += np.random.normal(0, noise_std, 3)

            action = np.clip(action, -1.0, 1.0)
            trajectory["actions"].append(action)

            # Step environment
            obs, reward, done, info = env.step(action)
            trajectory["rewards"].append(reward)

            if done:
                break

        success = info.get("success", False)
        trajectory["success"] = success
        trajectory["target_object"] = target
        trajectory["num_steps"] = step + 1

        if success:
            success_count += 1

        # Convert lists to numpy arrays
        for key in ["images", "actions", "rewards", "gripper_positions", "target_positions"]:
            trajectory[key] = np.array(trajectory[key])

        demos.append(trajectory)

        if verbose and (i + 1) % 10 == 0:
            rate = success_count / (i + 1)
            print(f"  Collected {i+1}/{num_demos} demos | "
                  f"Success rate: {rate:.0%} ({success_count}/{i+1})")

    # Save all demos
    save_path = os.path.join(save_dir, "expert_demos.npz")
    np.savez_compressed(
        save_path,
        # Save as object array to handle variable-length trajectories
        num_demos=len(demos),
        success_flags=np.array([d["success"] for d in demos]),
        target_objects=np.array([d["target_object"] for d in demos]),
        num_steps=np.array([d["num_steps"] for d in demos]),
    )

    # Also save individual trajectories for flexibility
    for idx, demo in enumerate(demos):
        traj_path = os.path.join(save_dir, f"demo_{idx:04d}.npz")
        np.savez_compressed(
            traj_path,
            images=demo["images"],
            actions=demo["actions"],
            rewards=demo["rewards"],
            gripper_positions=demo["gripper_positions"],
            target_positions=demo["target_positions"],
            instructions=demo["instructions"],
            success=demo["success"],
            target_object=demo["target_object"],
        )

    stats = {
        "total": num_demos,
        "success": success_count,
        "success_rate": success_count / num_demos if num_demos > 0 else 0,
        "save_dir": save_dir,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Collection complete!")
        print(f"  Total demos: {stats['total']}")
        print(f"  Successes: {stats['success']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Saved to: {save_dir}")
        print(f"{'='*50}")

    return demos, stats


# ─── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from envs.simple_grasp_env import SimpleGraspEnv

    print("=" * 60)
    print("Collecting Expert Demonstrations")
    print("=" * 60)

    env = SimpleGraspEnv(image_size=128, camera_name="frontview")

    demos, stats = collect_demos(
        env,
        num_demos=20,  # small number for quick test
        save_dir=str(project_root / "data" / "demos"),
        add_noise=True,
        noise_std=0.03,
    )

    env.close()
    print(f"\n✅ Demo collection test complete!")
    print(f"   Success rate: {stats['success_rate']:.1%}")
