"""
Success Rate Heatmap
====================
Visualize grasp success rate as a function of object position on the table.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_success_heatmap(env, model, grid_size=8, trials_per_pos=5,
                             max_steps=100, save_path=None,
                             use_oracle_info=True):
    """
    Evaluate success rate on a grid of object positions.

    Args:
        env: SimpleGraspEnv instance
        model: VLA model with predict_action() method
        grid_size: number of positions per axis
        trials_per_pos: evaluations per position
        max_steps: max steps per episode
        save_path: path to save heatmap image
        use_oracle_info: pass pos info to model

    Returns:
        success_map: (grid_size, grid_size) array of success rates
        x_range: x coordinates
        y_range: y coordinates
    """
    x_range = np.linspace(0.3, 0.7, grid_size)
    y_range = np.linspace(-0.2, 0.2, grid_size)
    success_map = np.zeros((grid_size, grid_size))

    total = grid_size * grid_size
    count = 0

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            successes = 0
            for trial in range(trials_per_pos):
                obs = env.reset(target_object="red_cube", randomize=False)

                # Override object position
                adr = env._obj_qpos_adr["red_cube"]
                env.data.qpos[adr] = x
                env.data.qpos[adr + 1] = y
                env.data.qpos[adr + 2] = 0.24

                # Re-step to stabilize
                import mujoco
                for _ in range(50):
                    mujoco.mj_step(env.model, env.data)

                obs = env._get_obs()

                for step in range(max_steps):
                    kwargs = {}
                    if use_oracle_info:
                        kwargs["gripper_pos"] = obs.get("gripper_pos")
                        kwargs["target_pos"] = obs.get("target_pos")

                    action, _ = model.predict_action(
                        obs["image"],
                        obs["instruction"],
                        **kwargs,
                    )
                    obs, reward, done, info = env.step(action)
                    if done:
                        break

                if info.get("success", False):
                    successes += 1

            success_map[j, i] = successes / trials_per_pos
            count += 1

            if count % max(1, total // 5) == 0:
                print(f"  Heatmap progress: {count}/{total} positions")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        success_map,
        extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
        origin='lower',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        aspect='auto',
        interpolation='bilinear',
    )

    cbar = plt.colorbar(im, ax=ax, label='Success Rate', shrink=0.85)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Grasp Success Rate by Object Position', fontsize=14,
                 fontweight='bold')

    # Add grid
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)
    ax.grid(True, alpha=0.2, color='white')

    # Annotate values
    for i in range(grid_size):
        for j in range(grid_size):
            val = success_map[j, i]
            color = 'white' if val < 0.5 else 'black'
            ax.text(x_range[i], y_range[j], f'{val:.0%}',
                    ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold')

    # Draw table boundary
    table_rect = plt.Rectangle(
        (0.2, -0.35), 0.6, 0.7,
        linewidth=2, edgecolor='brown', facecolor='none',
        linestyle='--', alpha=0.5, label='Table boundary'
    )
    ax.add_patch(table_rect)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap: {save_path}")

    plt.close()
    return success_map, x_range, y_range


# ─── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from envs.simple_grasp_env import SimpleGraspEnv
    from models.dummy_vla import DummyVLA

    print("=" * 60)
    print("Generating Success Heatmap")
    print("=" * 60)

    env = SimpleGraspEnv(image_size=64, camera_name="frontview")  # small for speed
    model = DummyVLA("flow_matching")

    success_map, x_range, y_range = generate_success_heatmap(
        env, model,
        grid_size=5,         # 5x5 for quick test
        trials_per_pos=3,
        max_steps=80,
        save_path=str(project_root / "assets" / "success_heatmap.png"),
    )

    env.close()
    print(f"\nMean success rate: {np.mean(success_map):.1%}")
    print("✅ Heatmap generation test passed!")
