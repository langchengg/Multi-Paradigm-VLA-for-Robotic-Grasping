"""
3D Trajectory Visualization
============================
Plot end-effector trajectories in 3D space for visual analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_trajectories_3d(trajectories_dict, target_pos=None, save_path=None,
                         title="End-Effector Trajectories"):
    """
    Plot 3D trajectories from multiple models.

    Args:
        trajectories_dict: {"model_name": [(N,3) positions], ...}
        target_pos: (3,) target object position to mark
        save_path: if provided, save figure to this path
        title: plot title
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.Set2(np.linspace(0, 1, len(trajectories_dict)))

    for (name, positions), color in zip(trajectories_dict.items(), colors):
        pos = np.array(positions)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                label=name, linewidth=2, color=color, alpha=0.8)
        # Mark start
        ax.scatter(*pos[0], color=color, s=80, marker='o', edgecolors='black',
                   linewidths=0.5, zorder=5)
        # Mark end
        ax.scatter(*pos[-1], color=color, s=80, marker='s', edgecolors='black',
                   linewidths=0.5, zorder=5)

    if target_pos is not None:
        ax.scatter(*target_pos, color='red', s=250, marker='*',
                   edgecolors='darkred', linewidths=1, label='Target', zorder=10)

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')

    # Better angle
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot: {save_path}")

    plt.close()
    return fig


def plot_trajectories_2d(trajectories_dict, target_pos=None, save_path=None):
    """
    Plot 2D projections (top-down XY and side XZ views).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(trajectories_dict)))

    for (name, positions), color in zip(trajectories_dict.items(), colors):
        pos = np.array(positions)
        # Top-down (XY)
        axes[0].plot(pos[:, 0], pos[:, 1], label=name, linewidth=2,
                     color=color, alpha=0.8)
        axes[0].scatter(pos[0, 0], pos[0, 1], color=color, s=60,
                        marker='o', edgecolors='black', zorder=5)
        # Side (XZ)
        axes[1].plot(pos[:, 0], pos[:, 2], label=name, linewidth=2,
                     color=color, alpha=0.8)
        axes[1].scatter(pos[0, 0], pos[0, 2], color=color, s=60,
                        marker='o', edgecolors='black', zorder=5)

    if target_pos is not None:
        axes[0].scatter(target_pos[0], target_pos[1], color='red', s=200,
                        marker='*', edgecolors='darkred', zorder=10, label='Target')
        axes[1].scatter(target_pos[0], target_pos[2], color='red', s=200,
                        marker='*', edgecolors='darkred', zorder=10, label='Target')

    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Top-Down View (XY)', fontweight='bold')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('Side View (XZ)', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Draw table surface line
    axes[1].axhline(y=0.22, color='brown', linestyle='--', alpha=0.5,
                    label='Table surface')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 2D trajectory plot: {save_path}")

    plt.close()
    return fig


# ─── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing trajectory visualization...")

    # Dummy data
    t = np.linspace(0, 2*np.pi, 50)
    trajectories = {
        "Autoregressive": np.column_stack([
            0.5 + 0.1*np.cos(t), 0.1*np.sin(t), 0.5 - 0.3*t/(2*np.pi)
        ]),
        "Flow-Matching": np.column_stack([
            0.5 + 0.08*np.cos(t*1.2), 0.08*np.sin(t*1.2), 0.5 - 0.3*t/(2*np.pi)
        ]),
    }

    plot_trajectories_3d(
        trajectories,
        target_pos=np.array([0.45, 0.1, 0.24]),
        save_path="/tmp/test_trajectories_3d.png",
    )

    plot_trajectories_2d(
        trajectories,
        target_pos=np.array([0.45, 0.1, 0.24]),
        save_path="/tmp/test_trajectories_2d.png",
    )

    print("✅ Trajectory visualization test passed!")
