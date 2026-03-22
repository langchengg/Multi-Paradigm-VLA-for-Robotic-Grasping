#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Kaggle Notebook 1: MuJoCo Environment Setup + Demo Collection
═══════════════════════════════════════════════════════════════════

Run this on Kaggle (CPU or GPU instance) to:
1. Install MuJoCo + dependencies
2. Set up off-screen rendering (osmesa for Linux)
3. Create custom grasping environment
4. Collect 100 expert demonstrations
5. Save demos as Kaggle Dataset for training

Output: data/demos/*.npz (upload as Kaggle Dataset)

⏱️ Estimated time: 5-10 minutes
💾 Output size: ~200 MB (100 demos with 256×256 images)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import subprocess
import sys
from pathlib import Path

def install_packages():
    """Install MuJoCo and dependencies for Kaggle."""
    packages = [
        "mujoco>=3.0.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "imageio>=2.30.0",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Install native OpenGL backends for headless rendering on Linux (Kaggle)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "libgl1-mesa-glx libgl1-mesa-dev libegl1-mesa-dev "
        "libosmesa6-dev libglew-dev patchelf",
        shell=True, capture_output=True
    )
    print("✅ Packages installed")

install_packages()

# ═══════════════════════════════════════════════════════════════
# Cell 2: Configure Rendering
# ═══════════════════════════════════════════════════════════════

import os

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs._rendering import configure_headless_rendering

backend = configure_headless_rendering()

import mujoco
import numpy as np
from PIL import Image

print(f"✅ MuJoCo {mujoco.__version__} loaded")
print(f"   Rendering backend: {backend or os.environ.get('MUJOCO_GL', 'default')}")

# Interactive viewer: set ENABLE_VIEWER=1 to open a 3D window (macOS/desktop only)
ENABLE_VIEWER = os.environ.get("ENABLE_VIEWER", "").strip() == "1"

# ═══════════════════════════════════════════════════════════════
# Cell 3: Load Shared Franka Panda Environment
# ═══════════════════════════════════════════════════════════════

from envs.franka_grasp_env import FrankaGraspEnv
from data.collect_demos import collect_demos

print("✅ FrankaGraspEnv ready")

# Test environment
env = FrankaGraspEnv(image_size=256, camera_name="frontview")
if ENABLE_VIEWER:
    env.launch_viewer()
    print("   Viewer enabled. A 3D window should open if on macOS/desktop.")
obs = env.reset(target_object="red_cube")
print(f"   Image: {obs['image'].shape}, Instruction: '{obs['instruction']}'")
print("   Action format: 7-DOF [dx, dy, dz, dax, day, daz, gripper]")
env.close()

# ═══════════════════════════════════════════════════════════════
# Cell 4: Collect 7-DOF Franka Demonstrations
# ═══════════════════════════════════════════════════════════════

NUM_DEMOS = 100
IMAGE_SIZE = 256
SAVE_DIR = "/kaggle/working/demos"
os.makedirs(SAVE_DIR, exist_ok=True)

env = FrankaGraspEnv(image_size=IMAGE_SIZE, camera_name="frontview")
if ENABLE_VIEWER:
    env.launch_viewer()
demos, stats = collect_demos(
    env,
    num_demos=NUM_DEMOS,
    save_dir=SAVE_DIR,
    add_noise=False,
    noise_std=0.0,
    verbose=True,
)
env.close()
print(f"\n✅ Collected {stats['total']} Franka demos → {SAVE_DIR}")
print(f"   Success rate: {stats['success_rate']:.0%}")
print("   Action format: 7-DOF [dx, dy, dz, dax, day, daz, gripper]")
print(f"   Upload this folder as a Kaggle Dataset for Notebook 2!")

# ═══════════════════════════════════════════════════════════════
# Cell 5: Generate Sample GIF
# ═══════════════════════════════════════════════════════════════

successful_demo = next((demo for demo in demos if demo["success"]), None)
if successful_demo is not None:
    frames = successful_demo["images"][::3]  # subsample
    imgs = [Image.fromarray(f) for f in frames]
    gif_path = "/kaggle/working/expert_demo.gif"
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    print(f"✅ Saved sample GIF: {gif_path} ({len(frames)} frames)")
else:
    print("⚠️ No successful Franka demo found for GIF export.")

print("\n" + "="*60)
print("📋 Next: Run Notebook 2 to fine-tune OpenVLA on 7-DOF Franka data")
print("="*60)
