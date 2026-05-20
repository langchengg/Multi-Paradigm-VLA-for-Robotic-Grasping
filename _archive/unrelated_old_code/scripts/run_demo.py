"""
Run Demo — Full Pipeline Test
==============================
One-click script that tests the entire VLA + MuJoCo pipeline:
1. Environment setup & rendering (Franka Panda 7-DOF)
2. Expert demo collection
3. Closed-loop VLA evaluation (3 decoders: autoregressive vs diffusion vs flow-matching)
4. Visualization assets (GIF, trajectory, heatmap)

Usage:
    python scripts/run_demo.py                # Full test with Franka Panda
    python scripts/run_demo.py --quick         # Quick test (~15 seconds)
    python scripts/run_demo.py --simple        # Use simple 3-DOF env (faster)
"""

import os
import sys
import time
import argparse
import zipfile
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def package_assets_dir(assets_dir, archive_path):
    """Package the contents of assets_dir into a zip archive."""
    assets_dir = Path(assets_dir)
    archive_path = Path(archive_path)

    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory does not exist: {assets_dir}")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(assets_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(assets_dir))

    return archive_path


def main(quick=False, use_simple=False, use_viewer=False):
    total_start = time.time()

    # ─── Config ───
    image_size = 128 if quick else 256
    num_demos = 10 if quick else 50
    num_eval_episodes = 5 if quick else 20
    heatmap_grid = 3 if quick else 6
    heatmap_trials = 2 if quick else 5
    assets_dir = PROJECT_ROOT / "assets"
    demos_dir = PROJECT_ROOT / "data" / "demos"

    env_type = "Simple 3-DOF" if use_simple else "Franka Panda 7-DOF"
    print("=" * 70)
    print("  VLA Action Decoder Benchmark — Demo Pipeline")
    print("  Autoregressive vs Diffusion vs Flow-Matching")
    print("=" * 70)
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print(f"  Robot: {env_type}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Num demos: {num_demos}")
    print(f"  Eval episodes: {num_eval_episodes}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Environment Test
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("STEP 1: Environment Setup & Rendering Test")
    print("─" * 50)

    if use_simple:
        from envs.simple_grasp_env import SimpleGraspEnv
        env = SimpleGraspEnv(image_size=image_size, camera_name="frontview")
    else:
        from envs.franka_grasp_env import FrankaGraspEnv
        env = FrankaGraspEnv(image_size=image_size, camera_name="frontview")

    if use_viewer:
        env.launch_viewer()

    obs = env.reset(target_object="red_cube")

    print(f"  ✓ Image: {obs['image'].shape}, dtype={obs['image'].dtype}")
    print(f"  ✓ Instruction: '{obs['instruction']}'")
    print(f"  ✓ Gripper pos: {obs['gripper_pos']}")
    print(f"  ✓ Target pos: {obs['target_pos']}")

    # Save a single frame
    from PIL import Image
    frame_path = assets_dir / "initial_frame.png"
    os.makedirs(assets_dir, exist_ok=True)
    Image.fromarray(obs["image"]).save(str(frame_path))
    print(f"  ✓ Saved initial frame: {frame_path}")

    # Multi-camera test
    for cam in ["frontview", "topdown", "sideview"]:
        frame = env.render_frame(camera_name=cam)
        save_p = assets_dir / f"view_{cam}.png"
        Image.fromarray(frame).save(str(save_p))
    print(f"  ✓ Saved multi-view renders")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Expert Demo Collection
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print(f"STEP 2: Collecting {num_demos} Expert Demos")
    print("─" * 50)

    from data.collect_demos import collect_demos

    demos, stats = collect_demos(
        env,
        num_demos=num_demos,
        save_dir=str(demos_dir),
        add_noise=False,
        noise_std=0.0,
    )

    # Save best demo as GIF
    success_demos = [d for d in demos if d["success"]]
    if success_demos:
        best = success_demos[0]
        # Subsample frames for GIF
        step = max(1, len(best["images"]) // 30)
        frames = best["images"][::step]
        env.render_video(frames, str(assets_dir / "expert_demo.gif"), fps=8)
        print(f"  ✓ Saved expert demo GIF ({len(frames)} frames)")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Closed-Loop Evaluation (3 Decoders)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print(f"STEP 3: Closed-Loop Evaluation ({num_eval_episodes} episodes × 3 decoders)")
    print("─" * 50)

    from models.dummy_vla import DummyVLA
    from evaluation.closed_loop_eval import VLAMuJoCoEvaluator

    models = {
        "autoregressive": DummyVLA("autoregressive"),
        "diffusion": DummyVLA("diffusion"),
        "flow_matching": DummyVLA("flow_matching"),
    }

    evaluator = VLAMuJoCoEvaluator(
        model=models["flow_matching"],
        env=env,
        use_oracle_info=True,
    )

    comparison = evaluator.evaluate_comparison(
        models,
        num_episodes=num_eval_episodes,
        max_steps=100,
        record_video=True,
        save_dir=str(assets_dir),
        visualize=use_viewer,
    )

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Trajectory Visualization
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print("STEP 4: Generating Trajectory Visualizations")
    print("─" * 50)

    from visualization.plot_trajectories import plot_trajectories_3d, plot_trajectories_2d

    # Collect one trajectory per model for visualization
    traj_dict = {}
    for name, result in comparison.items():
        if result["trajectories"]:
            traj = result["trajectories"][0]
            traj_dict[name] = traj["gripper_positions"]

    if traj_dict:
        # Get target position from the first trajectory
        target = comparison[list(comparison.keys())[0]]["trajectories"][0].get("target_object", "red_cube")
        target_pos = env._get_object_pos(target)

        plot_trajectories_3d(
            traj_dict,
            target_pos=target_pos,
            save_path=str(assets_dir / "trajectories_3d.png"),
        )

        plot_trajectories_2d(
            traj_dict,
            target_pos=target_pos,
            save_path=str(assets_dir / "trajectories_2d.png"),
        )

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Success Heatmap
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 50)
    print(f"STEP 5: Generating Success Heatmap ({heatmap_grid}x{heatmap_grid} grid)")
    print("─" * 50)

    from visualization.success_heatmap import generate_success_heatmap

    # Use smaller image for heatmap speed
    if use_simple:
        heatmap_env = SimpleGraspEnv(image_size=64, camera_name="frontview")
    else:
        heatmap_env = FrankaGraspEnv(image_size=64, camera_name="frontview")

    success_map, x_range, y_range = generate_success_heatmap(
        heatmap_env,
        DummyVLA("flow_matching"),
        grid_size=heatmap_grid,
        trials_per_pos=heatmap_trials,
        max_steps=80,
        save_path=str(assets_dir / "success_heatmap.png"),
    )

    heatmap_env.close()

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  Generated assets in: {assets_dir}")

    # List generated files
    if assets_dir.exists():
        files = sorted(assets_dir.rglob("*"))
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                print(f"    📄 {f.relative_to(PROJECT_ROOT)} ({size/1024:.1f} KB)")

    print(f"\n  Demo data in: {demos_dir}")
    if demos_dir.exists():
        npz_files = list(demos_dir.glob("*.npz"))
        print(f"    📦 {len(npz_files)} .npz files")

    print(f"\n  📊 Evaluation Summary:")
    for name, result in comparison.items():
        s = result["summary"]
        print(f"    {name}: success={s['success_rate']:.0%}, "
              f"latency={s['p50_inference_ms']:.1f}ms")

    archive_name = "assets_quick.zip" if quick else "assets_full.zip"
    if use_simple:
        archive_name = archive_name.replace("assets_", "assets_simple_")
    archive_path = package_assets_dir(assets_dir, PROJECT_ROOT / archive_name)
    archive_size_kb = archive_path.stat().st_size / 1024
    print(f"\n  📦 Packaged assets: {archive_path} ({archive_size_kb:.1f} KB)")

    env.close()
    print(f"\n✅ All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Action Decoder Benchmark — Demo Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with fewer episodes (~15 seconds)")
    parser.add_argument("--simple", action="store_true",
                        help="Use simple 3-DOF env instead of Franka Panda")
    parser.add_argument("--viewer", action="store_true",
                        help="Open interactive MuJoCo viewer window (macOS/desktop)")
    args = parser.parse_args()

    main(quick=args.quick, use_simple=args.simple, use_viewer=args.viewer)
