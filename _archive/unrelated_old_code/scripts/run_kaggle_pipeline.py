"""
Run Kaggle Pipeline
===================
Single-entry orchestration for the full Kaggle training workflow:
1. Collect MuJoCo Franka demos
2. Fine-tune OpenVLA with mixed MuJoCo + DROID data
3. Train diffusion / flow-matching baselines and run held-out DROID offline evaluation

This keeps everything inside one Kaggle session so intermediate outputs are reused
from /kaggle/working without manually publishing them as Kaggle Datasets.
Pass --droid-only to skip Step 1 and train/evaluate from DROID only.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_size_bytes(path):
    """Estimate the total on-disk size for a file or directory."""
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        return path.stat().st_size

    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _remove_path(path):
    """Remove a file or directory if it exists."""
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    shutil.rmtree(path, ignore_errors=True)


def cleanup_after_openvla_step(working_root=Path("/kaggle/working"), home_dir=Path.home()):
    """
    Reclaim disk after Notebook 2 while preserving everything Notebook 3 still needs.

    We intentionally keep:
    - /kaggle/working/openvla-finetuned/final
    - Hugging Face model/module caches used to reload OpenVLA in Notebook 3
    """
    openvla_dir = working_root / "openvla-finetuned"
    targets = [
        working_root / "demos",
        working_root / "expert_demo.gif",
        home_dir / ".cache" / "pip",
        home_dir / ".cache" / "huggingface" / "datasets",
    ]
    targets.extend(sorted(openvla_dir.glob("checkpoint-*")))

    removed = []
    reclaimed_bytes = 0
    for path in targets:
        if not path.exists():
            continue
        size_bytes = _path_size_bytes(path)
        _remove_path(path)
        reclaimed_bytes += size_bytes
        removed.append((path, size_bytes))

    print("\n" + "-" * 78)
    print("Post-Notebook 2 Cleanup")
    print("-" * 78)
    if not removed:
        print("No disposable caches or intermediates were found.")
        return

    for path, size_bytes in removed:
        print(f"Removed: {path} ({size_bytes / 1024 / 1024:.1f} MB)")
    print(f"Reclaimed approximately {reclaimed_bytes / 1024 / 1024 / 1024:.2f} GB")


def build_pipeline_steps(project_root, droid_only=False):
    """Return the ordered Kaggle workflow with the env overrides each step needs."""
    working_root = Path("/kaggle/working")
    return [
        {
            "name": "Notebook 1: Collect MuJoCo Franka demos",
            "script": project_root / "notebooks" / "01_env_setup_and_demo.py",
            "env": {},
            "outputs": [working_root / "demos"],
        },
        {
            "name": "Notebook 2: Fine-tune OpenVLA",
            "script": project_root / "notebooks" / "02_openvla_qlora_finetune.py",
            "env": (
                {"VLA_USE_MUJOCO_DEMOS": "0"}
                if droid_only
                else {"VLA_DEMO_DIR": str(working_root / "demos")}
            ),
            "outputs": [working_root / "openvla-finetuned" / "final"],
            "post_run": cleanup_after_openvla_step,
        },
        {
            "name": "Notebook 3: Train baselines + offline DROID eval",
            "script": project_root / "notebooks" / "03_flow_matching_eval.py",
            "env": {"VLA_SKIP_INSTALL": "1"},
            "outputs": [working_root / "results" / "real_offline_summary.json"],
        },
    ]


def parse_steps_arg(raw_value, total_steps):
    """Parse a comma-separated subset like '1,3' into zero-based step indices."""
    if raw_value.strip().lower() == "all":
        return list(range(total_steps))

    indices = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        step_num = int(chunk)
        if not 1 <= step_num <= total_steps:
            raise ValueError(f"Unsupported step '{step_num}'. Expected 1..{total_steps}.")
        indices.append(step_num - 1)
    if not indices:
        raise ValueError("No valid steps were selected.")
    return indices


def run_step(step, python_executable, keep_intermediates=False):
    """Execute one notebook script with the requested environment overrides."""
    env = os.environ.copy()
    env.update(step["env"])
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print("\n" + "=" * 78)
    print(step["name"])
    print("=" * 78)
    print(f"Script: {step['script']}")
    if step["env"]:
        for key, value in sorted(step["env"].items()):
            print(f"{key}={value}")

    start = time.time()
    subprocess.check_call([python_executable, str(step["script"])], cwd=PROJECT_ROOT, env=env)
    elapsed = time.time() - start

    print(f"✓ Completed in {elapsed / 60:.1f} min")
    for path in step["outputs"]:
        print(f"  Output check: {path} -> {'found' if path.exists() else 'missing'}")

    if not keep_intermediates and step.get("post_run") is not None:
        step["post_run"]()


def main():
    parser = argparse.ArgumentParser(description="Run the full Kaggle VLA pipeline in one session.")
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated subset of steps to run (1,2,3) or 'all'. Default: all.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Disable the low-disk cleanup that normally runs after Notebook 2.",
    )
    parser.add_argument(
        "--droid-only",
        action="store_true",
        help="Skip MuJoCo demo collection and run Notebook 2 in pure-DROID mode.",
    )
    args = parser.parse_args()

    steps = build_pipeline_steps(PROJECT_ROOT, droid_only=args.droid_only)
    selected_indices = parse_steps_arg(args.steps, len(steps))
    if args.droid_only:
        if args.steps.strip().lower() == "all":
            selected_indices = [1, 2]
        elif 0 in selected_indices:
            raise ValueError(
                "--droid-only skips Notebook 1. Use '--steps 2,3' or omit --steps."
            )

    print("=" * 78)
    print("Kaggle End-to-End VLA Pipeline")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Selected steps: {[index + 1 for index in selected_indices]}")
    print("This mode reuses /kaggle/working outputs and avoids manual Dataset publishing.")
    print(f"DROID-only mode: {'on' if args.droid_only else 'off'}")
    print(f"Low-disk cleanup after Notebook 2: {'off' if args.keep_intermediates else 'on'}")

    total_start = time.time()
    for index in selected_indices:
        run_step(steps[index], sys.executable, keep_intermediates=args.keep_intermediates)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 78)
    print("Pipeline complete")
    print("=" * 78)
    print(f"Total elapsed: {total_elapsed / 3600:.2f} h")
    print("Expected final artifacts:")
    print("  /kaggle/working/openvla-finetuned/final")
    print("  /kaggle/working/results/real_offline_summary.json")
    print("  /kaggle/working/results/technical_report.md")
    if args.keep_intermediates:
        print("  /kaggle/working/demos")


if __name__ == "__main__":
    main()
