"""
Run Kaggle Pipeline
===================
Single-entry orchestration for the full Kaggle training workflow:
1. Collect MuJoCo Franka demos
2. Fine-tune OpenVLA with mixed MuJoCo + DROID data
3. Train diffusion / flow-matching baselines and run held-out DROID offline evaluation

This keeps everything inside one Kaggle session so intermediate outputs are reused
from /kaggle/working without manually publishing them as Kaggle Datasets.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def build_pipeline_steps(project_root):
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
            "env": {"VLA_DEMO_DIR": str(working_root / "demos")},
            "outputs": [working_root / "openvla-finetuned" / "final"],
        },
        {
            "name": "Notebook 3: Train baselines + offline DROID eval",
            "script": project_root / "notebooks" / "03_flow_matching_eval.py",
            "env": {},
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


def run_step(step, python_executable):
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


def main():
    parser = argparse.ArgumentParser(description="Run the full Kaggle VLA pipeline in one session.")
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated subset of steps to run (1,2,3) or 'all'. Default: all.",
    )
    args = parser.parse_args()

    steps = build_pipeline_steps(PROJECT_ROOT)
    selected_indices = parse_steps_arg(args.steps, len(steps))

    print("=" * 78)
    print("Kaggle End-to-End VLA Pipeline")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Selected steps: {[index + 1 for index in selected_indices]}")
    print("This mode reuses /kaggle/working outputs and avoids manual Dataset publishing.")

    total_start = time.time()
    for index in selected_indices:
        run_step(steps[index], sys.executable)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 78)
    print("Pipeline complete")
    print("=" * 78)
    print(f"Total elapsed: {total_elapsed / 3600:.2f} h")
    print("Expected final artifacts:")
    print("  /kaggle/working/demos")
    print("  /kaggle/working/openvla-finetuned/final")
    print("  /kaggle/working/results/real_offline_summary.json")
    print("  /kaggle/working/results/technical_report.md")


if __name__ == "__main__":
    main()
