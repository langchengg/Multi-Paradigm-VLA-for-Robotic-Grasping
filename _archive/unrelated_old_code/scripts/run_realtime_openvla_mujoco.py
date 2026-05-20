#!/usr/bin/env python3
"""
Run a real OpenVLA policy in the Franka MuJoCo viewer.

This is for visualizing policy behavior, including saved Notebook 2 adapters
while training is producing checkpoints. The MuJoCo viewer shows the simulated
Franka Panda arm; the terminal prints model output, action parsing, latency,
reward, and success state.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.franka_grasp_env import FrankaGraspEnv
from models.openvla_policy import (
    OPENVLA_BASE_MODEL,
    OpenVLAPolicy,
    OpenVLAPolicyConfig,
    find_latest_adapter_dir,
    format_franka_action,
    format_physical_delta,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time MuJoCo rollout with a real OpenVLA adapter policy.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Path to a fine-tuned OpenVLA PEFT adapter, usually openvla-finetuned/final.",
    )
    parser.add_argument(
        "--watch-adapter-root",
        type=Path,
        default=None,
        help=(
            "Notebook 2 output root containing final/, best/, or checkpoint-*. "
            "The latest adapter is reloaded between episodes."
        ),
    )
    parser.add_argument("--base-model", default=OPENVLA_BASE_MODEL)
    parser.add_argument(
        "--local-base-dir",
        type=Path,
        default=None,
        help="Optional local cache/path for the base OpenVLA model.",
    )
    parser.add_argument(
        "--allow-base-model",
        action="store_true",
        help="Run the base model without a fine-tuned adapter. Mostly useful for smoke tests.",
    )
    parser.add_argument("--device", default=None, help="Preferred device, for example cuda or cpu.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantized loading.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--camera", default="frontview", choices=["frontview", "topdown", "sideview"])
    parser.add_argument(
        "--target-object",
        default=None,
        choices=["red_cube", "blue_cube", "green_cube"],
        help="Fixed object target. Defaults to a random target each episode.",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.03,
        help="Additional sleep after each control step so viewer updates are readable.",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Run headless without opening MuJoCo viewer.")
    parser.add_argument(
        "--print-generated",
        action="store_true",
        help="Print raw generated OpenVLA text at every step.",
    )
    return parser


def resolve_adapter(args) -> Path | None:
    if args.adapter_dir is not None:
        adapter_dir = args.adapter_dir.expanduser()
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")
        return adapter_dir
    if args.watch_adapter_root is not None:
        adapter_dir = find_latest_adapter_dir(args.watch_adapter_root)
        if adapter_dir is None:
            raise FileNotFoundError(
                "Could not find a PEFT adapter under "
                f"{args.watch_adapter_root}. Expected final/, best/, or checkpoint-*."
            )
        return adapter_dir
    return None


def maybe_reload_watched_adapter(policy: OpenVLAPolicy, args, current_adapter: Path | None) -> Path | None:
    if args.watch_adapter_root is None:
        return current_adapter
    latest = find_latest_adapter_dir(args.watch_adapter_root)
    if latest is not None and latest != current_adapter:
        print(f"[adapter] Reloading latest adapter: {latest}")
        policy.reload_adapter(latest)
        return latest
    return current_adapter


def run(args) -> int:
    if args.adapter_dir is None and args.watch_adapter_root is None and not args.allow_base_model:
        raise SystemExit(
            "Pass --adapter-dir, --watch-adapter-root, or --allow-base-model. "
            "A fine-tuned adapter is required for this repo's Franka action tokens."
        )

    adapter_dir = resolve_adapter(args)
    print("=" * 78)
    print("Real-time OpenVLA MuJoCo rollout")
    print("=" * 78)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {adapter_dir or '<base model only>'}")
    print(f"Viewer: {'off' if args.no_viewer else 'on'}")
    print(f"Robot: Franka Panda in MuJoCo simulation")

    policy = OpenVLAPolicy(
        OpenVLAPolicyConfig(
            adapter_dir=adapter_dir,
            base_model=args.base_model,
            local_base_dir=args.local_base_dir,
            device=args.device,
            load_in_4bit=not args.no_4bit,
            max_new_tokens=args.max_new_tokens,
            allow_base_model=args.allow_base_model,
        )
    )
    env = FrankaGraspEnv(image_size=args.image_size, camera_name=args.camera)
    if not args.no_viewer:
        env.launch_viewer()

    try:
        for episode in range(args.episodes):
            adapter_dir = maybe_reload_watched_adapter(policy, args, adapter_dir)
            obs = env.reset(target_object=args.target_object)
            print(
                f"\n[episode {episode + 1}/{args.episodes}] "
                f"instruction={obs['instruction']!r} adapter={adapter_dir or '<base>'}"
            )

            for step in range(args.max_steps):
                action, info = policy.predict_action(obs["image"], obs["instruction"])
                obs, reward, done, env_info = env.step(action)

                status = (
                    f"step={step + 1:03d} "
                    f"action={format_franka_action(action)} "
                    f"{format_physical_delta(action)} "
                    f"latency={info.get('inference_time_ms', 0):.1f}ms "
                    f"parse_failed={info.get('parse_failed', False)} "
                    f"reward={reward:+.3f} "
                    f"success={env_info.get('success', False)}"
                )
                print(status)
                if args.print_generated:
                    print(f"  generated={info.get('generated_text', '')!r}")

                if done:
                    break
                if args.step_sleep > 0:
                    time.sleep(args.step_sleep)

            print(
                f"[episode {episode + 1}] done "
                f"success={env_info.get('success', False)} "
                f"steps={env_info.get('step')}"
            )

    finally:
        policy.close()
        env.close()
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
