from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.envs.synthetic_franka_grasp_env import SyntheticFrankaGraspEnv
from src.envs.wrappers import flatten_robot_state
from src.models.policy import VLAPolicy
from src.utils.checkpoints import apply_checkpoint_config, latest_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.visualization.viewer import try_launch_viewer


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def make_policy(args, obs, device):
    robot_state_dim = flatten_robot_state(obs["robot_state"], obs.get("object_state")).shape[0]
    return VLAPolicy(
        decoder_type=args.decoder,
        robot_state_dim=robot_state_dim,
        action_dim=7,
        horizon=args.horizon,
        clip_model_name=args.clip_model_name,
        pretrained_clip=args.pretrained_clip,
        freeze_clip=not args.finetune_clip,
        finetune_clip=args.finetune_clip,
        local_files_only=args.local_files_only,
        tiny_random_clip=args.tiny_random_clip,
        decoder_hidden_dim=args.hidden_dim,
        decoder_num_layers=args.num_layers,
        diffusion_train_steps=args.diffusion_train_steps,
        inference_steps=args.inference_steps,
    ).to(device)


def live_watch(args) -> None:
    device = get_device(args.device)
    env = SyntheticFrankaGraspEnv(image_size=args.image_size, camera_name=args.camera_name)
    env._max_steps = args.max_steps
    try:
        if args.viewer and not try_launch_viewer(env):
            print("[live_watch] Continuing headless; frames are still rendered through MuJoCo offscreen rendering.")

        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            alt = Path("checkpoints") / args.dataset / args.decoder
            if alt.exists():
                checkpoint_dir = alt
        obs = env.reset(randomize=True)
        env._max_steps = args.max_steps
        ckpt = latest_checkpoint(checkpoint_dir)
        policy_args = apply_checkpoint_config(args, ckpt)
        policy = make_policy(policy_args, obs, device)
        last_checkpoint = None
        last_reload_time = 0.0
        print(f"[live_watch] watching checkpoints in {checkpoint_dir}")

        for episode in range(args.num_episodes):
            obs = env.reset(randomize=True)
            env._max_steps = args.max_steps
            total_return = 0.0
            step = 0
            while step < args.max_steps:
                ckpt = latest_checkpoint(checkpoint_dir)
                now = time.time()
                if ckpt is not None and ckpt != last_checkpoint and (
                    last_checkpoint is None or now - last_reload_time >= args.reload_interval
                ):
                    try:
                        if ckpt != last_checkpoint:
                            policy_args = apply_checkpoint_config(args, ckpt)
                            policy = make_policy(policy_args, obs, device)
                        load_checkpoint(ckpt, policy, map_location=device)
                        policy.eval()
                        last_checkpoint = ckpt
                        last_reload_time = now
                        print(f"[live_watch] loaded {ckpt}")
                    except Exception as exc:
                        print(f"[live_watch] could not load checkpoint {ckpt}: {exc}")
                with torch.no_grad():
                    chunk = policy.predict_action_chunk_from_obs(obs, device)
                chunk_np = chunk[0].detach().cpu().numpy()
                for chunk_step in range(min(args.exec_chunk_steps, chunk_np.shape[0])):
                    action = np.clip(chunk_np[chunk_step], -1.0, 1.0)
                    obs, reward, done, info = env.step(action)
                    total_return += reward
                    step += 1
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    if done or step >= args.max_steps:
                        break
                if done:
                    break
            print(
                f"[live_watch] episode={episode + 1}/{args.num_episodes} "
                f"return={total_return:.2f} success={bool(info.get('success', False))}"
            )
    finally:
        env.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Live MuJoCo watch mode for trained VLA decoders.")
    parser.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    parser.add_argument("--decoder", choices=["autoregressive", "diffusion", "flow_matching"], default="diffusion")
    parser.add_argument("--checkpoint_dir", default="checkpoints/synthetic/diffusion")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--reload_interval", type=float, default=5.0)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--camera_name", default="frontview")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--clip_model_name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--pretrained_clip", type=str_to_bool, default=True)
    parser.add_argument("--no_pretrained_clip", dest="pretrained_clip", action="store_false")
    parser.add_argument("--finetune_clip", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--tiny_random_clip", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--diffusion_train_steps", type=int, default=100)
    parser.add_argument("--inference_steps", type=int, default=10)
    parser.add_argument("--exec_chunk_steps", type=int, default=1)
    return parser


def main() -> None:
    live_watch(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
