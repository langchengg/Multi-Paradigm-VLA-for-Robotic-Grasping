from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.envs.controllers import ScriptedGraspController
from src.envs.synthetic_franka_grasp_env import SyntheticFrankaGraspEnv
from src.models.policy import VLAPolicy
from src.training.metrics import (
    action_mse,
    classify_failure,
    gripper_timing_error,
    summarize_rollouts,
    trajectory_smoothness,
)
from src.utils.checkpoints import latest_checkpoint, load_checkpoint
from src.utils.device import get_device
from src.utils.logging import write_json
from src.utils.seeding import seed_everything
from src.visualization.record_video import save_video


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _policy_from_obs(args, decoder: str, obs: dict, device: torch.device) -> VLAPolicy:
    from src.envs.wrappers import flatten_robot_state

    robot_state_dim = flatten_robot_state(obs["robot_state"]).shape[0]
    return VLAPolicy(
        decoder_type=decoder,
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
        inference_steps=args.inference_steps,
    ).to(device)


def evaluate_decoder(args, decoder: str) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    env = SyntheticFrankaGraspEnv(image_size=args.image_size, camera_name=args.camera_name)
    controller = ScriptedGraspController()
    rollouts = []
    saved_videos = []
    try:
        obs = env.reset(randomize=True)
        policy = _policy_from_obs(args, decoder, obs, device)
        ckpt_dir = Path(args.checkpoint_dir) / args.dataset / decoder
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            load_checkpoint(ckpt, policy, map_location=device)
            print(f"[eval] loaded {ckpt}")
        else:
            print(f"[eval] no checkpoint found in {ckpt_dir}; evaluating random initialized head")
        policy.eval()

        for episode in range(args.num_episodes):
            obs = env.reset(randomize=True)
            controller.reset(obs)
            pred_actions = []
            expert_actions = []
            rewards = []
            xy_errors = []
            latencies = []
            info = {"success": False}
            episode_frames = []
            for _step in range(args.max_steps):
                expert = controller.act(obs)
                start = time.perf_counter()
                chunk = policy.predict_action_chunk_from_obs(obs, device)
                latency_ms = (time.perf_counter() - start) * 1000.0
                action = chunk[0, 0].detach().cpu().numpy().astype(np.float32)
                action = np.clip(action, -1.0, 1.0)
                pred_actions.append(action)
                expert_actions.append(expert)
                latencies.append(latency_ms)
                xy_errors.append(float(np.linalg.norm(obs["robot_state"]["eef_pos"][:2] - obs["object_state"]["target_pos"][:2])))
                if args.save_video:
                    episode_frames.append(obs["image"])
                obs, reward, done, info = env.step(action)
                rewards.append(float(reward))
                if done:
                    break
            final_height = float(obs["object_state"]["target_pos"][2])
            success = bool(info.get("success", False))
            failure = classify_failure(success, final_height, float(np.mean(xy_errors) if xy_errors else 1.0))
            rollout = {
                "decoder": decoder,
                "episode": episode,
                "success": success,
                "return": float(np.sum(rewards)),
                "final_object_lift_height": final_height,
                "action_mse": action_mse(pred_actions, expert_actions),
                "trajectory_smoothness": trajectory_smoothness(pred_actions),
                "inference_latency_ms": float(np.mean(latencies) if latencies else 0.0),
                "number_of_inference_steps": 1 if decoder == "autoregressive" else args.inference_steps,
                "gripper_timing_error": gripper_timing_error(pred_actions, expert_actions),
                "failure_type": failure,
            }
            rollouts.append(rollout)
            if episode_frames:
                video_path = Path(args.results_dir) / args.dataset / decoder / "videos" / f"episode_{episode:03d}.gif"
                save_video(episode_frames, video_path, fps=10)
                saved_videos.append(str(video_path))
            print(f"[eval] decoder={decoder} episode={episode + 1}/{args.num_episodes} success={success}")
    finally:
        env.close()

    summary = summarize_rollouts(rollouts)
    summary["decoder"] = decoder
    summary["videos"] = saved_videos
    out_dir = Path(args.results_dir) / args.dataset / decoder
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", summary)
    with (out_dir / "rollouts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollouts[0].keys()) if rollouts else ["decoder"])
        writer.writeheader()
        writer.writerows(rollouts)
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-conditioned VLA action decoders.")
    parser.add_argument("--dataset", choices=["synthetic"], default="synthetic")
    parser.add_argument("--decoder", choices=["autoregressive", "diffusion", "flow_matching"], default="diffusion")
    parser.add_argument("--all_decoders", action="store_true")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--no_save_video", dest="save_video", action="store_false")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--inference_steps", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    decoders = ["autoregressive", "diffusion", "flow_matching"] if args.all_decoders else [args.decoder]
    summaries = [evaluate_decoder(args, decoder) for decoder in decoders]
    comparison_dir = Path(args.results_dir) / args.dataset
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "comparison_metrics.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
