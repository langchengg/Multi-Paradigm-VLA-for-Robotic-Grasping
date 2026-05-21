from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import h5py
import numpy as np

from src.data.common_dataset import VLADataset, collate_vla_batch
from src.data.libero_adapter import LiberoVLADataset
from src.models.policy import VLAPolicy
from src.training.losses import scalar_loss_dict
from src.utils.checkpoints import save_checkpoint
from src.utils.device import get_device, move_batch_to_device
from src.utils.seeding import seed_everything


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def build_dataset(args):
    if args.dataset == "synthetic":
        return VLADataset(args.data_path, horizon=args.horizon, dataset_name="synthetic")
    if args.dataset == "libero":
        return LiberoVLADataset(args.data_path, horizon=args.horizon, suite=args.libero_suite)
    raise ValueError(f"Unknown dataset: {args.dataset}")


def compute_action_stats(dataset) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Compute per-dimension action stats for continuous generative decoders."""
    if not isinstance(dataset, VLADataset):
        return None, None
    actions = []
    with h5py.File(dataset.hdf5_path, "r") as h5:
        for episode in dataset._episode_lengths:
            actions.append(np.asarray(h5[episode]["actions"], dtype=np.float32))
    if not actions:
        return None, None
    stacked = np.concatenate(actions, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def make_summary_writer(args):
    if args.disable_tensorboard:
        return NullSummaryWriter()
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(args.results_dir) / "tensorboard" / args.dataset / args.decoder
    return SummaryWriter(log_dir=str(log_dir))


def train(args) -> dict:
    seed_everything(args.seed)
    device = get_device(args.device)
    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_vla_batch,
    )
    if len(loader) == 0:
        raise RuntimeError("Dataset is empty; generate demonstrations first.")

    policy = VLAPolicy(
        decoder_type=args.decoder,
        robot_state_dim=dataset.robot_state_dim,
        action_dim=dataset.action_dim,
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
        num_action_bins=args.num_action_bins,
    ).to(device)

    action_mean, action_std = compute_action_stats(dataset)
    decoder_head = getattr(policy.decoder, "head", None)
    if decoder_head is not None and hasattr(decoder_head, "set_action_stats") and action_mean is not None:
        decoder_head.set_action_stats(action_mean, action_std)
        print(
            "[train] action stats mean="
            f"{np.round(action_mean, 4).tolist()} std={np.round(action_std, 4).tolist()}"
        )

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    checkpoint_dir = Path(args.checkpoint_dir) / args.dataset / args.decoder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = make_summary_writer(args)

    global_step = 0
    last_loss = None
    for epoch in range(args.num_epochs):
        policy.train()
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and global_step >= args.max_batches:
                break
            batch = move_batch_to_device(batch, device)
            loss, info = policy.training_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            last_loss = float(loss.detach().cpu())
            writer.add_scalar("train/loss", last_loss, global_step)
            for key, value in scalar_loss_dict(info).items():
                writer.add_scalar(f"train/{key}", value, global_step)
            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch + 1}/{args.num_epochs} "
                    f"step={global_step} loss={last_loss:.5f} device={device}"
                )
            global_step += 1
        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
        save_checkpoint(
            ckpt_path,
            policy,
            optimizer,
            extra={
                "args": vars(args),
                "policy_config": {
                    "decoder_type": args.decoder,
                    "robot_state_dim": dataset.robot_state_dim,
                    "action_dim": dataset.action_dim,
                    "horizon": args.horizon,
                    "clip_model_name": args.clip_model_name,
                    "pretrained_clip": args.pretrained_clip,
                    "freeze_clip": not args.finetune_clip,
                    "finetune_clip": args.finetune_clip,
                    "tiny_random_clip": args.tiny_random_clip,
                    "decoder_hidden_dim": args.hidden_dim,
                    "decoder_num_layers": args.num_layers,
                    "diffusion_train_steps": args.diffusion_train_steps,
                    "inference_steps": args.inference_steps,
                    "num_action_bins": args.num_action_bins,
                },
            },
        )
        if args.max_batches is not None and global_step >= args.max_batches:
            break
    writer.close()

    latest = checkpoint_dir / "latest.pt"
    save_checkpoint(
        latest,
        policy,
        optimizer,
        extra={"args": vars(args), "global_step": global_step},
    )
    summary = {
        "checkpoint": str(latest),
        "global_step": global_step,
        "last_loss": last_loss,
        "device": str(device),
        "trainable_parameters": sum(p.numel() for p in trainable_params),
        "action_mean": action_mean.tolist() if action_mean is not None else None,
        "action_std": action_std.tolist() if action_std is not None else None,
    }
    (checkpoint_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[train] saved {latest}")
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train CLIP-conditioned VLA action decoder heads.")
    parser.add_argument("--dataset", choices=["synthetic", "libero"], default="synthetic")
    parser.add_argument("--data_path", default="data/synthetic_demos.hdf5")
    parser.add_argument("--libero_suite", default="libero_object")
    parser.add_argument("--decoder", choices=["autoregressive", "diffusion", "flow_matching"], required=True)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--disable_tensorboard", action="store_true")
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
    parser.add_argument("--num_action_bins", type=int, default=256)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
