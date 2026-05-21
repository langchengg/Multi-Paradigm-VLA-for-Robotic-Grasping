from __future__ import annotations

from pathlib import Path
from typing import Optional
from copy import copy

import torch


def latest_checkpoint(checkpoint_dir) -> Optional[Path]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return None
    latest = root / "latest.pt"
    if latest.exists():
        return latest
    candidates = sorted(root.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def save_checkpoint(path, model, optimizer=None, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=False)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def checkpoint_training_args(path, map_location="cpu") -> dict:
    """Return train-time CLI args stored in a checkpoint, if available."""
    payload = torch.load(path, map_location=map_location)
    args = payload.get("args")
    return dict(args) if isinstance(args, dict) else {}


def apply_checkpoint_config(args, checkpoint_path):
    """Copy an argparse namespace and override model-shape args from a checkpoint."""
    if checkpoint_path is None:
        return args
    ckpt_args = checkpoint_training_args(checkpoint_path, map_location="cpu")
    if not ckpt_args:
        return args
    merged = copy(args)
    for key in (
        "horizon",
        "clip_model_name",
        "pretrained_clip",
        "finetune_clip",
        "local_files_only",
        "tiny_random_clip",
        "hidden_dim",
        "num_layers",
        "diffusion_train_steps",
        "inference_steps",
        "num_action_bins",
    ):
        if key in ckpt_args and hasattr(merged, key):
            setattr(merged, key, ckpt_args[key])
    return merged
