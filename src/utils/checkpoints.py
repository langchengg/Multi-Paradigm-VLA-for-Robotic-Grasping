from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def latest_checkpoint(checkpoint_dir) -> Optional[Path]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return None
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

