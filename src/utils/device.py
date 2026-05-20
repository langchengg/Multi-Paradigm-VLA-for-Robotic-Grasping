from __future__ import annotations

import torch


def get_device(requested: str = "auto") -> torch.device:
    """Resolve the training device without assuming CUDA is present."""
    requested = (requested or "auto").lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a batch dict while leaving strings/lists untouched."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved

