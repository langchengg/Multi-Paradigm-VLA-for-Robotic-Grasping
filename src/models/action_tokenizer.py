from __future__ import annotations

import torch


class ActionTokenizer:
    """Uniform scalar discretizer for normalized continuous robot actions."""

    def __init__(self, action_dim: int = 7, num_bins: int = 256, min_value: float = -1.0, max_value: float = 1.0):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")
        self.action_dim = int(action_dim)
        self.num_bins = int(num_bins)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.bos_token_id = self.num_bins

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions [..., action_dim] into integer tokens with same prefix shape."""
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, found {actions.shape[-1]}")
        clipped = actions.clamp(self.min_value, self.max_value)
        scaled = (clipped - self.min_value) / (self.max_value - self.min_value)
        return torch.round(scaled * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode integer tokens back to normalized continuous actions."""
        tokens = tokens.long().clamp(0, self.num_bins - 1)
        scaled = tokens.float() / (self.num_bins - 1)
        return scaled * (self.max_value - self.min_value) + self.min_value

    def flatten(self, token_grid: torch.Tensor) -> torch.Tensor:
        return token_grid.reshape(token_grid.shape[0], -1)

    def unflatten(self, token_sequence: torch.Tensor, horizon: int) -> torch.Tensor:
        return token_sequence.reshape(token_sequence.shape[0], horizon, self.action_dim)

