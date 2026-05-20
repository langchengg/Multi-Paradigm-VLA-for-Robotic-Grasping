from __future__ import annotations

import torch.nn as nn

from src.models.diffusion_head_impl import DiffusionHead


class DiffusionActionDecoder(nn.Module):
    """Continuous action chunk decoder trained with noise-prediction loss."""

    def __init__(
        self,
        condition_dim: int,
        action_dim: int = 7,
        horizon: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.head = DiffusionHead(
            feature_dim=condition_dim,
            action_dim=action_dim,
            action_horizon=horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
        )

    def forward(self, condition, actions):
        return self.head(condition, actions)

    def sample(self, condition, num_steps=None):
        return self.head.sample(condition, num_steps=num_steps)
