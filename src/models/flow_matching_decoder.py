from __future__ import annotations

import torch.nn as nn

from src.models.flow_matching_head_impl import FlowMatchingHead


class FlowMatchingActionDecoder(nn.Module):
    """Continuous action chunk decoder trained as a conditional velocity field."""

    def __init__(
        self,
        condition_dim: int,
        action_dim: int = 7,
        horizon: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.head = FlowMatchingHead(
            feature_dim=condition_dim,
            action_dim=action_dim,
            action_horizon=horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_inference_steps=num_inference_steps,
        )

    def forward(self, condition, actions):
        return self.head(condition, actions)

    def sample(self, condition, num_steps=None):
        return self.head.sample(condition, num_steps=num_steps)
