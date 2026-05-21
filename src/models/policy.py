from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.envs.wrappers import flatten_robot_state
from src.models.autoregressive_decoder import AutoregressiveActionDecoder
from src.models.diffusion_decoder import DiffusionActionDecoder
from src.models.encoders import CLIPConditioningEncoder
from src.models.flow_matching_decoder import FlowMatchingActionDecoder


DECODER_TYPES = ("autoregressive", "diffusion", "flow_matching")


class VLAPolicy(nn.Module):
    """CLIP-conditioned VLA action policy with pluggable decoder heads."""

    def __init__(
        self,
        decoder_type: str,
        robot_state_dim: int,
        action_dim: int = 7,
        horizon: int = 16,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        pretrained_clip: bool = True,
        freeze_clip: bool = True,
        finetune_clip: bool = False,
        local_files_only: bool = False,
        tiny_random_clip: bool = False,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 3,
        diffusion_train_steps: int = 100,
        inference_steps: int = 10,
        num_action_bins: int = 256,
    ):
        super().__init__()
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"decoder_type must be one of {DECODER_TYPES}, got {decoder_type!r}")
        self.decoder_type = decoder_type
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.encoder = CLIPConditioningEncoder(
            robot_state_dim=robot_state_dim,
            clip_model_name=clip_model_name,
            pretrained_clip=pretrained_clip,
            freeze_clip=freeze_clip,
            finetune_clip=finetune_clip,
            local_files_only=local_files_only,
            tiny_random_clip=tiny_random_clip,
        )
        condition_dim = self.encoder.condition_dim
        if decoder_type == "autoregressive":
            self.decoder = AutoregressiveActionDecoder(
                condition_dim=condition_dim,
                action_dim=action_dim,
                horizon=horizon,
                num_bins=num_action_bins,
                hidden_dim=decoder_hidden_dim,
                num_layers=max(1, decoder_num_layers),
            )
        elif decoder_type == "diffusion":
            self.decoder = DiffusionActionDecoder(
                condition_dim=condition_dim,
                action_dim=action_dim,
                horizon=horizon,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_num_layers,
                num_train_timesteps=diffusion_train_steps,
                num_inference_steps=inference_steps,
            )
        else:
            self.decoder = FlowMatchingActionDecoder(
                condition_dim=condition_dim,
                action_dim=action_dim,
                horizon=horizon,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_num_layers,
                num_inference_steps=inference_steps,
            )

    @property
    def condition_dim(self) -> int:
        return self.encoder.condition_dim

    def encode_condition(self, batch: dict) -> torch.Tensor:
        return self.encoder(batch["image"], batch["instruction"], batch["robot_state"])

    def training_loss(self, batch: dict):
        condition = self.encode_condition(batch)
        return self.decoder(condition, batch["action_chunk"])

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, num_steps: int | None = None) -> torch.Tensor:
        condition = self.encode_condition(batch)
        return self.decoder.sample(condition, num_steps=num_steps)

    @torch.no_grad()
    def predict_action_chunk_from_obs(self, obs: dict, device: torch.device) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(obs["image"], dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        robot_state = torch.from_numpy(
            flatten_robot_state(obs["robot_state"], obs.get("object_state"))
        ).unsqueeze(0)
        batch = {
            "image": image.to(device),
            "instruction": [obs["instruction"]],
            "robot_state": robot_state.to(device),
        }
        return self.predict_action_chunk(batch)
