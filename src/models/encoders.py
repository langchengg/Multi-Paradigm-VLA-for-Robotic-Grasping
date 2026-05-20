from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class CLIPConditioningEncoder(nn.Module):
    """Shared CLIP vision-language encoder used by all action decoders.

    Default benchmark mode freezes CLIP and exposes fixed image/text features
    plus raw robot state. This keeps Mac training lightweight and makes decoder
    comparisons fair. Set ``finetune_clip=True`` to train CLIP as well.
    """

    def __init__(
        self,
        robot_state_dim: int,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        pretrained_clip: bool = True,
        freeze_clip: bool = True,
        finetune_clip: bool = False,
        local_files_only: bool = False,
        tiny_random_clip: bool = False,
    ):
        super().__init__()
        self.robot_state_dim = int(robot_state_dim)
        self.clip_model_name = clip_model_name
        self.pretrained_clip = bool(pretrained_clip)
        self.freeze_clip = bool(freeze_clip) and not bool(finetune_clip)
        self.finetune_clip = bool(finetune_clip)
        self.local_files_only = bool(local_files_only)
        self.tiny_random_clip = bool(tiny_random_clip)

        try:
            from transformers import CLIPConfig, CLIPModel
        except Exception as exc:
            raise ImportError(
                "CLIPConditioningEncoder requires transformers. "
                "Install dependencies with pip install -r requirements.txt."
            ) from exc

        self._uses_hf_tokenizer = False
        self.tokenizer = None
        if self.pretrained_clip:
            try:
                from transformers import CLIPTokenizerFast

                self.clip = CLIPModel.from_pretrained(
                    clip_model_name,
                    local_files_only=local_files_only,
                )
                self.tokenizer = CLIPTokenizerFast.from_pretrained(
                    clip_model_name,
                    local_files_only=local_files_only,
                )
                self._uses_hf_tokenizer = True
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load pretrained CLIP model {clip_model_name!r}. "
                    "For offline smoke tests pass --no_pretrained_clip; for the "
                    "benchmark install/download the CLIP weights first."
                ) from exc
        else:
            if tiny_random_clip:
                config = CLIPConfig(
                    projection_dim=64,
                    vision_config={
                        "hidden_size": 64,
                        "intermediate_size": 128,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "image_size": 224,
                        "patch_size": 32,
                    },
                    text_config={
                        "vocab_size": 2048,
                        "hidden_size": 64,
                        "intermediate_size": 128,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "max_position_embeddings": 32,
                    },
                )
            else:
                config = CLIPConfig()
            self.clip = CLIPModel(config)

        self.projection_dim = int(self.clip.config.projection_dim)
        self.max_text_length = int(self.clip.config.text_config.max_position_embeddings)
        self.vocab_size = int(self.clip.config.text_config.vocab_size)
        self.condition_dim = self.projection_dim * 2 + self.robot_state_dim

        if self.freeze_clip:
            self.clip.requires_grad_(False)
            self.clip.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_clip:
            self.clip.eval()
        return self

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images [B,3,H,W] or [B,H,W,3], found {images.shape}")
        if images.shape[1] != 3 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        images = images.float()
        if images.max() > 2:
            images = images / 255.0
        images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = CLIP_MEAN.to(device=images.device, dtype=images.dtype)
        std = CLIP_STD.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def _hash_tokenize(self, instructions: Iterable[str], device: torch.device) -> dict[str, torch.Tensor]:
        rows = []
        for text in instructions:
            words = str(text).lower().replace(",", " ").split()
            ids = [self._hash_word(word) for word in words[: self.max_text_length - 2]]
            ids = [0] + ids + [1]
            ids = ids[: self.max_text_length]
            padding = [0] * (self.max_text_length - len(ids))
            rows.append(ids + padding)
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        attention_mask = (input_ids != 0).long()
        attention_mask[:, 0] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _hash_word(self, word: str) -> int:
        digest = hashlib.sha1(word.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "little")
        return 2 + (value % max(self.vocab_size - 2, 1))

    def _tokenize(self, instructions: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        if self._uses_hf_tokenizer:
            tokens = self.tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            return {key: value.to(device) for key, value in tokens.items()}
        return self._hash_tokenize(instructions, device)

    def forward(self, images: torch.Tensor, instructions: list[str], robot_state: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_images(images)
        tokens = self._tokenize(instructions, pixel_values.device)
        robot_state = robot_state.to(device=pixel_values.device, dtype=pixel_values.dtype)

        grad_enabled = self.finetune_clip and not self.freeze_clip
        with torch.set_grad_enabled(grad_enabled):
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            text_features = self.clip.get_text_features(**tokens)
        if not grad_enabled:
            image_features = image_features.detach()
            text_features = text_features.detach()

        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        return torch.cat([image_features, text_features, robot_state.float()], dim=-1)
