"""
Flow-Matching Action Decoder (π0-inspired)
===========================================
Implements conditional flow matching for VLA action generation.

Instead of autoregressive token generation (OpenVLA default) or
iterative denoising (Diffusion Policy), flow matching learns an
ODE flow from noise → action using a small MLP.

Key advantages:
- 4× faster inference than autoregressive (few ODE steps vs 7 tokens)
- Smooth continuous actions (no discretization artifacts)
- Supports action chunking (predict H future actions at once)

Inspired by:
- π0 (Physical Intelligence): Uses flow matching for 50Hz action generation
- Flow Matching for Generative Modeling (Lipman et al., 2023)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FlowMatchingHead(nn.Module):
    """
    Flow-matching action decoder head.

    Takes VLA backbone features and generates actions via conditional flow matching.

    Training:
        1. Sample t ~ Beta(α, β) (shifted beta distribution, per π0)
        2. Interpolate: x_t = (1-t) * noise + t * action_gt
        3. Predict velocity: v_θ(x_t, t, features)
        4. Loss = ||v_θ - (action_gt - noise)||²

    Inference:
        1. Start from noise x_0 ~ N(0, I)
        2. ODE integration: x_{t+dt} = x_t + v_θ(x_t, t, features) * dt
        3. After K steps: x_1 ≈ predicted_action
    """

    def __init__(
        self,
        feature_dim=4096,       # VLA backbone output dim (e.g., LLM hidden_size)
        action_dim=7,           # 7-DOF: dx,dy,dz,dax,day,daz,gripper
        action_horizon=4,       # Action chunking: predict H future actions
        hidden_dim=512,
        num_layers=4,
        dropout=0.1,
        beta_alpha=1.5,         # Shifted beta distribution α (π0 uses ~1.5)
        beta_beta=1.0,          # Shifted beta distribution β
        num_inference_steps=10, # ODE steps at inference
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.total_action_dim = action_dim * action_horizon
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.num_inference_steps = num_inference_steps

        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Time embedding (sinusoidal)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Velocity prediction network
        # v_θ(x_t, t, features) : (noisy_action, time_embed, features) → velocity
        layers = []
        input_dim = self.total_action_dim + hidden_dim + hidden_dim  # action + time + features
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else self.total_action_dim
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, out_dim),
                *([] if i == num_layers - 1 else [nn.SiLU(), nn.LayerNorm(out_dim), nn.Dropout(dropout)]),
            ])
        self.velocity_net = nn.Sequential(*layers)

        # Action normalization stats (filled during training)
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_action_stats(self, mean, std):
        """Set action normalization statistics from training data."""
        self.action_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.action_std.copy_(torch.tensor(std, dtype=torch.float32).clamp(min=1e-6))

    def normalize_action(self, action):
        """Normalize action to zero mean, unit variance."""
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action):
        """Denormalize action back to original scale."""
        return action * self.action_std + self.action_mean

    def forward(self, features, action_gt):
        """
        Training forward pass: compute flow matching loss.

        Args:
            features: (B, feature_dim) VLA backbone features
            action_gt: (B, H, action_dim) ground truth action sequences

        Returns:
            loss: scalar flow matching loss
            info: dict with training metrics
        """
        B = features.shape[0]
        device = features.device

        # Flatten action horizon: (B, H*action_dim)
        action_flat = action_gt.reshape(B, -1)

        # Normalize actions
        action_norm = self.normalize_action(action_flat.reshape(B, self.action_horizon, self.action_dim))
        action_norm = action_norm.reshape(B, -1)

        # Sample time t ~ Beta(α, β)
        t = torch.distributions.Beta(self.beta_alpha, self.beta_beta).sample((B,)).to(device)
        t = t.unsqueeze(-1)  # (B, 1)

        # Sample noise
        noise = torch.randn_like(action_norm)

        # Interpolate: x_t = (1-t) * noise + t * action_gt
        x_t = (1 - t) * noise + t * action_norm

        # Target velocity: v* = action_gt - noise (optimal transport direction)
        target_velocity = action_norm - noise

        # Predict velocity
        proj_features = self.feature_proj(features)        # (B, hidden_dim)
        time_embed = self.time_embed(t.squeeze(-1))        # (B, hidden_dim)
        pred_velocity = self.velocity_net(
            torch.cat([x_t, time_embed, proj_features], dim=-1)
        )

        # Flow matching loss
        loss = F.mse_loss(pred_velocity, target_velocity)

        info = {
            'flow_loss': loss.item(),
            'velocity_norm': pred_velocity.norm(dim=-1).mean().item(),
        }

        return loss, info

    @torch.no_grad()
    def sample(self, features, num_steps=None):
        """
        Inference: generate actions via Euler ODE integration.

        Args:
            features: (B, feature_dim) VLA backbone features

        Returns:
            actions: (B, H, action_dim) predicted action sequence
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        B = features.shape[0]
        device = features.device

        # Project features once
        proj_features = self.feature_proj(features)

        # Start from noise
        x = torch.randn(B, self.total_action_dim, device=device)

        # Euler integration: x_{t+dt} = x_t + v(x_t, t) * dt
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            time_embed = self.time_embed(t)

            velocity = self.velocity_net(
                torch.cat([x, time_embed, proj_features], dim=-1)
            )
            x = x + velocity * dt

        # Denormalize and reshape
        actions = self.denormalize_action(
            x.reshape(B, self.action_horizon, self.action_dim)
        )

        return actions


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time embedding (used in diffusion/flow models)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class FlowMatchingVLA(nn.Module):
    """
    Complete VLA with flow-matching action decoder.

    Architecture:
        Image → Vision Encoder → [features]
        Text  → Text Encoder  → [features]  → Fusion → FlowMatchingHead → Actions
        
    For Kaggle T4: Uses a lightweight backbone (ViT-B/16 + BERT-small)
    to demonstrate the architecture. In production, replace with
    OpenVLA's Prismatic backbone.
    """

    def __init__(
        self,
        vision_model_name="google/vit-base-patch16-224",
        text_model_name="prajjwal1/bert-small",
        action_dim=7,
        action_horizon=4,
        flow_hidden_dim=512,
        flow_num_layers=4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Vision encoder
        from transformers import BertModel, BertTokenizer, ViTModel
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        vision_dim = self.vision_encoder.config.hidden_size  # 768

        # prajjwal1/bert-small is a BERT checkpoint but its config can fail AutoModel
        # resolution on newer transformers, so load the concrete BERT classes directly.
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size  # 512

        # Fusion: project both modalities to shared space
        fusion_dim = 512
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # Cross-attention fusion
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True,
        )

        # Flow-matching action head
        self.flow_head = FlowMatchingHead(
            feature_dim=fusion_dim,
            action_dim=action_dim,
            action_horizon=action_horizon,
            hidden_dim=flow_hidden_dim,
            num_layers=flow_num_layers,
        )

    def encode(self, images, instructions):
        """
        Encode image + text into fused features.

        Args:
            images: (B, C, H, W) or preprocessed pixel_values
            instructions: list of str

        Returns:
            features: (B, fusion_dim)
        """
        # Vision encoding
        vision_out = self.vision_encoder(pixel_values=images)
        vision_cls = vision_out.last_hidden_state[:, 0]  # CLS token
        vision_feat = self.vision_proj(vision_cls)         # (B, fusion_dim)

        # Text encoding
        text_inputs = self.tokenizer(
            instructions, return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(images.device)
        text_out = self.text_encoder(**text_inputs)
        text_cls = text_out.last_hidden_state[:, 0]        # CLS token
        text_feat = self.text_proj(text_cls)                # (B, fusion_dim)

        # Simple fusion: concatenate + transformer
        combined = torch.stack([vision_feat, text_feat], dim=1)  # (B, 2, dim)
        fused = self.fusion_layer(combined)                       # (B, 2, dim)
        features = fused.mean(dim=1)                              # (B, dim) — pool

        return features

    def forward(self, images, instructions, actions_gt):
        """
        Training forward: image + text + gt_actions → loss.

        Args:
            images: (B, C, H, W)
            instructions: list of str
            actions_gt: (B, H, action_dim)

        Returns:
            loss, info
        """
        features = self.encode(images, instructions)
        return self.flow_head(features, actions_gt)

    @torch.no_grad()
    def predict(self, images, instructions, num_steps=10):
        """
        Inference: image + text → predicted actions.

        Returns:
            actions: (B, H, action_dim)
        """
        features = self.encode(images, instructions)
        return self.flow_head.sample(features, num_steps=num_steps)


# ─── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing FlowMatchingHead...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test standalone head
    head = FlowMatchingHead(
        feature_dim=512,
        action_dim=7,
        action_horizon=4,
        hidden_dim=256,
        num_layers=3,
    ).to(device)

    # Dummy data
    B = 4
    features = torch.randn(B, 512, device=device)
    actions_gt = torch.randn(B, 4, 7, device=device)  # (B, H, action_dim)

    # Training
    loss, info = head(features, actions_gt)
    print(f"  Flow loss: {loss.item():.4f}")
    print(f"  Velocity norm: {info['velocity_norm']:.4f}")

    # Inference
    pred_actions = head.sample(features, num_steps=10)
    print(f"  Predicted actions shape: {pred_actions.shape}")
    print(f"  Action range: [{pred_actions.min():.3f}, {pred_actions.max():.3f}]")

    # Count params
    total = sum(p.numel() for p in head.parameters())
    print(f"  FlowMatchingHead params: {total/1e6:.2f}M")

    print("\n✅ FlowMatchingHead test passed!")

    # Test full VLA (only if transformers available)
    try:
        print("\n" + "=" * 60)
        print("Testing FlowMatchingVLA (lightweight)...")
        print("=" * 60)

        vla = FlowMatchingVLA(
            action_dim=7,
            action_horizon=4,
            flow_hidden_dim=256,
            flow_num_layers=3,
        ).to(device)

        images = torch.randn(2, 3, 224, 224, device=device)
        instructions = ["pick up the red cube", "lift the blue block"]
        actions_gt = torch.randn(2, 4, 7, device=device)

        loss, info = vla(images, instructions, actions_gt)
        print(f"  Training loss: {loss.item():.4f}")

        pred = vla.predict(images, instructions, num_steps=5)
        print(f"  Predicted actions: {pred.shape}")

        total = sum(p.numel() for p in vla.parameters())
        print(f"  Total VLA params: {total/1e6:.1f}M")

        print("\n✅ FlowMatchingVLA test passed!")

    except ImportError as e:
        print(f"  Skipping VLA test (missing: {e})")
