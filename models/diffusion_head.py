"""
Diffusion Action Decoder (Diffusion-Policy-inspired)
=====================================================
Implements DDPM / DDIM denoising for VLA action generation.

Instead of autoregressive token generation (OpenVLA) or ODE integration
(flow matching), diffusion iteratively denoises a random sample into
an action via a learned noise-prediction network ε_θ.

Key characteristics:
- Predicts noise ε_θ(x_t, t, features) at each denoising step
- DDPM training: standard denoising score matching
- DDIM inference: deterministic, supports fewer steps (10-20 vs 50-100)
- Supports action chunking (predict H future actions at once)

Inspired by:
- Diffusion Policy (Chi et al., 2023)
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Denoising Diffusion Implicit Models (Song et al., 2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionHead(nn.Module):
    """
    Diffusion-based action decoder head.

    Takes VLA backbone features and generates actions via iterative denoising.

    Training (DDPM):
        1. Sample t ~ Uniform{1, ..., T}
        2. Sample noise ε ~ N(0, I)
        3. Compute x_t = √ᾱ_t * action_gt + √(1-ᾱ_t) * ε
        4. Predict noise: ε̂ = noise_net(x_t, t, features)
        5. Loss = ||ε̂ - ε||²

    Inference (DDIM, deterministic):
        1. Start from x_T ~ N(0, I)
        2. For each step: predict ε̂, compute x_{t-1}
        3. After K steps: x_0 ≈ predicted_action
    """

    def __init__(
        self,
        feature_dim=4096,       # VLA backbone output dim
        action_dim=7,           # 7-DOF: dx,dy,dz,dax,day,daz,gripper
        action_horizon=4,       # Action chunking: predict H future actions
        hidden_dim=512,
        num_layers=4,
        dropout=0.1,
        num_train_timesteps=100,  # DDPM diffusion steps T
        num_inference_steps=10,   # DDIM inference steps (fewer = faster)
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",  # cosine schedule (improved DDPM)
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.total_action_dim = action_dim * action_horizon
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # ── Noise schedule ──
        betas = self._make_beta_schedule(
            beta_schedule, num_train_timesteps, beta_start, beta_end
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # ── Feature projection ──
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # ── Time embedding (sinusoidal) ──
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ── Noise prediction network ──
        # ε_θ(x_t, t, features) : (noisy_action, time_embed, features) → noise
        layers = []
        input_dim = self.total_action_dim + hidden_dim + hidden_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else self.total_action_dim
            layers.extend([
                nn.Linear(input_dim if i == 0 else hidden_dim, out_dim),
                *(
                    []
                    if i == num_layers - 1
                    else [nn.SiLU(), nn.LayerNorm(out_dim), nn.Dropout(dropout)]
                ),
            ])
        self.noise_net = nn.Sequential(*layers)

        # ── Action normalization stats ──
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

        self._init_weights()

    @staticmethod
    def _make_beta_schedule(schedule, num_timesteps, beta_start, beta_end):
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "squaredcos_cap_v2":
            # Improved DDPM cosine schedule (Nichol & Dhariwal 2021)
            steps = num_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_timesteps, steps) / num_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_action_stats(self, mean, std):
        """Set action normalization statistics from training data."""
        self.action_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.action_std.copy_(
            torch.tensor(std, dtype=torch.float32).clamp(min=1e-6)
        )

    def normalize_action(self, action):
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action):
        return action * self.action_std + self.action_mean

    def forward(self, features, action_gt):
        """
        Training forward pass: compute DDPM denoising loss.

        Args:
            features: (B, feature_dim) VLA backbone features
            action_gt: (B, H, action_dim) ground truth action sequences

        Returns:
            loss: scalar denoising loss
            info: dict with training metrics
        """
        B = features.shape[0]
        device = features.device

        # Flatten and normalize
        action_flat = action_gt.reshape(B, -1)
        action_norm = self.normalize_action(
            action_flat.reshape(B, self.action_horizon, self.action_dim)
        ).reshape(B, -1)

        # Sample random timesteps
        t = torch.randint(
            0, self.num_train_timesteps, (B,), device=device
        ).long()

        # Sample noise
        noise = torch.randn_like(action_norm)

        # Forward diffusion: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_alpha * action_norm + sqrt_one_minus_alpha * noise

        # Predict noise
        proj_features = self.feature_proj(features)
        t_normalized = t.float() / self.num_train_timesteps
        time_embed = self.time_embed(t_normalized)
        pred_noise = self.noise_net(
            torch.cat([x_t, time_embed, proj_features], dim=-1)
        )

        # Denoising loss
        loss = F.mse_loss(pred_noise, noise)

        info = {
            "diffusion_loss": loss.item(),
            "noise_pred_norm": pred_noise.norm(dim=-1).mean().item(),
        }

        return loss, info

    @torch.no_grad()
    def sample(self, features, num_steps=None):
        """
        Inference: generate actions via DDIM deterministic sampling.

        Args:
            features: (B, feature_dim) VLA backbone features
            num_steps: number of DDIM steps (default: self.num_inference_steps)

        Returns:
            actions: (B, H, action_dim) predicted action sequence
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        B = features.shape[0]
        device = features.device

        proj_features = self.feature_proj(features)

        # Start from pure noise
        x = torch.randn(B, self.total_action_dim, device=device)

        # DDIM timestep subsequence (evenly spaced)
        step_ratio = self.num_train_timesteps // num_steps
        timesteps = (
            (np.arange(0, num_steps) * step_ratio)
            .round()
            .astype(np.int64)
        )
        timesteps = np.flip(timesteps)  # reverse: T, T-1, ..., 0

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_normalized = t_tensor.float() / self.num_train_timesteps
            time_embed = self.time_embed(t_normalized)

            # Predict noise
            pred_noise = self.noise_net(
                torch.cat([x, time_embed, proj_features], dim=-1)
            )

            # DDIM update (deterministic, η=0)
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0)
            )

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(
                alpha_t
            )
            pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise

            # x_{t-1}
            x = torch.sqrt(alpha_prev) * pred_x0 + dir_xt

        # Denormalize and reshape
        actions = self.denormalize_action(
            x.reshape(B, self.action_horizon, self.action_dim)
        )

        return actions


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time embedding (shared with flow matching models)."""

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


# ─── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DiffusionHead...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test standalone head
    head = DiffusionHead(
        feature_dim=512,
        action_dim=7,
        action_horizon=4,
        hidden_dim=256,
        num_layers=3,
        num_train_timesteps=100,
        num_inference_steps=10,
    ).to(device)

    # Dummy data
    B = 4
    features = torch.randn(B, 512, device=device)
    actions_gt = torch.randn(B, 4, 7, device=device)  # (B, H, action_dim)

    # Training
    loss, info = head(features, actions_gt)
    print(f"✅ Diffusion loss: {loss.item():.4f}")
    print(f"   Noise pred norm: {info['noise_pred_norm']:.4f}")

    # Inference
    pred_actions = head.sample(features, num_steps=10)
    print(f"✅ Predicted actions: {pred_actions.shape}")
    print(f"   Action range: [{pred_actions.min():.3f}, {pred_actions.max():.3f}]")

    # Count params
    total = sum(p.numel() for p in head.parameters())
    print(f"   DiffusionHead params: {total / 1e6:.2f}M")

    print("\n✅ DiffusionHead test passed!")
