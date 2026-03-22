#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Kaggle Notebook 3: Flow-Matching VLA Training + Closed-Loop
                     Evaluation + GIF Generation
═══════════════════════════════════════════════════════════════════

Prerequisites:
  - Kaggle T4 GPU accelerator enabled
  - Demo data from Notebook 1 (uploaded as Kaggle Dataset)

This notebook:
1. Trains a lightweight Flow-Matching VLA (ViT-B + BERT + FlowHead)
2. Runs closed-loop evaluation in MuJoCo with all 3 decoders
3. Generates comparison GIFs, trajectory plots, and success heatmap
4. Produces the final presentation assets

⏱️ Estimated time: 30-60 minutes (training) + 10 min (eval)
💾 GPU memory: ~8 GB (lightweight model fits T4 easily)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import json
import re
import subprocess
import sys
from pathlib import Path

def install():
    pkgs = [
        "torch==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "accelerate==0.30.1",
        "peft==0.11.1",
        "bitsandbytes==0.43.1",
        "timm==0.9.10",
        "mujoco>=3.0.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "imageio>=2.30.0",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs)
    subprocess.run(
                   "apt-get update -qq && apt-get install -y -qq "
                   "libgl1-mesa-glx libgl1-mesa-dev libegl1-mesa-dev "
                   "libosmesa6-dev libglew-dev patchelf",
                   shell=True, capture_output=True)
    print("✅ Dependencies installed")

install()

import os

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs._rendering import configure_headless_rendering

backend = configure_headless_rendering()

import math
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import mujoco

print(f"✅ Loaded: MuJoCo {mujoco.__version__}, PyTorch {torch.__version__}")
print(f"   Rendering backend: {backend or os.environ.get('MUJOCO_GL', 'default')}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════
# Cell 2: Configuration
# ═══════════════════════════════════════════════════════════════

DEMO_DIR = "/kaggle/input/vla-demos/demos"
OUTPUT_DIR = "/kaggle/working/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training config
FLOW_TRAIN_EPOCHS = 30
DIFFUSION_TRAIN_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224
ACTION_DIM = 7       # dx, dy, dz, dax, day, daz, gripper
ACTION_HORIZON = 4   # predict 4 future actions (action chunking)
NUM_EVAL_EPISODES = 20
EVAL_MAX_STEPS = 100
OPENVLA_BASE_MODEL = "openvla/openvla-7b"
OPENVLA_CANDIDATE_DIRS = [
    "/kaggle/input/openvla-finetuned/final",
    "/kaggle/input/openvla-finetuned/openvla-finetuned/final",
    "/kaggle/working/openvla-finetuned/final",
    str(PROJECT_ROOT / "openvla-finetuned" / "final"),
]

FRANKA_ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
TRANSLATION_STEP_M = 0.015
ROTATION_STEP_RAD = 0.05
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSE_VALUE = 1.0
ACTION_MIN = -1.0
ACTION_MAX = 1.0


def find_existing_directory(candidates):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


OPENVLA_MODEL_DIR = find_existing_directory(OPENVLA_CANDIDATE_DIRS)
print(f"   OpenVLA adapter dir: {OPENVLA_MODEL_DIR or 'not found yet'}")


def ensure_franka_action_7d(action, source_name="<unknown>"):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 7:
        return np.clip(action, ACTION_MIN, ACTION_MAX)
    if action.shape[0] == 4:
        return np.array(
            [action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]],
            dtype=np.float32,
        )
    raise ValueError(
        f"Unsupported action dimension {action.shape[0]} in {source_name}. "
        "Expected 7-DOF Franka actions or legacy 4-DOF actions."
    )


def format_vla_prompt(instruction):
    return (
        f"In: What normalized Franka Panda delta-pose action should the robot take to {instruction}?\n"
        "Out: Return dx dy dz dax day daz in [-1, 1] and gripper=open|close.\n"
        f"dx dy dz are Cartesian deltas scaled by {TRANSLATION_STEP_M:.3f} m/step. "
        f"dax day daz are angular deltas scaled by {ROTATION_STEP_RAD:.2f} rad/step. "
        "Use gripper=close for positive commands and gripper=open for negative commands.\n"
        "Action:"
    )


FRANKA_ACTION_PATTERN = re.compile(
    r"dx=(?P<dx>[+-]?\d+(?:\.\d+)?)\s+"
    r"dy=(?P<dy>[+-]?\d+(?:\.\d+)?)\s+"
    r"dz=(?P<dz>[+-]?\d+(?:\.\d+)?)\s+"
    r"dax=(?P<dax>[+-]?\d+(?:\.\d+)?)\s+"
    r"day=(?P<day>[+-]?\d+(?:\.\d+)?)\s+"
    r"daz=(?P<daz>[+-]?\d+(?:\.\d+)?)\s+"
    r"gripper=(?P<gripper>open|close)"
)


def parse_franka_action(text):
    match = FRANKA_ACTION_PATTERN.search(text)
    if match is None:
        return None
    values = [float(match.group(key)) for key in FRANKA_ACTION_KEYS[:-1]]
    values.append(GRIPPER_CLOSE_VALUE if match.group("gripper") == "close" else GRIPPER_OPEN_VALUE)
    return ensure_franka_action_7d(values)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# Cell 3: Flow-Matching Head (π0-inspired)
# ═══════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class FlowMatchingHead(nn.Module):
    """
    Flow-matching action decoder (π0/Physical Intelligence inspired).

    Training: Learn velocity field v_θ(x_t, t, z) where
      x_t = (1-t)*noise + t*action, t ~ Beta(1.5, 1)
      Loss = ||v_θ - (action - noise)||²

    Inference: Euler ODE from noise → action in K steps.
    """

    def __init__(self, feature_dim, action_dim=7, horizon=4,
                 hidden_dim=512, num_layers=4):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.total_dim = action_dim * horizon

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())

        layers = []
        in_dim = self.total_dim + 2 * hidden_dim
        for i in range(num_layers):
            out = hidden_dim if i < num_layers - 1 else self.total_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out))
            if i < num_layers - 1:
                layers.extend([nn.SiLU(), nn.LayerNorm(out), nn.Dropout(0.1)])
        self.vel_net = nn.Sequential(*layers)

        self.register_buffer('act_mean', torch.zeros(action_dim))
        self.register_buffer('act_std', torch.ones(action_dim))

    def set_stats(self, mean, std):
        self.act_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        self.act_std.copy_(torch.tensor(std, dtype=torch.float32).clamp(min=1e-6))

    def normalize(self, a):
        return (a - self.act_mean) / self.act_std

    def denormalize(self, a):
        return a * self.act_std + self.act_mean

    def forward(self, features, action_gt):
        """Training: compute flow matching loss."""
        B = features.shape[0]
        H = action_gt.shape[1]
        a_flat = self.normalize(action_gt).reshape(B, -1)

        t = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(features.device)
        noise = torch.randn_like(a_flat)
        x_t = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * a_flat
        target_v = a_flat - noise

        feat = self.feat_proj(features)
        t_emb = self.time_embed(t)
        pred_v = self.vel_net(torch.cat([x_t, t_emb, feat], dim=-1))

        loss = F.mse_loss(pred_v, target_v)
        return loss, {'flow_loss': loss.item()}

    @torch.no_grad()
    def sample(self, features, steps=10):
        """Inference: Euler ODE integration."""
        B = features.shape[0]
        feat = self.feat_proj(features)
        x = torch.randn(B, self.total_dim, device=features.device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((B,), i * dt, device=features.device)
            t_emb = self.time_embed(t)
            v = self.vel_net(torch.cat([x, t_emb, feat], dim=-1))
            x = x + v * dt

        actions = self.denormalize(x.reshape(B, self.horizon, self.action_dim))
        return actions

# ═══════════════════════════════════════════════════════════════
# Cell 4: Lightweight VLA Model
# ═══════════════════════════════════════════════════════════════

from transformers import BertModel, BertTokenizer, ViTModel
from torchvision import transforms
from models.diffusion_head import DiffusionHead


class FlowMatchingVLA(nn.Module):
    """
    Lightweight VLA with flow-matching decoder.
    Vision: ViT-B/16 (86M) + Text: BERT-small (29M) + Flow head (~2M)
    Total: ~117M params — trains easily on T4
    """

    def __init__(self, action_dim=7, horizon=4):
        super().__init__()
        # Vision encoder
        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vision.parameters():
            param.requires_grad = False  # freeze vision backbone
        # Unfreeze last 2 layers for fine-tuning
        for layer in self.vision.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Text encoder
        self.text_model = BertModel.from_pretrained("prajjwal1/bert-small")
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
        for param in self.text_model.parameters():
            param.requires_grad = False  # freeze
        for layer in self.text_model.encoder.layer[-1:]:
            for param in layer.parameters():
                param.requires_grad = True

        vis_dim = self.vision.config.hidden_size     # 768
        txt_dim = self.text_model.config.hidden_size  # 512
        fuse_dim = 512

        self.vis_proj = nn.Linear(vis_dim, fuse_dim)
        self.txt_proj = nn.Linear(txt_dim, fuse_dim)
        self.fusion = nn.TransformerEncoderLayer(
            d_model=fuse_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True)

        self.flow_head = FlowMatchingHead(
            feature_dim=fuse_dim, action_dim=action_dim,
            horizon=horizon, hidden_dim=512, num_layers=4)

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def encode(self, images, instructions):
        """Fuse vision + language features."""
        # Vision
        vis_out = self.vision(pixel_values=images).last_hidden_state[:, 0]
        vis_feat = self.vis_proj(vis_out)

        # Text
        txt_in = self.tokenizer(instructions, return_tensors="pt", padding=True,
                                truncation=True, max_length=64).to(images.device)
        txt_out = self.text_model(**txt_in).last_hidden_state[:, 0]
        txt_feat = self.txt_proj(txt_out)

        combined = torch.stack([vis_feat, txt_feat], dim=1)
        fused = self.fusion(combined).mean(dim=1)
        return fused

    def forward(self, images, instructions, actions_gt):
        """Training: images + text + gt_actions → loss."""
        features = self.encode(images, instructions)
        return self.flow_head(features, actions_gt)

    @torch.no_grad()
    def predict(self, images, instructions, steps=10):
        """Inference: images + text → predicted actions."""
        features = self.encode(images, instructions)
        return self.flow_head.sample(features, steps=steps)


print("✅ FlowMatchingVLA defined")


class DiffusionVLA(nn.Module):
    """
    Lightweight VLA with diffusion action decoder.
    Shares the same vision/language backbone as FlowMatchingVLA.
    """

    def __init__(self, action_dim=7, horizon=4):
        super().__init__()
        # Vision encoder
        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vision.parameters():
            param.requires_grad = False
        for layer in self.vision.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Text encoder
        self.text_model = BertModel.from_pretrained("prajjwal1/bert-small")
        self.tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
        for param in self.text_model.parameters():
            param.requires_grad = False
        for layer in self.text_model.encoder.layer[-1:]:
            for param in layer.parameters():
                param.requires_grad = True

        vis_dim = self.vision.config.hidden_size
        txt_dim = self.text_model.config.hidden_size
        fuse_dim = 512

        self.vis_proj = nn.Linear(vis_dim, fuse_dim)
        self.txt_proj = nn.Linear(txt_dim, fuse_dim)
        self.fusion = nn.TransformerEncoderLayer(
            d_model=fuse_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True)

        self.diffusion_head = DiffusionHead(
            feature_dim=fuse_dim,
            action_dim=action_dim,
            action_horizon=horizon,
            hidden_dim=512,
            num_layers=4,
            num_train_timesteps=100,
            num_inference_steps=10,
        )

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def encode(self, images, instructions):
        vis_out = self.vision(pixel_values=images).last_hidden_state[:, 0]
        vis_feat = self.vis_proj(vis_out)

        txt_in = self.tokenizer(instructions, return_tensors="pt", padding=True,
                                truncation=True, max_length=64).to(images.device)
        txt_out = self.text_model(**txt_in).last_hidden_state[:, 0]
        txt_feat = self.txt_proj(txt_out)

        combined = torch.stack([vis_feat, txt_feat], dim=1)
        fused = self.fusion(combined).mean(dim=1)
        return fused

    def set_action_stats(self, mean, std):
        self.diffusion_head.set_action_stats(mean, std)

    def forward(self, images, instructions, actions_gt):
        features = self.encode(images, instructions)
        return self.diffusion_head(features, actions_gt)

    @torch.no_grad()
    def predict(self, images, instructions, steps=10):
        features = self.encode(images, instructions)
        return self.diffusion_head.sample(features, num_steps=steps)


print("✅ DiffusionVLA defined")

# ═══════════════════════════════════════════════════════════════
# Cell 5: Dataset for Flow-Matching Training
# ═══════════════════════════════════════════════════════════════

class FlowDemoDataset(Dataset):
    """Dataset with action chunking (horizon=4)."""

    def __init__(self, demo_dir, horizon=4, img_size=224):
        self.horizon = horizon
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        self.samples = []
        files = sorted(glob.glob(os.path.join(demo_dir, "demo_*.npz")))

        all_actions = []
        for f in files:
            data = np.load(f, allow_pickle=True)
            if not data.get("success", False):
                continue
            images = data["images"]
            actions = data["actions"]
            instructions = data["instructions"]

            # Create samples with action horizons
            for t in range(len(actions) - horizon + 1):
                action_chunk = actions[t:t+horizon]
                if action_chunk.shape[-1] == 4:
                    padded = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
                    padded[:, :3] = action_chunk[:, :3]
                    padded[:, 6] = action_chunk[:, 3]
                    action_chunk = padded
                elif action_chunk.shape[-1] != ACTION_DIM:
                    raise ValueError(
                        f"Unsupported demo action dim {action_chunk.shape[-1]} in {f}. "
                        f"Expected {ACTION_DIM} or legacy 4."
                    )

                self.samples.append({
                    "image": images[t],
                    "instruction": str(instructions[t]),
                    "actions": action_chunk.astype(np.float32),  # (H, 7)
                })
                all_actions.append(action_chunk[0])

        # Compute action stats
        all_actions = np.array(all_actions)
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0)
        print(f"  Loaded {len(self.samples)} samples, action_mean={self.action_mean}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.fromarray(s["image"])
        img_t = self.transform(img)
        acts = torch.tensor(s["actions"], dtype=torch.float32)
        return img_t, s["instruction"], acts


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    instrs = [b[1] for b in batch]
    actions = torch.stack([b[2] for b in batch])
    return imgs, instrs, actions


# Load dataset
if not os.path.exists(DEMO_DIR):
    raise FileNotFoundError(
        f"Demo dir not found: {DEMO_DIR}. Run Notebook 1 first and attach its demos dataset."
    )

dataset = FlowDemoDataset(DEMO_DIR, horizon=ACTION_HORIZON)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)
print(f"✅ Dataset ready: {len(dataset)} samples, {len(loader)} batches")


def set_model_action_stats(model, mean, std):
    if hasattr(model, "flow_head"):
        model.flow_head.set_stats(mean, std)
    elif hasattr(model, "diffusion_head"):
        model.diffusion_head.set_action_stats(mean, std)
    else:
        raise AttributeError("Model does not expose a supported action head.")


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return total, trainable


def train_vla_model(model, loader, num_epochs, run_name, checkpoint_name, curve_name):
    set_model_action_stats(model, dataset.action_mean, dataset.action_std)
    total, trainable = count_parameters(model)
    print(f"✅ {run_name}: {total/1e6:.1f}M params, {trainable/1e6:.1f}M trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs * len(loader), 1),
    )

    print(f"\n{'='*60}")
    print(f"Training {run_name} ({num_epochs} epochs)")
    print(f"{'='*60}")

    model.train()
    train_losses = []
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (imgs, instrs, acts) in enumerate(loader):
            imgs, acts = imgs.to(DEVICE), acts.to(DEVICE)

            loss, info = model(imgs, instrs, acts)
            loss.backward()

            if (batch_idx + 1) % 2 == 0 or (batch_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        avg = epoch_loss / max(len(loader), 1)
        train_losses.append(avg)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            metric_name, metric_value = next(iter(info.items()))
            print(
                f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg:.4f} | "
                f"{metric_name}: {metric_value:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "action_mean": dataset.action_mean,
        "action_std": dataset.action_std,
        "train_losses": train_losses,
        "run_name": run_name,
    }
    torch.save(checkpoint, os.path.join(OUTPUT_DIR, checkpoint_name))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve: {run_name}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, curve_name), dpi=150)
    if run_name == "flow_matching":
        plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"), dpi=150)
    plt.close()

    print(f"✅ {run_name} training complete! Final loss: {train_losses[-1]:.4f}")
    model.eval()
    return model, train_losses


# ═══════════════════════════════════════════════════════════════
# Cell 6: Train Flow-Matching + Diffusion VLAs
# ═══════════════════════════════════════════════════════════════

flow_model = FlowMatchingVLA(action_dim=ACTION_DIM, horizon=ACTION_HORIZON).to(DEVICE)
flow_model, flow_train_losses = train_vla_model(
    flow_model,
    loader,
    FLOW_TRAIN_EPOCHS,
    run_name="flow_matching",
    checkpoint_name="flow_matching_vla.pt",
    curve_name="flow_matching_training_curve.png",
)

diffusion_model = DiffusionVLA(action_dim=ACTION_DIM, horizon=ACTION_HORIZON).to(DEVICE)
diffusion_model, diffusion_train_losses = train_vla_model(
    diffusion_model,
    loader,
    DIFFUSION_TRAIN_EPOCHS,
    run_name="diffusion",
    checkpoint_name="diffusion_vla.pt",
    curve_name="diffusion_training_curve.png",
)


# ═══════════════════════════════════════════════════════════════
# Cell 7: MuJoCo Environment + Comparison Evaluator
# ═══════════════════════════════════════════════════════════════

from envs.franka_grasp_env import FrankaGraspEnv
from evaluation.closed_loop_eval import VLAMuJoCoEvaluator
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForVision2Seq as OpenVLAModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as OpenVLAModelClass


class FlowMatchingPolicyWrapper:
    decoder_type = "flow_matching"

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def predict_action(self, image_np, instruction, **_kwargs):
        img = Image.fromarray(image_np)
        img_t = self.model.img_transform(img).unsqueeze(0).to(DEVICE)
        sync_cuda()
        start = time.time()
        actions = self.model.predict(img_t, [instruction], steps=10)
        sync_cuda()
        action = ensure_franka_action_7d(actions[0, 0].detach().cpu().numpy(), "flow_prediction")
        return action, {"inference_time_ms": (time.time() - start) * 1000}


class DiffusionPolicyWrapper:
    decoder_type = "diffusion"

    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def predict_action(self, image_np, instruction, **_kwargs):
        img = Image.fromarray(image_np)
        img_t = self.model.img_transform(img).unsqueeze(0).to(DEVICE)
        sync_cuda()
        start = time.time()
        actions = self.model.predict(img_t, [instruction], steps=10)
        sync_cuda()
        action = ensure_franka_action_7d(actions[0, 0].detach().cpu().numpy(), "diffusion_prediction")
        return action, {"inference_time_ms": (time.time() - start) * 1000}


class OpenVLAPolicyWrapper:
    decoder_type = "autoregressive"

    def __init__(self, adapter_dir, base_model_name=OPENVLA_BASE_MODEL):
        self.adapter_dir = adapter_dir
        self.base_model_name = base_model_name
        self.model = None
        self.processor = None
        self.input_device = torch.device(DEVICE)

    def _load(self):
        if self.adapter_dir is None:
            raise FileNotFoundError(
                "OpenVLA adapter dir not found. Run Notebook 2 first and attach "
                "`openvla-finetuned/final` to Notebook 3."
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=torch.cuda.is_available(),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base_model = OpenVLAModelClass.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.processor = AutoProcessor.from_pretrained(
            self.adapter_dir,
            trust_remote_code=True,
        )
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict_action(self, image_np, instruction, **_kwargs):
        if self.model is None:
            self._load()

        prompt = format_vla_prompt(instruction)
        img = Image.fromarray(image_np)
        inputs = self.processor(
            images=[img],
            text=[prompt],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

        sync_cuda()
        start = time.time()
        generated = self.model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
        )
        sync_cuda()
        generated_text = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        action = parse_franka_action(generated_text)
        parse_failed = action is None
        if action is None:
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN_VALUE], dtype=np.float32)

        return action, {
            "inference_time_ms": (time.time() - start) * 1000,
            "generated_text": generated_text,
            "parse_failed": parse_failed,
        }

    def close(self):
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_policy(name, policy, env):
    save_dir = os.path.join(OUTPUT_DIR, name)
    evaluator = VLAMuJoCoEvaluator(
        model=policy,
        env=env,
        use_oracle_info=False,
    )
    return evaluator.evaluate(
        num_episodes=NUM_EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
        record_video=True,
        save_dir=save_dir,
        verbose=True,
    )


# ═══════════════════════════════════════════════════════════════
# Cell 8: Closed-Loop Comparison Across All 3 Methods
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Closed-Loop Evaluation: Autoregressive vs Diffusion vs Flow-Matching")
print("=" * 60)

if OPENVLA_MODEL_DIR is None:
    raise FileNotFoundError(
        "Could not find Notebook 2 output. Expected one of:\n- " +
        "\n- ".join(OPENVLA_CANDIDATE_DIRS)
    )

comparison = {}
env = FrankaGraspEnv(image_size=256, camera_name="frontview")

flow_policy = FlowMatchingPolicyWrapper(flow_model)
comparison["flow_matching"] = evaluate_policy("flow_matching", flow_policy, env)
flow_model.to("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

diffusion_policy = DiffusionPolicyWrapper(diffusion_model)
comparison["diffusion"] = evaluate_policy("diffusion", diffusion_policy, env)
diffusion_model.to("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

openvla_policy = OpenVLAPolicyWrapper(OPENVLA_MODEL_DIR)
comparison["autoregressive"] = evaluate_policy("autoregressive", openvla_policy, env)
openvla_policy.close()

env.close()


# ═══════════════════════════════════════════════════════════════
# Cell 9: Save Comparison Summary + Plots
# ═══════════════════════════════════════════════════════════════

comparison_summary = {
    name: {
        "success_rate": float(result["summary"]["success_rate"]),
        "avg_steps": float(result["summary"]["avg_steps"]),
        "std_steps": float(result["summary"]["std_steps"]),
        "avg_reward": float(result["summary"].get("avg_reward", 0.0)),
        "avg_inference_ms": float(result["summary"].get("avg_inference_ms", 0.0)),
        "p50_inference_ms": float(result["summary"].get("p50_inference_ms", 0.0)),
        "p95_inference_ms": float(result["summary"].get("p95_inference_ms", 0.0)),
        "num_episodes": int(result["summary"]["num_episodes"]),
    }
    for name, result in comparison.items()
}
save_json(os.path.join(OUTPUT_DIR, "comparison_summary.json"), comparison_summary)

ordered_names = ["autoregressive", "diffusion", "flow_matching"]
display_names = {
    "autoregressive": "Autoregressive",
    "diffusion": "Diffusion",
    "flow_matching": "Flow-Matching",
}

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)
print(f"{'Decoder':<20} {'Success%':>10} {'Avg Steps':>10} {'P50(ms)':>12}")
print("-" * 60)
for name in ordered_names:
    summary = comparison[name]["summary"]
    print(
        f"{display_names[name]:<20} {summary['success_rate']:>9.0%} "
        f"{summary['avg_steps']:>10.1f} {summary['p50_inference_ms']:>11.1f}"
    )
print("=" * 70)

with open(os.path.join(OUTPUT_DIR, "comparison_table.md"), "w") as f:
    f.write("| Decoder | Success Rate | Avg Steps | P50 Latency |\n")
    f.write("|---|---:|---:|---:|\n")
    for name in ordered_names:
        summary = comparison[name]["summary"]
        f.write(
            f"| {display_names[name]} | {summary['success_rate']:.1%} | "
            f"{summary['avg_steps']:.1f} | {summary['p50_inference_ms']:.1f} ms |\n"
        )

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
success_values = [comparison[name]["summary"]["success_rate"] for name in ordered_names]
latency_values = [comparison[name]["summary"]["p50_inference_ms"] for name in ordered_names]
bar_colors = ["#4e79a7", "#f28e2b", "#59a14f"]

axes[0].bar([display_names[n] for n in ordered_names], success_values, color=bar_colors)
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("Success Rate")
axes[0].set_title("Closed-Loop Success Rate", fontweight="bold")

axes[1].bar([display_names[n] for n in ordered_names], latency_values, color=bar_colors)
axes[1].set_ylabel("P50 Inference Latency (ms)")
axes[1].set_title("Closed-Loop Inference Latency", fontweight="bold")

for ax in axes:
    ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_metrics.png"), dpi=150)
plt.close()

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
colors = {
    "autoregressive": "#4e79a7",
    "diffusion": "#f28e2b",
    "flow_matching": "#59a14f",
}
for name in ordered_names:
    trajectory = comparison[name]["trajectories"][0]["gripper_positions"]
    pos = np.array(trajectory)
    success = comparison[name]["successes"][0]
    label = f"{display_names[name]} ({'✓' if success else '✗'})"
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=label, linewidth=2.5, color=colors[name])
    ax.scatter(*pos[0], color=colors[name], s=80, marker='o', edgecolors='black')
    ax.scatter(*pos[-1], color=colors[name], s=80, marker='s', edgecolors='black')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Decoder Comparison: End-Effector Trajectories', fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.view_init(elev=25, azim=-60)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "decoder_trajectories_3d.png"), dpi=150)
plt.savefig(os.path.join(OUTPUT_DIR, "flow_trajectories_3d.png"), dpi=150)
plt.close()


# ═══════════════════════════════════════════════════════════════
# Cell 10: Generate Technical Report from Real Results
# ═══════════════════════════════════════════════════════════════

best_success_name = max(ordered_names, key=lambda name: comparison[name]["summary"]["success_rate"])
fastest_name = min(ordered_names, key=lambda name: comparison[name]["summary"]["p50_inference_ms"])

report_lines = [
    "# Multi-Paradigm VLA for Robotic Grasping — Technical Report",
    "",
    "## 1. Overview",
    "This run compares three closed-loop VLA action decoders on the Franka Panda grasping task:",
    "- Autoregressive: fine-tuned OpenVLA from Notebook 2",
    "- Diffusion: lightweight ViT+BERT+DiffusionHead baseline",
    "- Flow-Matching: lightweight ViT+BERT+FlowMatchingHead baseline",
    "",
    "## 2. Environment",
    "- MuJoCo Franka Panda 7-DOF arm with parallel gripper",
    "- 7-DOF action interface: [dx, dy, dz, dax, day, daz, gripper]",
    "- 256×256 camera observations for closed-loop control",
    "",
    "## 3. Closed-Loop Results",
]
for name in ordered_names:
    summary = comparison[name]["summary"]
    report_lines.extend([
        f"### {display_names[name]}",
        f"- Success rate: {summary['success_rate']:.1%}",
        f"- Average steps: {summary['avg_steps']:.1f} ± {summary['std_steps']:.1f}",
        f"- P50 inference latency: {summary['p50_inference_ms']:.1f} ms",
        "",
    ])

report_lines.extend([
    "## 4. Summary",
    f"- Best success rate: {display_names[best_success_name]} ({comparison[best_success_name]['summary']['success_rate']:.1%})",
    f"- Fastest inference: {display_names[fastest_name]} ({comparison[fastest_name]['summary']['p50_inference_ms']:.1f} ms p50)",
    "- GIFs for the first 5 episodes of each decoder are saved in separate output folders.",
    "",
    "## 5. Output Files",
    "- autoregressive/episode_*.gif",
    "- diffusion/episode_*.gif",
    "- flow_matching/episode_*.gif",
    "- comparison_summary.json",
    "- comparison_table.md",
    "- comparison_metrics.png",
    "- decoder_trajectories_3d.png",
])

with open(os.path.join(OUTPUT_DIR, "technical_report.md"), "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\n✅ All outputs saved to {OUTPUT_DIR}")
print(f"   📊 training_curve.png")
print(f"   📊 diffusion_training_curve.png")
print(f"   📊 comparison_metrics.png")
print(f"   🎬 autoregressive/episode_*.gif")
print(f"   🎬 diffusion/episode_*.gif")
print(f"   🎬 flow_matching/episode_*.gif")
print(f"   📈 decoder_trajectories_3d.png")
print(f"   📄 comparison_summary.json")
print(f"   📄 comparison_table.md")
print(f"   📄 technical_report.md")

for root, _, files in os.walk(OUTPUT_DIR):
    for name in sorted(files):
        path = os.path.join(root, name)
        size = os.path.getsize(path)
        rel = os.path.relpath(path, OUTPUT_DIR)
        print(f"   📄 {rel} ({size/1024:.1f} KB)")
