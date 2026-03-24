#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Kaggle Notebook 3: Flow-Matching VLA Training + Real-Data
                     Offline Evaluation on Held-Out DROID
═══════════════════════════════════════════════════════════════════

Prerequisites:
  - Kaggle T4 GPU accelerator enabled
  - Notebook 2 finished and saved an OpenVLA adapter
  - Internet access to stream a small DROID subset from Hugging Face

This notebook:
1. Streams a small held-out slice of real DROID Franka data
2. Trains lightweight Flow-Matching and Diffusion VLAs on the train split
3. Evaluates all 3 decoders offline on held-out real robot frames
4. Saves metrics, plots, sample visualizations, and a technical report

⏱️ Estimated time: 45-90 minutes
💾 GPU memory: ~8 GB (lightweight models fit on T4)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

NUMPY_VERSION = "1.26.4"


def verify_torch_numpy_bridge():
    """Fail early if Kaggle keeps a NumPy build incompatible with the pinned torch wheel."""
    check_code = (
        "import numpy as np, torch; "
        "major = int(np.__version__.split('.')[0]); "
        "assert major < 2, f'Expected NumPy < 2, found {np.__version__}'; "
        "torch.tensor([1.0]).numpy(); "
        "print(f'numpy={np.__version__} torch={torch.__version__}')"
    )
    try:
        output = subprocess.check_output([sys.executable, "-c", check_code], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "PyTorch cannot convert tensors to NumPy in this environment. "
            f"Pin numpy=={NUMPY_VERSION} before running Notebook 3."
        ) from exc
    print(f"✅ Verified torch↔numpy bridge ({output})")


def install():
    pkgs = [
        "torch==2.2.0",
        "torchvision==0.17.0",
        "opencv-python-headless>=4.9.0",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "accelerate==0.30.1",
        "peft==0.11.1",
        "bitsandbytes==0.43.1",
        "timm==0.9.10",
        "Pillow>=9.0.0",
        f"numpy=={NUMPY_VERSION}",
        "matplotlib>=3.7.0",
        "imageio>=2.30.0",
        "imageio-ffmpeg>=0.4.9",
        "datasets",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs)
    verify_torch_numpy_bridge()
    print("✅ Dependencies installed")


install()

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import BertModel, BertTokenizer, ViTModel

from models.diffusion_head import DiffusionHead
from models.flow_matching_head import FlowMatchingHead
from data.droid_utils import (
    ACTION_MAX,
    ACTION_MIN,
    DROID_DEFAULT_FPS,
    FRANKA_ACTION_KEYS,
    GRIPPER_CLOSE_VALUE,
    GRIPPER_OPEN_VALUE,
    ROTATION_STEP_RAD,
    TRANSLATION_STEP_M,
    droid_action_to_franka_action,
    droid_cartesian_velocity_to_franka_action,
    ensure_franka_action_7d,
    image_to_uint8_array,
    iter_droid_v30_stream,
    load_droid_info,
    load_droid_task_lookup,
    sample_get,
)

print(f"✅ Loaded: PyTorch {torch.__version__}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════
# Cell 2: Configuration
# ═══════════════════════════════════════════════════════════════

OUTPUT_DIR = "/kaggle/working/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Real-data source
DROID_DATASET_REPO_CANDIDATES = [
    repo for repo in [
        os.environ.get("DROID_DATASET_REPO", "").strip() or None,
        "cadene/droid_1.0.1_v30",
    ]
    if repo
]
DROID_SPLIT = "train"
DROID_MAX_SAMPLES = 500  # keep the real-data subset small enough for low-disk Kaggle runs
DROID_EVAL_FRACTION = 0.2
DROID_FPS = DROID_DEFAULT_FPS

# Training config
FLOW_TRAIN_EPOCHS = 10
DIFFUSION_TRAIN_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
IMAGE_SIZE = 224
ACTION_DIM = 7
ACTION_HORIZON = 1
MAX_SEQ_LEN = 256
MAX_EVAL_SAMPLES = 200
QUALITATIVE_EXAMPLES = 6

OPENVLA_BASE_MODEL = "openvla/openvla-7b"
OPENVLA_CANDIDATE_DIRS = [
    "/kaggle/input/openvla-finetuned/final",
    "/kaggle/input/openvla-finetuned/openvla-finetuned/final",
    "/kaggle/working/openvla-finetuned/final",
    str(PROJECT_ROOT / "openvla-finetuned" / "final"),
]

def find_existing_directory(candidates):
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


OPENVLA_MODEL_DIR = find_existing_directory(OPENVLA_CANDIDATE_DIRS)

print("✅ Config ready")
print(f"   DROID repos: {DROID_DATASET_REPO_CANDIDATES}")
print(f"   DROID max streamed samples: {DROID_MAX_SAMPLES}")
print(f"   Held-out eval fraction: {DROID_EVAL_FRACTION:.0%}")
print(f"   DROID control rate assumption: {DROID_FPS:g} Hz")
print(f"   Action horizon: {ACTION_HORIZON}")
print(f"   OpenVLA adapter dir: {OPENVLA_MODEL_DIR or 'not found yet'}")


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
# Cell 3: DROID Real-Data Loader
# ═══════════════════════════════════════════════════════════════


def load_real_droid_records(max_samples):
    """Materialize a manageable subset of held-out real DROID frames into memory."""
    repo_id = None
    records = []
    last_exc = None
    skip_stats = {}
    for candidate in DROID_DATASET_REPO_CANDIDATES:
        try:
            task_lookup = load_droid_task_lookup(candidate)
            droid_info = load_droid_info(candidate)
        except Exception as exc:
            last_exc = exc
            skip_stats[candidate] = {"load_error": f"{type(exc).__name__}: {exc}"}
            continue

        candidate_skip_stats = {
            "unsuccessful": 0,
            "missing_image": 0,
            "missing_instruction": 0,
            "missing_action": 0,
            "bad_image": 0,
            "bad_action": 0,
        }

        max_raw_droid_frames = max(max_samples * 8, 2000)
        for idx, sample in enumerate(
            iter_droid_v30_stream(
                candidate,
                split=DROID_SPLIT,
            )
        ):
            success = sample_get(sample, "is_episode_successful")
            if success is False:
                candidate_skip_stats["unsuccessful"] += 1
                continue

            image = sample_get(sample, "decoded_image")
            if image is None:
                if sample_get(sample, "decode_error") is not None:
                    candidate_skip_stats["bad_image"] += 1
                else:
                    candidate_skip_stats["missing_image"] += 1
                if idx >= max_raw_droid_frames and not records:
                    break
                continue

            instruction = sample_get(
                sample,
                "language_instruction",
                "language_instruction_2",
                "language_instruction_3",
                "episode_instruction",
            )
            if instruction is None:
                task_index = sample_get(sample, "task_index")
                if task_index is not None:
                    instruction = task_lookup.get(int(task_index))
            if isinstance(instruction, str) and not instruction.strip():
                instruction = None
            if instruction is None:
                candidate_skip_stats["missing_instruction"] += 1
                if idx >= max_raw_droid_frames and not records:
                    break
                continue

            raw_action = sample_get(sample, "action.original", "action")
            cartesian_velocity = sample_get(sample, "action.cartesian_velocity")
            gripper_position = sample_get(sample, "action.gripper_position")
            gripper_velocity = sample_get(sample, "action.gripper_velocity")
            if cartesian_velocity is None and raw_action is None:
                candidate_skip_stats["missing_action"] += 1
                continue

            source_name = f"{candidate}:{idx}"
            try:
                if cartesian_velocity is not None:
                    action_7d = droid_cartesian_velocity_to_franka_action(
                        cartesian_velocity,
                        gripper_position=gripper_position,
                        gripper_velocity=gripper_velocity,
                        source_name=source_name,
                    )
                else:
                    action_7d = droid_action_to_franka_action(raw_action, source_name=source_name)
            except Exception:
                candidate_skip_stats["bad_action"] += 1
                continue

            try:
                image_arr = image_to_uint8_array(image, source_name)
            except Exception:
                candidate_skip_stats["bad_image"] += 1
                continue

            episode_index = sample_get(sample, "episode_index")
            if episode_index is None:
                episode_index = int(idx)
            frame_index = sample_get(sample, "frame_index")
            if frame_index is None:
                frame_index = 0

            records.append({
                "image": image_arr,
                "instruction": str(instruction),
                "action_7d": action_7d,
                "episode_index": int(np.asarray(episode_index).reshape(-1)[0]),
                "frame_index": int(np.asarray(frame_index).reshape(-1)[0]),
                "source": "droid",
                "sample_index": idx,
            })
            repo_id = candidate
            if len(records) >= max_samples:
                break

        skip_stats[candidate] = candidate_skip_stats
        if records:
            break

    if not records:
        raise RuntimeError(
            "No usable DROID samples were loaded for offline evaluation. "
            f"Skip stats: {skip_stats}"
        ) from last_exc

    print(
        f"✅ Loaded {len(records)} real DROID frames from {repo_id} "
        f"at {droid_info.get('fps', DROID_FPS)} Hz"
    )
    print(f"   DROID skip stats: {skip_stats}")
    return repo_id, records


def split_records_by_episode(records, eval_fraction):
    """Split by episode id to avoid train/eval leakage from adjacent frames."""
    records = sorted(records, key=lambda r: (r["episode_index"], r["frame_index"]))
    episode_ids = []
    seen = set()
    for record in records:
        episode_index = record["episode_index"]
        if episode_index not in seen:
            seen.add(episode_index)
            episode_ids.append(episode_index)

    if len(episode_ids) < 2:
        raise RuntimeError(
            f"Need at least 2 DROID episodes for an offline split, found {len(episode_ids)}."
        )

    eval_episodes = max(1, int(round(len(episode_ids) * eval_fraction)))
    eval_episodes = min(eval_episodes, len(episode_ids) - 1)
    eval_episode_ids = set(episode_ids[-eval_episodes:])

    train_records = [r for r in records if r["episode_index"] not in eval_episode_ids]
    eval_records = [r for r in records if r["episode_index"] in eval_episode_ids]

    if not train_records or not eval_records:
        raise RuntimeError(
            f"Invalid DROID split: {len(train_records)} train records, {len(eval_records)} eval records."
        )

    print(
        "✅ DROID offline split ready: "
        f"{len(train_records)} train frames, {len(eval_records)} eval frames, "
        f"{len(episode_ids) - eval_episodes} train episodes, {eval_episodes} eval episodes"
    )
    return train_records, eval_records


def compute_action_stats(records):
    actions = np.stack([record["action_7d"] for record in records], axis=0)
    return actions.mean(axis=0), actions.std(axis=0).clip(min=1e-6)


class RealRobotActionDataset(Dataset):
    """Frame-level DROID dataset for offline one-step action prediction."""

    def __init__(self, records, img_size=224):
        self.records = list(records)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img = Image.fromarray(record["image"])
        img_t = self.transform(img)
        action = torch.tensor(record["action_7d"], dtype=torch.float32).unsqueeze(0)
        return img_t, record["instruction"], action


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    instrs = [b[1] for b in batch]
    actions = torch.stack([b[2] for b in batch])
    return imgs, instrs, actions


ACTIVE_DROID_REPO, all_droid_records = load_real_droid_records(DROID_MAX_SAMPLES)
train_records, eval_records = split_records_by_episode(all_droid_records, DROID_EVAL_FRACTION)
train_action_mean, train_action_std = compute_action_stats(train_records)

train_dataset = RealRobotActionDataset(train_records, img_size=IMAGE_SIZE)
eval_dataset = RealRobotActionDataset(eval_records, img_size=IMAGE_SIZE)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

print(f"✅ Offline datasets ready: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples")
print(f"   Using DROID repo: {ACTIVE_DROID_REPO}")


# ═══════════════════════════════════════════════════════════════
# Cell 4: Lightweight Offline VLA Models
# ═══════════════════════════════════════════════════════════════


class FlowMatchingVLA(nn.Module):
    """Lightweight VLA with a flow-matching decoder."""

    def __init__(self, action_dim=7, horizon=1):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vision.parameters():
            param.requires_grad = False
        for layer in self.vision.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

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
            d_model=fuse_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.flow_head = FlowMatchingHead(
            feature_dim=fuse_dim,
            action_dim=action_dim,
            action_horizon=horizon,
            hidden_dim=512,
            num_layers=4,
            num_inference_steps=10,
        )
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def encode(self, images, instructions):
        vis_out = self.vision(pixel_values=images).last_hidden_state[:, 0]
        vis_feat = self.vis_proj(vis_out)

        txt_in = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(images.device)
        txt_out = self.text_model(**txt_in).last_hidden_state[:, 0]
        txt_feat = self.txt_proj(txt_out)

        combined = torch.stack([vis_feat, txt_feat], dim=1)
        fused = self.fusion(combined).mean(dim=1)
        return fused

    def forward(self, images, instructions, actions_gt):
        features = self.encode(images, instructions)
        return self.flow_head(features, actions_gt)

    def set_action_stats(self, mean, std):
        self.flow_head.set_action_stats(mean, std)

    @torch.no_grad()
    def predict(self, images, instructions, steps=10):
        features = self.encode(images, instructions)
        return self.flow_head.sample(features, num_steps=steps)


class DiffusionVLA(nn.Module):
    """Lightweight VLA with a diffusion action decoder."""

    def __init__(self, action_dim=7, horizon=1):
        super().__init__()
        self.vision = ViTModel.from_pretrained("google/vit-base-patch16-224")
        for param in self.vision.parameters():
            param.requires_grad = False
        for layer in self.vision.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

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
            d_model=fuse_dim, nhead=8, dim_feedforward=1024, dropout=0.1, batch_first=True)
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
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def encode(self, images, instructions):
        vis_out = self.vision(pixel_values=images).last_hidden_state[:, 0]
        vis_feat = self.vis_proj(vis_out)

        txt_in = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(images.device)
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


print("✅ FlowMatchingVLA and DiffusionVLA defined")


def set_model_action_stats(model, mean, std):
    if hasattr(model, "set_action_stats"):
        model.set_action_stats(mean, std)
    else:
        raise AttributeError("Model does not expose a supported action head.")


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return total, trainable


def train_vla_model(model, loader, num_epochs, run_name, checkpoint_name, curve_name, action_mean, action_std):
    set_model_action_stats(model, action_mean, action_std)
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

    print(f"\n{'=' * 60}")
    print(f"Training {run_name} on real DROID data ({num_epochs} epochs)")
    print(f"{'=' * 60}")

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

        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == num_epochs - 1:
            metric_name, metric_value = next(iter(info.items()))
            print(
                f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg:.4f} | "
                f"{metric_name}: {metric_value:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "action_mean": action_mean,
        "action_std": action_std,
        "train_losses": train_losses,
        "run_name": run_name,
        "data_repo": ACTIVE_DROID_REPO,
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
# Cell 5: Train Flow-Matching + Diffusion VLAs on DROID
# ═══════════════════════════════════════════════════════════════

flow_model = FlowMatchingVLA(action_dim=ACTION_DIM, horizon=ACTION_HORIZON).to(DEVICE)
flow_model, flow_train_losses = train_vla_model(
    flow_model,
    train_loader,
    FLOW_TRAIN_EPOCHS,
    run_name="flow_matching",
    checkpoint_name="flow_matching_vla.pt",
    curve_name="flow_matching_training_curve.png",
    action_mean=train_action_mean,
    action_std=train_action_std,
)

diffusion_model = DiffusionVLA(action_dim=ACTION_DIM, horizon=ACTION_HORIZON).to(DEVICE)
diffusion_model, diffusion_train_losses = train_vla_model(
    diffusion_model,
    train_loader,
    DIFFUSION_TRAIN_EPOCHS,
    run_name="diffusion",
    checkpoint_name="diffusion_vla.pt",
    curve_name="diffusion_training_curve.png",
    action_mean=train_action_mean,
    action_std=train_action_std,
)


# ═══════════════════════════════════════════════════════════════
# Cell 6: Offline Evaluators for All 3 Decoders
# ═══════════════════════════════════════════════════════════════

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
    def predict_action(self, image_np, instruction):
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
    def predict_action(self, image_np, instruction):
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
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base_model = OpenVLAModelClass.from_pretrained(self.base_model_name, **model_kwargs)
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.processor = AutoProcessor.from_pretrained(self.adapter_dir, trust_remote_code=True)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict_action(self, image_np, instruction):
        if self.model is None:
            self._load()

        prompt = format_vla_prompt(instruction)
        img = Image.fromarray(image_np)
        inputs = self.processor(images=[img], text=[prompt], return_tensors="pt")
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
            action = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN_VALUE],
                dtype=np.float32,
            )

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


def compute_prediction_metrics(pred_action, target_action):
    pred_action = ensure_franka_action_7d(pred_action, "prediction")
    target_action = ensure_franka_action_7d(target_action, "target")
    diff = pred_action - target_action
    translation_mae_cm = float(np.mean(np.abs(diff[:3]) * TRANSLATION_STEP_M * 100.0))
    rotation_mae_deg = float(np.mean(np.abs(diff[3:6]) * ROTATION_STEP_RAD * 180.0 / np.pi))
    gripper_accuracy = float((pred_action[6] > 0) == (target_action[6] > 0))
    normalized_l1 = float(np.mean(np.abs(diff)))
    return {
        "translation_mae_cm": translation_mae_cm,
        "rotation_mae_deg": rotation_mae_deg,
        "gripper_accuracy": gripper_accuracy,
        "normalized_l1": normalized_l1,
    }


class OfflineRealDataEvaluator:
    """Evaluate a policy on held-out real robot frames without environment rollouts."""

    def __init__(self, model, records):
        self.model = model
        self.records = records

    def evaluate(self, max_samples=None, verbose=True):
        subset = self.records[:max_samples] if max_samples is not None else self.records
        latencies = []
        translation_errors = []
        rotation_errors = []
        normalized_l1s = []
        gripper_matches = []
        parse_failures = []
        examples = []

        for index, record in enumerate(subset):
            pred_action, info = self.model.predict_action(record["image"], record["instruction"])
            metrics = compute_prediction_metrics(pred_action, record["action_7d"])
            latencies.append(float(info.get("inference_time_ms", 0.0)))
            translation_errors.append(metrics["translation_mae_cm"])
            rotation_errors.append(metrics["rotation_mae_deg"])
            normalized_l1s.append(metrics["normalized_l1"])
            gripper_matches.append(metrics["gripper_accuracy"])
            parse_failures.append(float(info.get("parse_failed", False)))

            example = {
                "sample_id": f"{record['episode_index']}-{record['frame_index']}",
                "episode_index": int(record["episode_index"]),
                "frame_index": int(record["frame_index"]),
                "instruction": record["instruction"],
                "target_action": record["action_7d"].tolist(),
                "pred_action": pred_action.tolist(),
                "metrics": metrics,
            }
            if "generated_text" in info:
                example["generated_text"] = info["generated_text"]
            examples.append(example)

            if verbose and (index + 1) % 25 == 0:
                print(
                    f"  Evaluated {index+1}/{len(subset)} samples | "
                    f"translation MAE={np.mean(translation_errors):.2f} cm | "
                    f"gripper acc={np.mean(gripper_matches):.1%}"
                )

        summary = {
            "num_examples": int(len(subset)),
            "translation_mae_cm": float(np.mean(translation_errors)),
            "rotation_mae_deg": float(np.mean(rotation_errors)),
            "normalized_l1": float(np.mean(normalized_l1s)),
            "gripper_accuracy": float(np.mean(gripper_matches)),
            "parse_fail_rate": float(np.mean(parse_failures)),
            "avg_inference_ms": float(np.mean(latencies)),
            "p50_inference_ms": float(np.percentile(latencies, 50)),
            "p95_inference_ms": float(np.percentile(latencies, 95)),
        }
        return {"summary": summary, "examples": examples}


def evaluate_policy_offline(name, policy, records):
    save_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(save_dir, exist_ok=True)
    evaluator = OfflineRealDataEvaluator(model=policy, records=records)
    result = evaluator.evaluate(max_samples=MAX_EVAL_SAMPLES, verbose=True)
    save_json(os.path.join(save_dir, "offline_eval.json"), result)
    return result


# ═══════════════════════════════════════════════════════════════
# Cell 7: Offline Real-Data Comparison Across All 3 Methods
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Offline Real-Data Evaluation: Autoregressive vs Diffusion vs Flow-Matching")
print("=" * 60)

if OPENVLA_MODEL_DIR is None:
    raise FileNotFoundError(
        "Could not find Notebook 2 output. Expected one of:\n- "
        + "\n- ".join(OPENVLA_CANDIDATE_DIRS)
    )

comparison = {}

flow_policy = FlowMatchingPolicyWrapper(flow_model)
comparison["flow_matching"] = evaluate_policy_offline("flow_matching", flow_policy, eval_records)
flow_model.to("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

diffusion_policy = DiffusionPolicyWrapper(diffusion_model)
comparison["diffusion"] = evaluate_policy_offline("diffusion", diffusion_policy, eval_records)
diffusion_model.to("cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

openvla_policy = OpenVLAPolicyWrapper(OPENVLA_MODEL_DIR)
comparison["autoregressive"] = evaluate_policy_offline("autoregressive", openvla_policy, eval_records)
openvla_policy.close()


# ═══════════════════════════════════════════════════════════════
# Cell 8: Save Offline Comparison Summary + Plots
# ═══════════════════════════════════════════════════════════════

ordered_names = ["autoregressive", "diffusion", "flow_matching"]
display_names = {
    "autoregressive": "Autoregressive",
    "diffusion": "Diffusion",
    "flow_matching": "Flow-Matching",
}

comparison_summary = {
    name: {
        "translation_mae_cm": float(result["summary"]["translation_mae_cm"]),
        "rotation_mae_deg": float(result["summary"]["rotation_mae_deg"]),
        "normalized_l1": float(result["summary"]["normalized_l1"]),
        "gripper_accuracy": float(result["summary"]["gripper_accuracy"]),
        "parse_fail_rate": float(result["summary"]["parse_fail_rate"]),
        "avg_inference_ms": float(result["summary"]["avg_inference_ms"]),
        "p50_inference_ms": float(result["summary"]["p50_inference_ms"]),
        "p95_inference_ms": float(result["summary"]["p95_inference_ms"]),
        "num_examples": int(result["summary"]["num_examples"]),
    }
    for name, result in comparison.items()
}
save_json(os.path.join(OUTPUT_DIR, "real_offline_summary.json"), comparison_summary)

print("\n" + "=" * 78)
print("OFFLINE REAL-DATA COMPARISON TABLE")
print("=" * 78)
print(f"{'Decoder':<20} {'Trans(cm)':>10} {'Rot(deg)':>10} {'Grip Acc':>10} {'P50(ms)':>10}")
print("-" * 70)
for name in ordered_names:
    summary = comparison_summary[name]
    print(
        f"{display_names[name]:<20} {summary['translation_mae_cm']:>10.2f} "
        f"{summary['rotation_mae_deg']:>10.2f} {summary['gripper_accuracy']:>9.0%} "
        f"{summary['p50_inference_ms']:>10.1f}"
    )
print("=" * 78)

with open(os.path.join(OUTPUT_DIR, "real_offline_table.md"), "w") as f:
    f.write("| Decoder | Translation MAE (cm) | Rotation MAE (deg) | Gripper Accuracy | P50 Latency |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for name in ordered_names:
        summary = comparison_summary[name]
        f.write(
            f"| {display_names[name]} | {summary['translation_mae_cm']:.2f} | "
            f"{summary['rotation_mae_deg']:.2f} | {summary['gripper_accuracy']:.1%} | "
            f"{summary['p50_inference_ms']:.1f} ms |\n"
        )

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
bar_colors = ["#4e79a7", "#f28e2b", "#59a14f"]
name_labels = [display_names[n] for n in ordered_names]

axes[0, 0].bar(name_labels, [comparison_summary[n]["translation_mae_cm"] for n in ordered_names], color=bar_colors)
axes[0, 0].set_ylabel("Translation MAE (cm)")
axes[0, 0].set_title("Held-Out Real Data", fontweight="bold")

axes[0, 1].bar(name_labels, [comparison_summary[n]["rotation_mae_deg"] for n in ordered_names], color=bar_colors)
axes[0, 1].set_ylabel("Rotation MAE (deg)")
axes[0, 1].set_title("Held-Out Real Data", fontweight="bold")

axes[1, 0].bar(name_labels, [comparison_summary[n]["gripper_accuracy"] for n in ordered_names], color=bar_colors)
axes[1, 0].set_ylim(0, 1.0)
axes[1, 0].set_ylabel("Gripper Accuracy")
axes[1, 0].set_title("Binary Open/Close Accuracy", fontweight="bold")

axes[1, 1].bar(name_labels, [comparison_summary[n]["p50_inference_ms"] for n in ordered_names], color=bar_colors)
axes[1, 1].set_ylabel("P50 Inference Latency (ms)")
axes[1, 1].set_title("Inference Latency", fontweight="bold")

for ax in axes.ravel():
    ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "real_offline_metrics.png"), dpi=150)
plt.close()


def render_qualitative_examples(eval_records, comparison, output_path):
    selected = eval_records[: min(QUALITATIVE_EXAMPLES, len(eval_records))]
    row_height = 220
    canvas_width = 1200
    canvas = Image.new("RGB", (canvas_width, row_height * len(selected)), color="white")

    prediction_lookup = {
        name: {example["sample_id"]: example for example in result["examples"]}
        for name, result in comparison.items()
    }

    for row, record in enumerate(selected):
        sample_id = f"{record['episode_index']}-{record['frame_index']}"
        img = Image.fromarray(record["image"]).resize((320, 180))
        panel = Image.new("RGB", (canvas_width, row_height), color="white")
        panel.paste(img, (20, 20))
        draw = ImageDraw.Draw(panel)

        lines = [
            f"Example {row+1} | episode={record['episode_index']} frame={record['frame_index']}",
            f"Instruction: {record['instruction']}",
            f"GT: {record['action_7d'].tolist()}",
        ]
        for name in ordered_names:
            example = prediction_lookup[name][sample_id]
            lines.append(f"{display_names[name]}: {example['pred_action']}")
            lines.append(
                f"  trans={example['metrics']['translation_mae_cm']:.2f}cm "
                f"rot={example['metrics']['rotation_mae_deg']:.2f}deg "
                f"grip={'✓' if example['metrics']['gripper_accuracy'] else '✗'}"
            )
        draw.multiline_text((370, 20), "\n".join(lines), fill="black", spacing=5)
        canvas.paste(panel, (0, row * row_height))

    canvas.save(output_path)


render_qualitative_examples(
    eval_records[:MAX_EVAL_SAMPLES],
    comparison,
    os.path.join(OUTPUT_DIR, "real_offline_examples.png"),
)

combined_predictions = []
per_model_examples = {
    name: {example["sample_id"]: example for example in result["examples"]}
    for name, result in comparison.items()
}
for record in eval_records[:MAX_EVAL_SAMPLES]:
    sample_id = f"{record['episode_index']}-{record['frame_index']}"
    combined_predictions.append({
        "sample_id": sample_id,
        "instruction": record["instruction"],
        "target_action": record["action_7d"].tolist(),
        "autoregressive": per_model_examples["autoregressive"][sample_id],
        "diffusion": per_model_examples["diffusion"][sample_id],
        "flow_matching": per_model_examples["flow_matching"][sample_id],
    })

with open(os.path.join(OUTPUT_DIR, "real_offline_predictions.jsonl"), "w") as f:
    for row in combined_predictions:
        f.write(json.dumps(row) + "\n")


# ═══════════════════════════════════════════════════════════════
# Cell 9: Generate Technical Report from Real Offline Results
# ═══════════════════════════════════════════════════════════════

best_translation_name = min(ordered_names, key=lambda name: comparison_summary[name]["translation_mae_cm"])
best_gripper_name = max(ordered_names, key=lambda name: comparison_summary[name]["gripper_accuracy"])
fastest_name = min(ordered_names, key=lambda name: comparison_summary[name]["p50_inference_ms"])

report_lines = [
    "# Multi-Paradigm VLA for Robotic Grasping — Offline Real-Data Report",
    "",
    "## 1. Overview",
    "This run compares three VLA action decoders on held-out real DROID robot frames:",
    "- Autoregressive: fine-tuned OpenVLA from Notebook 2",
    "- Diffusion: lightweight ViT+BERT+DiffusionHead baseline",
    "- Flow-Matching: lightweight ViT+BERT+FlowMatchingHead baseline",
    "",
    "## 2. Dataset",
    f"- Source: {ACTIVE_DROID_REPO}",
    f"- Total streamed frames: {len(all_droid_records)}",
    f"- Train frames: {len(train_records)}",
    f"- Eval frames: {min(len(eval_records), MAX_EVAL_SAMPLES)}",
    "- Robot platform: real Franka Panda data from DROID",
    "- Evaluation mode: offline one-step action prediction on held-out frames",
    "",
    "## 3. Metrics",
    "- Translation MAE (cm): average absolute XYZ delta error after converting to this repo's control interface",
    "- Rotation MAE (deg): average absolute roll/pitch/yaw delta error",
    "- Gripper Accuracy: binary open/close agreement",
    "- P50 Latency: median inference latency per frame",
    "",
    "## 4. Results",
]
for name in ordered_names:
    summary = comparison_summary[name]
    report_lines.extend([
        f"### {display_names[name]}",
        f"- Translation MAE: {summary['translation_mae_cm']:.2f} cm",
        f"- Rotation MAE: {summary['rotation_mae_deg']:.2f} deg",
        f"- Gripper Accuracy: {summary['gripper_accuracy']:.1%}",
        f"- Parse Fail Rate: {summary['parse_fail_rate']:.1%}",
        f"- P50 Inference Latency: {summary['p50_inference_ms']:.1f} ms",
        "",
    ])

report_lines.extend([
    "## 5. Summary",
    f"- Best translation error: {display_names[best_translation_name]} ({comparison_summary[best_translation_name]['translation_mae_cm']:.2f} cm)",
    f"- Best gripper accuracy: {display_names[best_gripper_name]} ({comparison_summary[best_gripper_name]['gripper_accuracy']:.1%})",
    f"- Fastest inference: {display_names[fastest_name]} ({comparison_summary[fastest_name]['p50_inference_ms']:.1f} ms p50)",
    "",
    "## 6. Output Files",
    "- flow_matching_vla.pt",
    "- diffusion_vla.pt",
    "- training_curve.png",
    "- diffusion_training_curve.png",
    "- real_offline_summary.json",
    "- real_offline_table.md",
    "- real_offline_metrics.png",
    "- real_offline_examples.png",
    "- real_offline_predictions.jsonl",
    "- technical_report.md",
])

with open(os.path.join(OUTPUT_DIR, "technical_report.md"), "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\n✅ All outputs saved to {OUTPUT_DIR}")
print(f"   📊 training_curve.png")
print(f"   📊 diffusion_training_curve.png")
print(f"   📊 real_offline_metrics.png")
print(f"   🖼️ real_offline_examples.png")
print(f"   📄 real_offline_summary.json")
print(f"   📄 real_offline_table.md")
print(f"   📄 real_offline_predictions.jsonl")
print(f"   📄 technical_report.md")
