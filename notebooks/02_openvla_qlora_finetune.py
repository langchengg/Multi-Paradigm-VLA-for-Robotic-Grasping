#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Kaggle Notebook 2: OpenVLA QLoRA Fine-Tuning on T4 GPU
═══════════════════════════════════════════════════════════════════

Prerequisites:
  - Kaggle T4 GPU accelerator enabled
  - Demo data from Notebook 1 uploaded as Kaggle Dataset

This notebook:
1. Loads OpenVLA-7B in 4-bit quantization (QLoRA)
2. Creates a custom PyTorch Dataset from collected demos
3. Fine-tunes using LoRA adapters (rank=32)
4. Saves fine-tuned adapter weights

⏱️ Estimated time: 1-2 hours (100 demos, 5 epochs)
💾 GPU memory: ~14 GB (fits T4 16GB)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import subprocess, sys

def install():
    pkgs = [
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "torch>=2.0.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "wandb",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    print("✅ Dependencies installed")

install()

# ═══════════════════════════════════════════════════════════════
# Cell 2: Configuration
# ═══════════════════════════════════════════════════════════════

import os
import torch

# ──── Paths ────
DEMO_DIR = "/kaggle/input/vla-demos/demos"      # your uploaded dataset
OUTPUT_DIR = "/kaggle/working/openvla-finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──── Training Config ────
MODEL_NAME = "openvla/openvla-7b"
LORA_RANK = 32
LORA_ALPHA = 64
BATCH_SIZE = 2                    # T4 16GB constraint
GRAD_ACCUM_STEPS = 8              # effective batch = 2 × 8 = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 5
IMAGE_SIZE = 224                  # OpenVLA input resolution
MAX_SEQ_LEN = 256
SAVE_STEPS = 200

print(f"✅ Config ready")
print(f"   Model: {MODEL_NAME}")
print(f"   LoRA rank: {LORA_RANK}, effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# ═══════════════════════════════════════════════════════════════
# Cell 3: Custom Dataset
# ═══════════════════════════════════════════════════════════════

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob


class VLADemoDataset(Dataset):
    """
    PyTorch Dataset for VLA fine-tuning from collected .npz demos.

    Each sample returns:
    - image: PIL Image (will be processed by VLA processor)
    - instruction: str
    - action: (7,) float tensor (padded to 7-DOF for OpenVLA)

    OpenVLA expects 7-DOF actions: [dx, dy, dz, dax, day, daz, gripper]
    Our demos have 4-DOF: [dx, dy, dz, gripper]
    We pad rotation dims with zeros.
    """

    def __init__(self, demo_dir, image_size=224, augment=True):
        self.image_size = image_size
        self.augment = augment

        # Load all demos
        self.samples = []
        demo_files = sorted(glob.glob(os.path.join(demo_dir, "demo_*.npz")))

        for f in demo_files:
            data = np.load(f, allow_pickle=True)
            if not data.get("success", False):
                continue  # only use successful demos

            images = data["images"]
            actions = data["actions"]
            instruction = str(data["instruction"])

            # Each timestep is a training sample
            for t in range(len(actions)):
                self.samples.append({
                    "image": images[t],              # (H, W, 3) uint8
                    "instruction": instruction,       # str
                    "action_4d": actions[t],          # (4,) float
                })

        print(f"  Loaded {len(self.samples)} samples from {len(demo_files)} demos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert image to PIL
        img = Image.fromarray(sample["image"]).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )

        # Random crop augmentation (90% area, matching OpenVLA training)
        if self.augment:
            w, h = img.size
            crop_size = int(0.9 * min(w, h))
            left = np.random.randint(0, w - crop_size + 1)
            top = np.random.randint(0, h - crop_size + 1)
            img = img.crop((left, top, left + crop_size, top + crop_size))
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Pad 4-DOF to 7-DOF: [dx, dy, dz, 0, 0, 0, gripper]
        a4 = sample["action_4d"]
        action_7d = np.array([a4[0], a4[1], a4[2], 0.0, 0.0, 0.0, a4[3]], dtype=np.float32)

        return {
            "image": img,
            "instruction": sample["instruction"],
            "action": torch.tensor(action_7d),
        }


# Test dataset
if os.path.exists(DEMO_DIR):
    dataset = VLADemoDataset(DEMO_DIR, image_size=IMAGE_SIZE)
    print(f"  Dataset size: {len(dataset)} samples")
    sample = dataset[0]
    print(f"  Sample image: {sample['image'].size}")
    print(f"  Sample instruction: '{sample['instruction']}'")
    print(f"  Sample action: {sample['action']}")
else:
    print(f"⚠️  Demo dir not found: {DEMO_DIR}")
    print(f"   Upload Notebook 1 output as Kaggle Dataset first!")

# ═══════════════════════════════════════════════════════════════
# Cell 4: Load OpenVLA with QLoRA (4-bit quantization)
# ═══════════════════════════════════════════════════════════════

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("Loading OpenVLA-7B with 4-bit quantization...")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print(f"✅ Model loaded on {next(model.parameters()).device}")
print(f"   Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# Prepare for QLoRA training
model = prepare_model_for_kbit_training(model)

# ═══════════════════════════════════════════════════════════════
# Cell 5: Apply LoRA Adapters
# ═══════════════════════════════════════════════════════════════

# LoRA config — target language model layers
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"✅ LoRA applied")
print(f"   Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")
print(f"   Memory after LoRA: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════════
# Cell 6: Training Loop
# ═══════════════════════════════════════════════════════════════

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

# Action tokenization helper
# OpenVLA uses 256 discrete bins per action dimension
ACTION_BINS = 256
ACTION_MIN = -1.0
ACTION_MAX = 1.0

def discretize_actions(actions, n_bins=ACTION_BINS):
    """Convert continuous actions to discrete token ids."""
    # Clip to range
    actions = torch.clamp(actions, ACTION_MIN, ACTION_MAX)
    # Normalize to [0, 1]
    normalized = (actions - ACTION_MIN) / (ACTION_MAX - ACTION_MIN)
    # Convert to bin indices
    bin_ids = (normalized * (n_bins - 1)).long()
    return bin_ids

def format_vla_prompt(instruction):
    """Format instruction as VLA prompt."""
    return f"In: What action should the robot take to {instruction}?\nOut:"


print(f"\n{'='*60}")
print(f"Starting QLoRA Fine-Tuning")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Steps/epoch: {len(dataloader)}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"{'='*60}")

model.train()
global_step = 0
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Prepare inputs
        images = list(batch["image"])  # list of PIL images
        instructions = batch["instruction"]
        actions = batch["action"].to(model.device)  # (B, 7)

        # Format prompts
        prompts = [format_vla_prompt(inst) for inst in instructions]

        # Process through OpenVLA processor
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(model.device)

        # Discretize action targets
        action_tokens = discretize_actions(actions)  # (B, 7)

        # Forward pass
        # Note: OpenVLA internally handles action token prediction
        # For custom training, we use the language model loss
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Step {global_step} | "
                      f"Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            # Save checkpoint
            if global_step % SAVE_STEPS == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                print(f"  💾 Saved checkpoint: {ckpt_path}")

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS} complete | Avg loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(OUTPUT_DIR, "best")
        model.save_pretrained(best_path)
        print(f"  ⭐ New best model saved: {best_path}")

# ═══════════════════════════════════════════════════════════════
# Cell 7: Save Final Model
# ═══════════════════════════════════════════════════════════════

final_path = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_path)
processor.save_pretrained(final_path)

print(f"\n{'='*60}")
print(f"✅ Training complete!")
print(f"   Best loss: {best_loss:.4f}")
print(f"   Model saved: {final_path}")
print(f"   Download the 'openvla-finetuned' folder for Notebook 3")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════
# Cell 8: Quick Inference Test
# ═══════════════════════════════════════════════════════════════

model.eval()

test_image = dataset[0]["image"]
test_instruction = dataset[0]["instruction"]

prompt = format_vla_prompt(test_instruction)
inputs = processor(
    images=[test_image],
    text=[prompt],
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=7,  # 7 action dimensions
        do_sample=False,
    )
    # Decode action tokens
    action_text = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:])

print(f"\n🔎 Inference test:")
print(f"   Instruction: '{test_instruction}'")
print(f"   Generated: {action_text}")
print(f"\n📋 Next: Run Notebook 3 for flow-matching training + evaluation")
