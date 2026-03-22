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
13. Creates a custom PyTorch Dataset from collected demos + LIBERO
14. Fine-tunes using LoRA adapters (rank=32, 7 projection layers)
15. Saves fine-tuned adapter weights

⏱️ Estimated time: 1-2 hours
💾 GPU memory: ~14 GB (fits T4 16GB)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import json
import io
import re
import subprocess
import sys
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
            f"Pin numpy=={NUMPY_VERSION} before loading OpenVLA."
        ) from exc
    print(f"✅ Verified torch↔numpy bridge ({output})")


def install():
    pkgs = [
        # Official OpenVLA-compatible stack
        "torch==2.2.0",
        "torchvision==0.17.0",
        "transformers==4.40.1",
        "tokenizers==0.19.1",
        "accelerate==0.30.1",
        "peft==0.11.1",
        "bitsandbytes==0.43.1",
        "timm==0.9.10",
        f"numpy=={NUMPY_VERSION}",
        "wandb",
        "datasets",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade"] + pkgs)
    verify_torch_numpy_bridge()
    print("✅ Official OpenVLA dependencies installed")

install()

# ═══════════════════════════════════════════════════════════════
# Cell 2: Configuration
# ═══════════════════════════════════════════════════════════════

import os
import torch

# ──── Paths & Data Sources ────
PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
USE_LIBERO = True                               # Mix in real LIBERO samples from HF
DEMO_DIR = os.environ.get("VLA_DEMO_DIR", "/kaggle/input/vla-demos/demos")
LIBERO_DATASET_REPO = "physical-intelligence/libero"
LIBERO_SPLIT = "train"
LIBERO_MAX_SAMPLES = 5000                       # cap streaming download to keep Kaggle practical
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
LOG_STEPS = 10

print(f"✅ Config ready")
print(f"   Model: {MODEL_NAME}")
print(f"   LoRA rank: {LORA_RANK}, effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"   Log every {LOG_STEPS} optimizer steps")
print(f"   Demo dir hint: {DEMO_DIR}")
print(
    f"   LIBERO: {'enabled' if USE_LIBERO else 'disabled'}"
    + (f" ({LIBERO_DATASET_REPO}, max {LIBERO_MAX_SAMPLES} samples)" if USE_LIBERO else "")
)

# ──── Franka Delta-Pose Action Semantics ────
FRANKA_ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
TRANSLATION_STEP_M = 0.03
ROTATION_STEP_RAD = 0.05
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSE_VALUE = 1.0
ACTION_MIN = -1.0
ACTION_MAX = 1.0


def ensure_franka_action_7d(action, source_name="<unknown>"):
    """Convert demo actions to the normalized 7-DOF Franka delta-pose format."""
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


def franka_action_to_physical_delta(action):
    """Map a normalized action to the actual per-step delta applied in the env."""
    action = ensure_franka_action_7d(action)
    return {
        "dx_m": float(action[0] * TRANSLATION_STEP_M),
        "dy_m": float(action[1] * TRANSLATION_STEP_M),
        "dz_m": float(action[2] * TRANSLATION_STEP_M),
        "dax_rad": float(action[3] * ROTATION_STEP_RAD),
        "day_rad": float(action[4] * ROTATION_STEP_RAD),
        "daz_rad": float(action[5] * ROTATION_STEP_RAD),
        "gripper_cmd": "close" if action[6] > 0 else "open",
    }


def format_franka_action(action):
    """
    Serialize a normalized Franka delta-pose action as a compact supervision target.

    The first 6 dimensions remain normalized to [-1, 1] to match the demos.
    The gripper is emitted as open/close because the environment uses its sign only.
    """
    action = ensure_franka_action_7d(action)
    gripper_cmd = "close" if action[6] > 0 else "open"
    values = [f"{key}={action[idx]:+.3f}" for idx, key in enumerate(FRANKA_ACTION_KEYS[:-1])]
    values.append(f"gripper={gripper_cmd}")
    return " ".join(values)


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
    """Parse the generated textual action back into the env's normalized 7-DOF control."""
    match = FRANKA_ACTION_PATTERN.search(text)
    if match is None:
        return None
    values = [float(match.group(key)) for key in FRANKA_ACTION_KEYS[:-1]]
    values.append(GRIPPER_CLOSE_VALUE if match.group("gripper") == "close" else GRIPPER_OPEN_VALUE)
    return ensure_franka_action_7d(values)


def format_physical_delta(action):
    """Human-readable physical interpretation of the normalized action."""
    delta = franka_action_to_physical_delta(action)
    return (
        f"xyz=({delta['dx_m']:+.4f}, {delta['dy_m']:+.4f}, {delta['dz_m']:+.4f}) m/step | "
        f"rpy=({delta['dax_rad']:+.4f}, {delta['day_rad']:+.4f}, {delta['daz_rad']:+.4f}) rad/step | "
        f"gripper={delta['gripper_cmd']}"
    )


def format_vla_prompt(instruction):
    """Describe the exact Franka end-effector control interface expected from the model."""
    return (
        f"In: What normalized Franka Panda delta-pose action should the robot take to {instruction}?\n"
        "Out: Return dx dy dz dax day daz in [-1, 1] and gripper=open|close.\n"
        f"dx dy dz are Cartesian deltas scaled by {TRANSLATION_STEP_M:.3f} m/step. "
        f"dax day daz are angular deltas scaled by {ROTATION_STEP_RAD:.2f} rad/step. "
        "Use gripper=close for positive commands and gripper=open for negative commands.\n"
        "Action:"
    )


def build_supervised_batch(processor, images, prompts, actions, device, max_length):
    """
    Build causal-LM supervision with prompt tokens masked out.

    We serialize the Franka action explicitly so the loss is tied to the 7-DOF
    delta-pose command instead of only reconstructing the prompt text.
    """
    target_texts = [format_franka_action(action) for action in actions.cpu().numpy()]
    full_texts = [f"{prompt} {target}" for prompt, target in zip(prompts, target_texts)]

    prompt_inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    full_inputs = processor(
        images=images,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
    labels = full_inputs["input_ids"].clone()
    labels[full_inputs["attention_mask"] == 0] = -100
    for i, prompt_len in enumerate(prompt_lengths.tolist()):
        labels[i, :prompt_len] = -100

    full_inputs = full_inputs.to(device)
    full_inputs["labels"] = labels.to(device)
    return full_inputs, target_texts


def save_franka_action_metadata(save_dir):
    """Persist the action semantics alongside the fine-tuned adapter."""
    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        "control_mode": "franka_delta_pose",
        "action_order": list(FRANKA_ACTION_KEYS),
        "normalized_range": [ACTION_MIN, ACTION_MAX],
        "translation_step_m": TRANSLATION_STEP_M,
        "rotation_step_rad": ROTATION_STEP_RAD,
        "gripper_semantics": {
            "negative": "open",
            "positive": "close",
        },
        "target_format": "dx=... dy=... dz=... dax=... day=... daz=... gripper=open|close",
    }
    with open(os.path.join(save_dir, "franka_action_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

# ═══════════════════════════════════════════════════════════════
# Cell 3: Custom Dataset
# ═══════════════════════════════════════════════════════════════

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob


def resolve_demo_dir(preferred_dir):
    """Find the first directory that actually contains demo_*.npz files."""
    candidates = []
    seen = set()

    def add_candidate(path_like):
        if not path_like:
            return
        path = os.path.abspath(os.fspath(path_like))
        if path not in seen:
            candidates.append(path)
            seen.add(path)

    add_candidate(preferred_dir)
    add_candidate(os.environ.get("VLA_DEMO_DIR"))
    add_candidate("/kaggle/input/vla-demos/demos")
    add_candidate("/kaggle/input/vla-demos")
    add_candidate("/kaggle/working/demos")
    add_candidate(PROJECT_ROOT / "data" / "demos")

    for pattern in ("/kaggle/input/*/demos", "/kaggle/input/*"):
        for match in sorted(glob.glob(pattern)):
            add_candidate(match)

    for candidate in candidates:
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "demo_*.npz")):
            return candidate
    return None


def image_to_uint8_array(image, source_name):
    """Normalize PIL / HF image feature outputs to HWC uint8 arrays."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            arr = np.array(Image.open(io.BytesIO(image["bytes"])).convert("RGB"), dtype=np.uint8)
        elif image.get("path"):
            arr = np.array(Image.open(image["path"]).convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported image payload in {source_name}: {image.keys()}")
    else:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dims in {source_name}, got shape {arr.shape}")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.size and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def sample_get(sample, *paths):
    """Retrieve a possibly nested value from a HF sample dict."""
    for path in paths:
        value = sample
        found = True
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                found = False
                break
        if found and value is not None:
            return value
    return None


def load_libero_task_lookup(repo_id):
    """Map LIBERO task indices to natural-language instructions."""
    from huggingface_hub import hf_hub_download

    tasks_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="meta/tasks.jsonl",
    )
    task_lookup = {}
    with open(tasks_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                task_lookup[int(record["task_index"])] = record["task"]
    return task_lookup


class VLADemoDataset(Dataset):
    """
    PyTorch Dataset for VLA fine-tuning from collected .npz demos.

    Each sample returns:
    - image: PIL Image (will be processed by VLA processor)
    - instruction: str
    - action: (7,) float tensor for Franka end-effector control

    Franka demos store 7-DOF actions:
    [dx, dy, dz, dax, day, daz, gripper]

    Legacy 4-DOF demos are still supported and get padded to 7-DOF:
    [dx, dy, dz, 0, 0, 0, gripper]
    """

    def __init__(self, demo_dir, image_size=224, augment=True, use_libero=False):
        self.image_size = image_size
        self.augment = augment
        self.source_counts = {}
        self.demo_dir = resolve_demo_dir(demo_dir)

        # Load all self-collected demos (MuJoCo)
        self.samples = []
        if self.demo_dir is not None:
            demo_files = sorted(glob.glob(os.path.join(self.demo_dir, "demo_*.npz")))
            for f in demo_files:
                data = np.load(f, allow_pickle=True)
                if not data.get("success", False):
                    continue  # only use successful demos

                images = data["images"]
                actions = data["actions"]
                instructions = data["instructions"]

                for t in range(len(actions)):
                    action_7d = ensure_franka_action_7d(actions[t], source_name=f)
                    self.samples.append({
                        "image": images[t],               # (H, W, 3) uint8
                        "instruction": str(instructions[t]),  # str
                        "action_7d": action_7d,           # (7,) float
                        "source": "mujoco",
                    })
            self.source_counts["mujoco"] = len(self.samples)
            print(
                f"  Loaded {self.source_counts['mujoco']} samples from "
                f"{len(demo_files)} self-collected demos in {self.demo_dir}"
            )
        else:
            print("  No self-collected demos found under Kaggle input/working directories.")

        # Load real LIBERO dataset from HuggingFace
        if use_libero and LIBERO_MAX_SAMPLES > 0:
            print(
                f"  Streaming up to {LIBERO_MAX_SAMPLES} real LIBERO samples "
                f"from {LIBERO_DATASET_REPO}..."
            )
            try:
                from datasets import load_dataset
                task_lookup = load_libero_task_lookup(LIBERO_DATASET_REPO)
                libero_ds = load_dataset(
                    LIBERO_DATASET_REPO,
                    split=LIBERO_SPLIT,
                    streaming=True,
                )

                libero_count = 0
                for idx, sample in enumerate(libero_ds):
                    image = sample_get(
                        sample,
                        "image",
                        "observation.images.image",
                        "observation.image",
                    )
                    action = sample_get(sample, "actions", "action")
                    instruction = sample_get(
                        sample,
                        "task",
                        "instruction",
                        "language_instruction",
                        "text",
                    )
                    if instruction is None:
                        task_index = sample_get(sample, "task_index")
                        if task_index is not None:
                            instruction = task_lookup.get(int(task_index))

                    if image is None or action is None or instruction is None:
                        continue

                    action_7d = ensure_franka_action_7d(
                        action,
                        source_name=f"{LIBERO_DATASET_REPO}:{idx}",
                    )
                    self.samples.append({
                        "image": image_to_uint8_array(image, f"{LIBERO_DATASET_REPO}:{idx}"),
                        "instruction": str(instruction),
                        "action_7d": action_7d,
                        "source": "libero",
                    })
                    libero_count += 1
                    if libero_count >= LIBERO_MAX_SAMPLES:
                        break

                self.source_counts["libero"] = libero_count
                print(
                    f"  Loaded {libero_count} real LIBERO samples "
                    f"from {LIBERO_DATASET_REPO}/{LIBERO_SPLIT}"
                )
            except Exception as exc:
                print(f"  ⚠️ Failed to load LIBERO dataset: {type(exc).__name__}: {exc}")

        if len(self.samples) == 0:
            raise RuntimeError(
                "Could not find any training samples. "
                f"Tried DEMO_DIR={demo_dir!r}, resolved demo dir={self.demo_dir!r}, "
                f"and LIBERO={'enabled' if use_libero else 'disabled'}. "
                "Attach the Notebook 1 demos as a Kaggle Dataset or set VLA_DEMO_DIR "
                "to a directory containing demo_*.npz files."
            )

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

        return {
            "image": img,
            "instruction": sample["instruction"],
            "action": torch.tensor(sample["action_7d"], dtype=torch.float32),
        }


def collate_vla_batch(batch):
    """Keep PIL images as a Python list; default_collate cannot stack them."""
    return {
        "image": [sample["image"] for sample in batch],
        "instruction": [sample["instruction"] for sample in batch],
        "action": torch.stack([sample["action"] for sample in batch]),
    }


# Test dataset
dataset = VLADemoDataset(DEMO_DIR, image_size=IMAGE_SIZE, use_libero=USE_LIBERO)
print(f"  Total Dataset size: {len(dataset)} samples")
print(f"  Source counts: {dataset.source_counts}")
sample = dataset[0]
print(f"  Sample image: {sample['image'].size}")
print(f"  Sample instruction: '{sample['instruction']}'")
print(f"  Sample action (normalized): {sample['action']}")
print(f"  Sample action (serialized): {format_franka_action(sample['action'].numpy())}")
print(f"  Sample action (physical): {format_physical_delta(sample['action'].numpy())}")

# ═══════════════════════════════════════════════════════════════
# Cell 4: Load OpenVLA with QLoRA (4-bit quantization)
# ═══════════════════════════════════════════════════════════════

from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import AutoModelForVision2Seq as OpenVLAModelClass
except ImportError:
    from transformers import AutoModelForImageTextToText as OpenVLAModelClass
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
model = OpenVLAModelClass.from_pretrained(
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
if hasattr(model, "config"):
    model.config.use_cache = False

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
    collate_fn=collate_vla_batch,
)

scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(dataloader))

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

        # Build prompt + structured Franka action targets with prompt masking
        inputs, target_texts = build_supervised_batch(
            processor,
            images,
            prompts,
            actions,
            model.device,
            MAX_SEQ_LEN,
        )

        # Forward pass on the structured delta-pose target
        outputs = model(**inputs)
        loss = outputs.loss / GRAD_ACCUM_STEPS

        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % LOG_STEPS == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Step {global_step} | "
                      f"Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
                print(f"    Target format: {target_texts[0]}")

            # Save checkpoint
            if global_step % SAVE_STEPS == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                save_franka_action_metadata(ckpt_path)
                print(f"  💾 Saved checkpoint: {ckpt_path}")

        epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"\n  Epoch {epoch+1}/{NUM_EPOCHS} complete | Avg loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(OUTPUT_DIR, "best")
        model.save_pretrained(best_path)
        save_franka_action_metadata(best_path)
        print(f"  ⭐ New best model saved: {best_path}")

# ═══════════════════════════════════════════════════════════════
# Cell 7: Save Final Model
# ═══════════════════════════════════════════════════════════════

final_path = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_path)
processor.save_pretrained(final_path)
save_franka_action_metadata(final_path)

print(f"\n{'='*60}")
print(f"✅ Training complete!")
print(f"   Best loss: {best_loss:.4f}")
print(f"   Model saved: {final_path}")
print(f"   Franka action spec: {os.path.join(final_path, 'franka_action_config.json')}")
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
        max_new_tokens=48,
        do_sample=False,
    )
    generated_text = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )[0]

parsed_action = parse_franka_action(generated_text)

print(f"\n🔎 Inference test:")
print(f"   Instruction: '{test_instruction}'")
print(f"   Generated text: {generated_text}")
print(f"   Ground-truth target: {format_franka_action(dataset[0]['action'].numpy())}")
if parsed_action is not None:
    print(f"   Parsed action (normalized): {parsed_action}")
    print(f"   Parsed action (physical): {format_physical_delta(parsed_action)}")
else:
    print("   Parsed action: <failed to match Franka delta-pose format>")
print(f"\n📋 Next: Run Notebook 3 for flow-matching training + evaluation")
