#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Kaggle Notebook 2: OpenVLA QLoRA Fine-Tuning on T4 GPU
═══════════════════════════════════════════════════════════════════

Prerequisites:
  - Kaggle T4 GPU accelerator enabled
  - Demo data from Notebook 1 uploaded as Kaggle Dataset

This notebook:
1. Installs the OpenVLA-compatible training stack on Kaggle
2. Builds a mixed MuJoCo + DROID training dataset with shared Franka semantics
3. Fine-tunes OpenVLA-7B in 4-bit with LoRA adapters
4. Saves the adapter weights plus action-format metadata

⏱️ Estimated time: 1-3 hours with the Kaggle-fast preset
💾 GPU memory: ~14 GB (fits T4 16GB)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import json
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
        "av>=12.0.0",
        "opencv-python-headless>=4.9.0",
        "imageio>=2.30.0",
        "imageio-ffmpeg>=0.4.9",
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    load_droid_task_lookup,
    load_droid_info,
    sample_get,
)

USE_DROID = True                                # Mix in real DROID robot data from HF
USE_MUJOCO_DEMOS = os.environ.get("VLA_USE_MUJOCO_DEMOS", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
DEMO_DIR = os.environ.get("VLA_DEMO_DIR", "/kaggle/input/vla-demos/demos")
DROID_DATASET_REPO_CANDIDATES = [
    repo for repo in [
        os.environ.get("DROID_DATASET_REPO", "").strip() or None,
        "cadene/droid_1.0.1_v30",
    ]
    if repo
]
DROID_SPLIT = "train"
DROID_MAX_SAMPLES = 500                         # lower the real-data mix for low-disk Kaggle runs
DROID_FPS = DROID_DEFAULT_FPS                   # keep one shared control-rate assumption
OUTPUT_DIR = "/kaggle/working/openvla-finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──── Training Config ────
MODEL_NAME = "openvla/openvla-7b"
LORA_RANK = 32
LORA_ALPHA = 64
BATCH_SIZE = 2                    # T4 16GB constraint
GRAD_ACCUM_STEPS = 8              # effective batch = 2 × 8 = 16
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1
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
print(f"   MuJoCo demos: {'enabled' if USE_MUJOCO_DEMOS else 'disabled'}")
print(f"   Demo dir hint: {DEMO_DIR}")
print(
    f"   DROID: {'enabled' if USE_DROID else 'disabled'}"
    + (f" ({DROID_DATASET_REPO_CANDIDATES}, max {DROID_MAX_SAMPLES} samples)" if USE_DROID else "")
)
if USE_DROID:
    print(f"   DROID control rate assumption: {DROID_FPS:g} Hz")

# ──── Shared Franka Delta-Pose Action Semantics ────


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
    if not USE_MUJOCO_DEMOS:
        return None

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

    def __init__(self, demo_dir, image_size=224, augment=True, use_droid=False, use_mujoco_demos=True):
        self.image_size = image_size
        self.augment = augment
        self.source_counts = {}
        self.use_mujoco_demos = use_mujoco_demos
        self.demo_dir = resolve_demo_dir(demo_dir) if use_mujoco_demos else None

        # Load all self-collected demos (MuJoCo)
        self.samples = []
        if not self.use_mujoco_demos:
            print("  MuJoCo demos disabled; training with DROID-only data.")
        elif self.demo_dir is not None:
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

        # Load real DROID dataset from HuggingFace
        if use_droid and DROID_MAX_SAMPLES > 0:
            try:
                droid_count = 0
                droid_repo = None
                skip_stats = {}

                for candidate in DROID_DATASET_REPO_CANDIDATES:
                    print(
                        f"  Streaming up to {DROID_MAX_SAMPLES} real DROID samples "
                        f"from {candidate}..."
                    )
                    task_lookup = load_droid_task_lookup(candidate)
                    droid_info = load_droid_info(candidate)
                    candidate_skip_stats = {
                        "unsuccessful": 0,
                        "missing_image": 0,
                        "missing_instruction": 0,
                        "missing_action": 0,
                        "bad_image": 0,
                        "bad_action": 0,
                    }

                    max_raw_droid_frames = max(DROID_MAX_SAMPLES * 8, 2000)
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
                            if idx >= max_raw_droid_frames and droid_count == 0:
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
                            if idx >= max_raw_droid_frames and droid_count == 0:
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
                                action_7d = droid_action_to_franka_action(
                                    raw_action,
                                    source_name=source_name,
                                )
                        except Exception:
                            candidate_skip_stats["bad_action"] += 1
                            continue

                        try:
                            image_arr = image_to_uint8_array(image, source_name)
                        except Exception:
                            candidate_skip_stats["bad_image"] += 1
                            continue

                        self.samples.append({
                            "image": image_arr,
                            "instruction": str(instruction),
                            "action_7d": action_7d,
                            "source": "droid",
                        })
                        droid_count += 1
                        droid_repo = candidate
                        if droid_count >= DROID_MAX_SAMPLES:
                            break

                    skip_stats[candidate] = candidate_skip_stats
                    if droid_count > 0:
                        break

                self.source_counts["droid"] = droid_count
                if droid_count > 0:
                    print(
                        f"  Loaded {droid_count} real DROID samples "
                        f"from {droid_repo}/{DROID_SPLIT} at {droid_info.get('fps', DROID_FPS)} Hz"
                    )
                else:
                    print("  Loaded 0 real DROID samples from all candidate repos.")
                print(f"  DROID skip stats: {skip_stats}")
            except Exception as exc:
                print(f"  ⚠️ Failed to load DROID dataset: {type(exc).__name__}: {exc}")

        if len(self.samples) == 0:
            guidance = (
                "Enable DROID loading, or attach Notebook 1 demos as a Kaggle Dataset, "
                "or set VLA_DEMO_DIR to a directory containing demo_*.npz files."
                if self.use_mujoco_demos
                else "Pure-DROID mode is enabled, so at least one real DROID sample must load."
            )
            raise RuntimeError(
                "Could not find any training samples. "
                f"Tried DEMO_DIR={demo_dir!r}, resolved demo dir={self.demo_dir!r}, "
                f"MuJoCo demos={'enabled' if self.use_mujoco_demos else 'disabled'}, "
                f"and DROID={'enabled' if use_droid else 'disabled'}. "
                f"{guidance}"
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
dataset = VLADemoDataset(
    DEMO_DIR,
    image_size=IMAGE_SIZE,
    use_droid=USE_DROID,
    use_mujoco_demos=USE_MUJOCO_DEMOS,
)
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
model_kwargs = {
    "quantization_config": bnb_config,
    "torch_dtype": torch.float16,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}
if torch.cuda.is_available():
    # Avoid automatic CPU offload; partial CPU placement makes 7B training extremely slow.
    model_kwargs["device_map"] = {"": 0}
else:
    model_kwargs["device_map"] = "cpu"

model = OpenVLAModelClass.from_pretrained(MODEL_NAME, **model_kwargs)

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print(f"✅ Model loaded on {next(model.parameters()).device}")
if torch.cuda.is_available():
    print(f"   Memory allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"   Memory reserved: {torch.cuda.memory_reserved()/1e9:.1f} GB")

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
