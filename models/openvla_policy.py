"""
OpenVLA policy wrapper for the Franka MuJoCo environment.

This module is intentionally import-light. Transformer / PEFT dependencies are
loaded only when the policy first runs inference, so tests and local utilities
can use the action parser without a GPU stack.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


OPENVLA_BASE_MODEL = "openvla/openvla-7b"
FRANKA_ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
TRANSLATION_STEP_M = 0.03
ROTATION_STEP_RAD = 0.05
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSE_VALUE = 1.0
ACTION_MIN = -1.0
ACTION_MAX = 1.0
ACTION_BIN_SIZE = 0.05
ACTION_BIN_LIMIT = int(round(ACTION_MAX / ACTION_BIN_SIZE))
FRANKA_AXIS_TOKEN_PREFIXES = ("dx", "dy", "dz", "ax", "ay", "az")

FRANKA_ACTION_PATTERN = re.compile(
    r"(?<!\S)"
    r"(?P<dx>dx[pnz]\d{2})\s+"
    r"(?P<dy>dy[pnz]\d{2})\s+"
    r"(?P<dz>dz[pnz]\d{2})\s+"
    r"(?P<dax>ax[pnz]\d{2})\s+"
    r"(?P<day>ay[pnz]\d{2})\s+"
    r"(?P<daz>az[pnz]\d{2})\s+"
    r"(?P<gripper>g[oc])\b",
    re.IGNORECASE,
)
FRANKA_LEGACY_ACTION_PATTERN = re.compile(
    r"dx=(?P<dx>[+-]?\d+(?:\.\d+)?)\s+"
    r"dy=(?P<dy>[+-]?\d+(?:\.\d+)?)\s+"
    r"dz=(?P<dz>[+-]?\d+(?:\.\d+)?)\s+"
    r"dax=(?P<dax>[+-]?\d+(?:\.\d+)?)\s+"
    r"day=(?P<day>[+-]?\d+(?:\.\d+)?)\s+"
    r"daz=(?P<daz>[+-]?\d+(?:\.\d+)?)\s+"
    r"gripper=(?P<gripper>open|close)",
    re.IGNORECASE,
)
FRANKA_KEY_VALUE_PATTERN = re.compile(
    r"\b(?P<key>dx|dy|dz|dax|day|daz)\s*=\s*(?P<value>[+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
FRANKA_GRIPPER_PATTERN = re.compile(r"\bgripper\s*=\s*(open|close)\b", re.IGNORECASE)


def ensure_franka_action_7d(action, source_name="<unknown>") -> np.ndarray:
    """Convert normalized Franka actions to this repo's 7-DOF interface."""
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


def decode_franka_axis_token(token: str) -> float:
    """Decode one axis-scoped token like ``dxp06`` into normalized units."""
    token = token.strip().lower()
    if len(token) != 5:
        raise ValueError(f"Expected a 5-character axis token, found {token!r}")
    sign = token[2]
    magnitude = int(token[3:])
    if sign == "p":
        return magnitude * ACTION_BIN_SIZE
    if sign == "n":
        return -magnitude * ACTION_BIN_SIZE
    if sign == "z":
        return 0.0
    raise ValueError(f"Unsupported sign code in token {token!r}")


def format_franka_action(action) -> str:
    """Serialize a normalized 7-DOF Franka action as OpenVLA target tokens."""
    action = ensure_franka_action_7d(action)
    bins = np.clip(
        np.round(action[:6] / ACTION_BIN_SIZE),
        -ACTION_BIN_LIMIT,
        ACTION_BIN_LIMIT,
    ).astype(int)
    values = []
    for prefix, value in zip(FRANKA_AXIS_TOKEN_PREFIXES, bins.tolist()):
        sign = "p" if value > 0 else "n" if value < 0 else "z"
        values.append(f"{prefix}{sign}{abs(value):02d}")
    values.append("gc" if action[6] > 0 else "go")
    return " ".join(values)


def parse_franka_action(text: str) -> Optional[np.ndarray]:
    """Parse generated OpenVLA text into the repo's normalized 7-DOF action."""
    match = FRANKA_ACTION_PATTERN.search(text or "")
    if match is not None:
        values = [decode_franka_axis_token(match.group(key)) for key in FRANKA_ACTION_KEYS[:-1]]
        values.append(
            GRIPPER_CLOSE_VALUE if match.group("gripper").lower() == "gc" else GRIPPER_OPEN_VALUE
        )
        return ensure_franka_action_7d(values)

    match = FRANKA_LEGACY_ACTION_PATTERN.search(text or "")
    if match is not None:
        values = [float(match.group(key)) for key in FRANKA_ACTION_KEYS[:-1]]
        values.append(
            GRIPPER_CLOSE_VALUE
            if match.group("gripper").lower() == "close"
            else GRIPPER_OPEN_VALUE
        )
        return ensure_franka_action_7d(values)

    keyed_values = {}
    for keyed_match in FRANKA_KEY_VALUE_PATTERN.finditer(text or ""):
        keyed_values[keyed_match.group("key").lower()] = float(keyed_match.group("value"))
    if all(key in keyed_values for key in FRANKA_ACTION_KEYS[:-1]):
        gripper_match = FRANKA_GRIPPER_PATTERN.search(text or "")
        if gripper_match is not None:
            values = [keyed_values[key] for key in FRANKA_ACTION_KEYS[:-1]]
            values.append(
                GRIPPER_CLOSE_VALUE
                if gripper_match.group(1).lower() == "close"
                else GRIPPER_OPEN_VALUE
            )
            return ensure_franka_action_7d(values)

    numeric_values = [float(value) for value in re.findall(r"[+-]?\d+(?:\.\d+)?", text or "")]
    if len(numeric_values) >= 7:
        numeric_values[6] = GRIPPER_CLOSE_VALUE if numeric_values[6] > 0 else GRIPPER_OPEN_VALUE
        return ensure_franka_action_7d(numeric_values[:7])
    return None


def format_vla_prompt(instruction: str) -> str:
    """Prompt OpenVLA for this repo's exact Franka delta-pose interface."""
    return (
        f"Task: {instruction}\n"
        "Return 7 action tokens in order dx dy dz ax ay az grip. "
        "Use dimension-scoped bins like dxp06 dyn03 dzz00 axp02 ayn11 azz00. "
        f"Each bin is {ACTION_BIN_SIZE:.2f} normalized units in "
        f"[{ACTION_MIN:g}, {ACTION_MAX:g}]. "
        "Use gc for gripper=close and go for gripper=open.\n"
        "Action:"
    )


def format_physical_delta(action) -> str:
    """Return a compact physical interpretation of a normalized action."""
    action = ensure_franka_action_7d(action)
    xyz = action[:3] * TRANSLATION_STEP_M
    rpy = action[3:6] * ROTATION_STEP_RAD
    gripper = "close" if action[6] > 0 else "open"
    return (
        f"xyz=({xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}) m/step | "
        f"rpy=({rpy[0]:+.4f}, {rpy[1]:+.4f}, {rpy[2]:+.4f}) rad/step | "
        f"gripper={gripper}"
    )


def _looks_like_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def _checkpoint_number(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_adapter_dir(root) -> Optional[Path]:
    """
    Resolve a PEFT adapter directory from a Notebook 2 output tree.

    Preference is:
    1. the path itself if it is already an adapter directory
    2. ``final`` after training has completed
    3. the highest numbered ``checkpoint-*`` during training
    4. ``best`` as a fallback
    """
    if root is None:
        return None

    root = Path(root).expanduser()
    if _looks_like_adapter_dir(root):
        return root

    final = root / "final"
    if _looks_like_adapter_dir(final):
        return final

    checkpoints = [
        path for path in root.glob("checkpoint-*")
        if _looks_like_adapter_dir(path)
    ]
    if checkpoints:
        return max(checkpoints, key=_checkpoint_number)

    best = root / "best"
    if _looks_like_adapter_dir(best):
        return best

    return None


@dataclass
class OpenVLAPolicyConfig:
    adapter_dir: Optional[Path]
    base_model: str = OPENVLA_BASE_MODEL
    local_base_dir: Optional[Path] = None
    device: Optional[str] = None
    load_in_4bit: bool = True
    max_new_tokens: int = 32
    allow_base_model: bool = False
    hf_token: Optional[str] = None


class OpenVLAPolicy:
    """Lazy-loading OpenVLA/PEFT policy with the ``predict_action`` evaluator API."""

    decoder_type = "autoregressive_openvla"

    def __init__(self, config: OpenVLAPolicyConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.input_device = None
        self.input_dtype = None

    def reload_adapter(self, adapter_dir) -> None:
        """Drop the loaded model so the next inference uses a newer adapter."""
        adapter_dir = Path(adapter_dir).expanduser() if adapter_dir is not None else None
        if adapter_dir == self.config.adapter_dir and self.model is not None:
            return
        self.close()
        self.config.adapter_dir = adapter_dir

    def _base_model_source(self):
        local = self.config.local_base_dir
        if local is not None and Path(local).expanduser().exists():
            return str(Path(local).expanduser())
        return self.config.base_model

    def _load(self) -> None:
        if self.model is not None:
            return
        if self.config.adapter_dir is None and not self.config.allow_base_model:
            raise FileNotFoundError(
                "No OpenVLA adapter was provided. Pass --adapter-dir or "
                "--watch-adapter-root pointing at Notebook 2 output."
            )

        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForVision2Seq as OpenVLAModelClass
        except ImportError:
            from transformers import AutoModelForImageTextToText as OpenVLAModelClass

        base_source = self._base_model_source()
        token = (
            self.config.hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )

        model_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        use_cuda = torch.cuda.is_available() and (self.config.device or "cuda").startswith("cuda")
        if use_cuda:
            if self.config.load_in_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"
        if token:
            model_kwargs["token"] = token

        base_model = OpenVLAModelClass.from_pretrained(base_source, **model_kwargs)
        if self.config.adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base_model, str(self.config.adapter_dir))
            processor_source = str(self.config.adapter_dir)
        else:
            self.model = base_model
            processor_source = base_source

        try:
            self.processor = AutoProcessor.from_pretrained(
                processor_source,
                trust_remote_code=True,
            )
        except Exception:
            self.processor = AutoProcessor.from_pretrained(base_source, trust_remote_code=True)

        self.model.eval()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        first_param = next(self.model.parameters())
        self.input_device = first_param.device
        self.input_dtype = first_param.dtype

    def _prepare_inputs(self, raw_inputs):
        import torch

        prepared = {}
        for key, value in raw_inputs.items():
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    prepared[key] = value.to(device=self.input_device, dtype=self.input_dtype)
                else:
                    prepared[key] = value.to(self.input_device)
            else:
                prepared[key] = value
        return prepared

    def predict_action(self, image, instruction, **_unused):
        """Run OpenVLA autoregressive generation and parse a 7-DOF action."""
        self._load()

        import torch

        image_pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
        prompt = format_vla_prompt(instruction)
        raw_inputs = self.processor(images=[image_pil], text=[prompt], return_tensors="pt")
        inputs = self._prepare_inputs(raw_inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prompt_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        generated_text = self.processor.batch_decode(
            generated[:, prompt_tokens:],
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
            "decoder_type": self.decoder_type,
            "generated_text": generated_text,
            "parse_failed": parse_failed,
            "inference_time_ms": (time.time() - start) * 1000.0,
            "adapter_dir": str(self.config.adapter_dir) if self.config.adapter_dir else None,
        }

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.input_device = None
        self.input_dtype = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


__all__ = [
    "ACTION_BIN_SIZE",
    "ACTION_BIN_LIMIT",
    "OPENVLA_BASE_MODEL",
    "OpenVLAPolicy",
    "OpenVLAPolicyConfig",
    "decode_franka_axis_token",
    "find_latest_adapter_dir",
    "format_franka_action",
    "format_physical_delta",
    "format_vla_prompt",
    "parse_franka_action",
]
