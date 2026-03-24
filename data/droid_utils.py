"""
Shared DROID + Franka action utilities.

These helpers keep Notebook 2 and Notebook 3 aligned on the same:
- Franka delta-pose action interface
- DROID action conversion assumptions
- Hugging Face sample parsing helpers
"""

import io
import json
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


FRANKA_ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")
TRANSLATION_STEP_M = 0.03
ROTATION_STEP_RAD = 0.05
GRIPPER_OPEN_VALUE = -1.0
GRIPPER_CLOSE_VALUE = 1.0
ACTION_MIN = -1.0
ACTION_MAX = 1.0
DROID_DEFAULT_FPS = 15.0
DROID_CAMERA_KEYS = (
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
)

__all__ = [
    "ACTION_MAX",
    "ACTION_MIN",
    "DROID_CAMERA_KEYS",
    "DROID_DEFAULT_FPS",
    "FRANKA_ACTION_KEYS",
    "GRIPPER_CLOSE_VALUE",
    "GRIPPER_OPEN_VALUE",
    "ROTATION_STEP_RAD",
    "TRANSLATION_STEP_M",
    "droid_action_to_franka_action",
    "droid_cartesian_velocity_to_franka_action",
    "ensure_franka_action_7d",
    "gripper_value_to_binary_command",
    "image_to_uint8_array",
    "iter_droid_v30_stream",
    "load_droid_info",
    "load_droid_task_lookup",
    "sample_get",
]


def ensure_franka_action_7d(
    action,
    source_name="<unknown>",
    action_min=ACTION_MIN,
    action_max=ACTION_MAX,
):
    """Convert actions to the repo's normalized 7-DOF Franka delta-pose interface."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == 7:
        return np.clip(action, action_min, action_max)
    if action.shape[0] == 4:
        return np.array(
            [action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]],
            dtype=np.float32,
        )
    raise ValueError(
        f"Unsupported action dimension {action.shape[0]} in {source_name}. "
        "Expected 7-DOF Franka actions or legacy 4-DOF actions."
    )


def gripper_value_to_binary_command(
    value,
    gripper_open_value=GRIPPER_OPEN_VALUE,
    gripper_close_value=GRIPPER_CLOSE_VALUE,
):
    """Collapse continuous gripper annotations to this repo's open/close sign convention."""
    scalar = float(np.asarray(value, dtype=np.float32).reshape(-1)[0])
    if 0.0 <= scalar <= 1.0:
        return gripper_close_value if scalar >= 0.5 else gripper_open_value
    return gripper_close_value if scalar > 0.0 else gripper_open_value


def droid_cartesian_velocity_to_franka_action(
    cartesian_velocity,
    *,
    gripper_position=None,
    gripper_velocity=None,
    droid_fps=DROID_DEFAULT_FPS,
    translation_step_m=TRANSLATION_STEP_M,
    rotation_step_rad=ROTATION_STEP_RAD,
    action_min=ACTION_MIN,
    action_max=ACTION_MAX,
    gripper_open_value=GRIPPER_OPEN_VALUE,
    gripper_close_value=GRIPPER_CLOSE_VALUE,
    source_name="<unknown>",
):
    """
    Convert DROID Cartesian velocity commands into the normalized Franka delta-pose action.

    DROID stores real Franka commands at a fixed control rate. We integrate one control
    period, then normalize by the MuJoCo Franka controller's per-step translation and
    rotation scales used throughout this repo.
    """
    velocity = np.asarray(cartesian_velocity, dtype=np.float32).reshape(-1)
    if velocity.shape[0] < 6:
        raise ValueError(
            f"Unsupported DROID action dimension {velocity.shape[0]} in {source_name}. "
            "Expected at least 6 Cartesian velocity values."
        )

    delta_xyz = velocity[:3] / droid_fps
    delta_rpy = velocity[3:6] / droid_fps
    if gripper_position is not None:
        gripper = gripper_value_to_binary_command(
            gripper_position,
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
    elif gripper_velocity is not None:
        gripper = gripper_value_to_binary_command(
            gripper_velocity,
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
    else:
        gripper = gripper_open_value

    normalized = np.array(
        [
            delta_xyz[0] / translation_step_m,
            delta_xyz[1] / translation_step_m,
            delta_xyz[2] / translation_step_m,
            delta_rpy[0] / rotation_step_rad,
            delta_rpy[1] / rotation_step_rad,
            delta_rpy[2] / rotation_step_rad,
            gripper,
        ],
        dtype=np.float32,
    )
    return ensure_franka_action_7d(
        normalized,
        source_name=source_name,
        action_min=action_min,
        action_max=action_max,
    )


def droid_action_to_franka_action(
    action,
    *,
    translation_step_m=TRANSLATION_STEP_M,
    rotation_step_rad=ROTATION_STEP_RAD,
    action_min=ACTION_MIN,
    action_max=ACTION_MAX,
    gripper_open_value=GRIPPER_OPEN_VALUE,
    gripper_close_value=GRIPPER_CLOSE_VALUE,
    source_name="<unknown>",
):
    """
    Adapt flattened DROID actions to this repo's normalized Franka delta-pose interface.

    The Hugging Face DROID conversions expose a flattened action vector. When XYZ / RPY
    values are already near [-1, 1], we keep them as normalized controls; otherwise we
    treat them as metric / radian deltas and scale them into the shared Franka interface.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 6:
        raise ValueError(
            f"Unsupported DROID flattened action dimension {action.shape[0]} in {source_name}."
        )

    out = np.zeros(7, dtype=np.float32)
    out[:6] = action[:6]
    if action.shape[0] >= 7:
        out[6] = gripper_value_to_binary_command(
            action[6],
            gripper_open_value=gripper_open_value,
            gripper_close_value=gripper_close_value,
        )
    else:
        out[6] = gripper_open_value

    if np.max(np.abs(out[:3])) > 1.0:
        out[:3] = out[:3] / translation_step_m
    if np.max(np.abs(out[3:6])) > 1.0:
        out[3:6] = out[3:6] / rotation_step_rad
    return ensure_franka_action_7d(
        out,
        source_name=source_name,
        action_min=action_min,
        action_max=action_max,
    )


def sample_get(sample, *paths):
    """Retrieve a possibly nested value from a Hugging Face sample dict."""
    for path in paths:
        if isinstance(sample, dict) and path in sample and sample[path] is not None:
            return sample[path]
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


def image_to_uint8_array(image, source_name):
    """Normalize PIL / Hugging Face image payloads to HWC uint8 arrays."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, dict):
        if image.get("bytes") is not None:
            arr = np.array(Image.open(io.BytesIO(image["bytes"])).convert("RGB"), dtype=np.uint8)
        elif image.get("path"):
            arr = np.array(Image.open(image["path"]).convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported image payload in {source_name}: {image.keys()}")
    elif hasattr(image, "to_pil"):
        arr = np.array(image.to_pil().convert("RGB"), dtype=np.uint8)
    elif hasattr(image, "numpy"):
        arr = np.asarray(image.numpy())
    elif hasattr(image, "asnumpy"):
        arr = np.asarray(image.asnumpy())
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


def load_droid_task_lookup(repo_id):
    """Map DROID task indices to natural-language instructions when annotations are missing."""
    from huggingface_hub import hf_hub_download

    try:
        tasks_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="meta/tasks.jsonl",
        )
    except Exception:
        return {}

    task_lookup = {}
    with open(tasks_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                task_lookup[int(record["task_index"])] = record["task"]
    return task_lookup


def _extract_episode_instruction(tasks):
    """Return the first non-empty episode-level instruction, if present."""
    for task in tasks or []:
        if isinstance(task, str) and task.strip():
            return task.strip()
    return None


@lru_cache(maxsize=4)
def load_droid_info(repo_id):
    """Load DROID dataset metadata required to resolve video-backed image streams."""
    from huggingface_hub import hf_hub_download

    info_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="meta/info.json",
    )
    with open(info_path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _list_droid_data_files(repo_id):
    """Return sorted parquet shards for the DROID train split."""
    from huggingface_hub import HfApi

    api = HfApi()
    return tuple(
        sorted(
            file_name
            for file_name in api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            if file_name.startswith("data/") and file_name.endswith(".parquet")
        )
    )


@lru_cache(maxsize=16)
def _load_droid_episode_rows(repo_id, meta_file):
    """Load per-file episode metadata rows used to map frame rows to MP4 shards."""
    from huggingface_hub import hf_hub_download

    meta_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=meta_file,
    )
    return tuple(pq.read_table(meta_path).to_pylist())


class _OpenCVVideoCache:
    """Small LRU cache for MP4 readers so DROID frame extraction stays cheap."""

    def __init__(self, max_open=4):
        self.max_open = max_open
        self._pyav_states = OrderedDict()
        self._caps = OrderedDict()
        self._imageio_readers = OrderedDict()

    def close(self):
        for state in self._pyav_states.values():
            state["container"].close()
        self._pyav_states.clear()
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        for reader in self._imageio_readers.values():
            reader.close()
        self._imageio_readers.clear()

    def _get_cap(self, video_path):
        import cv2

        cap = self._caps.pop(video_path, None)
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open DROID video file: {video_path}")
        self._caps[video_path] = cap
        while len(self._caps) > self.max_open:
            _, old_cap = self._caps.popitem(last=False)
            old_cap.release()
        return cap

    def read_frame(self, video_path, frame_index):
        import cv2

        errors = {}

        try:
            return self._read_frame_pyav(video_path, frame_index)
        except Exception as exc:
            errors["pyav"] = f"{type(exc).__name__}: {exc}"

        cap = self._get_cap(video_path)
        try:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
                raise ValueError(f"Failed to seek to frame {frame_index} in {video_path}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Failed to decode frame {frame_index} in {video_path}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            errors["cv2"] = f"{type(exc).__name__}: {exc}"

        try:
            return self._read_frame_imageio(video_path, frame_index)
        except Exception as exc:
            errors["imageio"] = f"{type(exc).__name__}: {exc}"

        raise ValueError(
            f"Failed to decode frame {frame_index} in {video_path} via cv2/imageio. Errors: {errors}"
        )

    def _get_imageio_reader(self, video_path):
        import imageio.v2 as imageio

        reader = self._imageio_readers.pop(video_path, None)
        if reader is None:
            reader = imageio.get_reader(video_path, format="ffmpeg")
        self._imageio_readers[video_path] = reader
        while len(self._imageio_readers) > self.max_open:
            _, old_reader = self._imageio_readers.popitem(last=False)
            old_reader.close()
        return reader

    def _read_frame_imageio(self, video_path, frame_index):
        reader = self._get_imageio_reader(video_path)
        frame = np.asarray(reader.get_data(int(frame_index)))
        if frame.ndim != 3:
            raise ValueError(
                f"ImageIO returned frame with shape {frame.shape} for {video_path}:{frame_index}"
            )
        return frame

    def _get_pyav_state(self, video_path, frame_index):
        import av

        state = self._pyav_states.pop(video_path, None)
        if state is None or int(frame_index) < state["next_index"]:
            if state is not None:
                state["container"].close()
            container = av.open(video_path)
            stream = container.streams.video[0]
            state = {
                "container": container,
                "frames": container.decode(stream),
                "next_index": 0,
            }
        self._pyav_states[video_path] = state
        while len(self._pyav_states) > self.max_open:
            _, old_state = self._pyav_states.popitem(last=False)
            old_state["container"].close()
        return state

    def _read_frame_pyav(self, video_path, frame_index):
        state = self._get_pyav_state(video_path, frame_index)
        target = int(frame_index)
        while state["next_index"] <= target:
            frame = next(state["frames"])
            current = state["next_index"]
            state["next_index"] += 1
            if current == target:
                return frame.to_ndarray(format="rgb24")
        raise ValueError(f"PyAV could not reach frame {frame_index} in {video_path}")


def iter_droid_v30_stream(
    repo_id,
    *,
    split="train",
    max_samples=None,
    camera_keys=DROID_CAMERA_KEYS,
    max_open_videos=4,
    skip_unlabeled_episodes=True,
):
    """
    Yield DROID v3 frame rows with decoded RGB images.

    The Hugging Face parquet rows do not inline video frames. Instead, each parquet
    file has a matching `meta/episodes/...` file that tells us which MP4 shard holds
    the frames for each episode. We walk data shards sequentially, track the local
    episode inside the shard via `is_first`, then decode the requested frame from the
    corresponding MP4 on demand.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    if split != "train":
        raise ValueError(f"Unsupported DROID split {split!r}; expected 'train'.")

    info = load_droid_info(repo_id)
    video_path_template = info.get("video_path")
    if not video_path_template:
        raise ValueError(f"DROID repo {repo_id} does not expose video_path metadata.")
    fps = float(info.get("fps", DROID_DEFAULT_FPS))

    data_files = _list_droid_data_files(repo_id)
    if not data_files:
        raise FileNotFoundError(f"No DROID parquet shards found in repo {repo_id}")

    reader = _OpenCVVideoCache(max_open=max_open_videos)
    yielded = 0

    try:
        for data_file in data_files:
            meta_file = data_file.replace("data/", "meta/episodes/")
            episode_rows = _load_droid_episode_rows(repo_id, meta_file)
            if not episode_rows:
                continue

            local_episode_idx = -1
            data_stream = load_dataset(
                repo_id,
                split=split,
                streaming=True,
                data_files=data_file,
            )

            for sample in data_stream:
                if sample_get(sample, "is_first"):
                    local_episode_idx += 1
                if local_episode_idx < 0 or local_episode_idx >= len(episode_rows):
                    raise IndexError(
                        f"DROID episode pointer {local_episode_idx} is out of bounds for {meta_file}"
                    )

                episode_row = episode_rows[local_episode_idx]
                episode_instruction = _extract_episode_instruction(episode_row.get("tasks"))
                if skip_unlabeled_episodes and episode_instruction is None:
                    continue
                frame_index = int(sample_get(sample, "frame_index") or 0)

                image = None
                last_error = None
                used_camera = None
                for camera_key in camera_keys:
                    chunk_idx = episode_row.get(f"videos/{camera_key}/chunk_index")
                    file_idx = episode_row.get(f"videos/{camera_key}/file_index")
                    from_ts = episode_row.get(f"videos/{camera_key}/from_timestamp")
                    if chunk_idx is None or file_idx is None or from_ts is None:
                        continue

                    absolute_frame = int(round(float(from_ts) * fps)) + frame_index
                    video_file = video_path_template.format(
                        video_key=camera_key,
                        chunk_index=int(chunk_idx),
                        file_index=int(file_idx),
                    )
                    try:
                        local_video_path = hf_hub_download(
                            repo_id=repo_id,
                            repo_type="dataset",
                            filename=video_file,
                        )
                        image = reader.read_frame(local_video_path, absolute_frame)
                        used_camera = camera_key
                        break
                    except Exception as exc:
                        last_error = exc

                if image is None:
                    out = dict(sample)
                    out["observation.images.active_camera"] = None
                    out["episode_instruction"] = episode_instruction
                    out["decoded_image"] = None
                    if last_error is not None:
                        out["decode_error"] = f"{type(last_error).__name__}: {last_error}"
                    yield out
                    yielded += 1
                    if max_samples is not None and yielded >= max_samples:
                        return
                    continue

                out = dict(sample)
                out["observation.images.active_camera"] = used_camera
                out["episode_instruction"] = episode_instruction
                out["decoded_image"] = image
                yield out
                yielded += 1

                if max_samples is not None and yielded >= max_samples:
                    return
    finally:
        reader.close()
