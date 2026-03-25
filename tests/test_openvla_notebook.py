from pathlib import Path


def test_openvla_notebook_pins_compatible_transformers_stack():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert '"transformers==4.40.1"' in source
    assert '"tokenizers==0.19.1"' in source
    assert '"timm==0.9.10"' in source
    assert 'NUMPY_VERSION = "1.26.4"' in source
    assert 'f"numpy=={NUMPY_VERSION}"' in source
    assert "def verify_torch_numpy_bridge():" in source
    assert "torch.tensor([1.0]).numpy()" in source
    assert "LOG_STEPS = 10" in source
    assert 'model.config.use_cache = False' in source
    assert 'if global_step % LOG_STEPS == 0:' in source
    assert '"--upgrade"' in source


def test_openvla_notebook_loads_real_data_from_droid():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert 'USE_DROID = True' in source
    assert 'USE_MUJOCO_DEMOS = os.environ.get("VLA_USE_MUJOCO_DEMOS", "1")' in source
    assert 'DROID_DATASET_REPO_CANDIDATES = [' in source
    assert '"cadene/droid_1.0.1_v30"' in source
    assert "DROID_MAX_SAMPLES = 500" in source
    assert "DROID_FPS = DROID_DEFAULT_FPS" in source
    assert "NUM_EPOCHS = 1" in source
    assert '"av>=12.0.0"' in source
    assert '"opencv-python-headless>=4.9.0"' in source
    assert '"imageio-ffmpeg>=0.4.9"' in source
    assert "from data.droid_utils import (" in source
    assert "def resolve_demo_dir(preferred_dir):" in source
    assert 'if not USE_MUJOCO_DEMOS:' in source
    assert "iter_droid_v30_stream," in source
    assert "load_droid_info," in source
    assert "load_droid_task_lookup," in source
    assert "droid_cartesian_velocity_to_franka_action," in source
    assert "droid_action_to_franka_action," in source
    assert 'image = sample_get(sample, "decoded_image")' in source
    assert 'sample_get(sample, "decode_error")' in source
    assert '"episode_instruction",' in source
    assert "max_raw_droid_frames = max(DROID_MAX_SAMPLES * 8, 2000)" in source
    assert 'raw_action = sample_get(sample, "action.original", "action")' in source
    assert 'cartesian_velocity = sample_get(sample, "action.cartesian_velocity")' in source
    assert 'gripper_position = sample_get(sample, "action.gripper_position")' in source
    assert 'gripper_velocity = sample_get(sample, "action.gripper_velocity")' in source
    assert 'Loaded 0 real DROID samples from all candidate repos.' in source
    assert 'DROID skip stats:' in source
    assert 'use_mujoco_demos=USE_MUJOCO_DEMOS' in source
    assert "def load_droid_task_lookup(repo_id):" not in source
    assert "def droid_cartesian_velocity_to_franka_action(" not in source
    assert "def droid_action_to_franka_action(action, source_name=" not in source
    assert 'Simulated loading' not in source
    assert 'physical-intelligence/libero' not in source
    assert "Could not find any training samples." in source
    assert "MuJoCo demos disabled; training with DROID-only data." in source


def test_openvla_notebook_uses_vision_model_fallback_and_real_demo_key():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert "AutoModelForVision2Seq as OpenVLAModelClass" in source
    assert "AutoModelForImageTextToText as OpenVLAModelClass" in source
    assert 'model = OpenVLAModelClass.from_pretrained(' in source
    assert 'model_kwargs["device_map"] = {"": 0}' in source
    assert 'device_map="auto"' not in source
    assert 'instructions = data["instructions"]' in source
    assert 'data["instruction"]' not in source


def test_openvla_notebook_supervises_structured_franka_delta_pose_targets():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert "FRANKA_ACTION_KEYS," in source
    assert "TRANSLATION_STEP_M," in source
    assert "ROTATION_STEP_RAD," in source
    assert "def format_franka_action(action):" in source
    assert "def parse_franka_action(text):" in source
    assert 'ACTION_BIN_SIZE = 0.05' in source
    assert '"action_encoding": "compact_integer_bins_v1"' in source
    assert '"+06 -03 +00 +02 -11 +00 o"' in source
    assert "WeightedRandomSampler" in source
    assert "def build_training_sampler(samples):" in source
    assert 'outputs = model(**inputs)' in source
    assert 'labels=inputs["input_ids"]' not in source
    assert 'labels[i, :prompt_len] = -100' in source
    assert "franka_action_config.json" in source
    assert "def collate_vla_batch(batch):" in source
