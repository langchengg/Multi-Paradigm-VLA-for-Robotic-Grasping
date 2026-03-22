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


def test_openvla_notebook_loads_real_data_instead_of_simulating_libero():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert 'LIBERO_DATASET_REPO = "physical-intelligence/libero"' in source
    assert "LIBERO_MAX_SAMPLES = 5000" in source
    assert "def resolve_demo_dir(preferred_dir):" in source
    assert 'load_dataset(' in source
    assert 'streaming=True' in source
    assert 'load_libero_task_lookup(LIBERO_DATASET_REPO)' in source
    assert 'Simulated loading' not in source
    assert "Could not find any training samples." in source


def test_openvla_notebook_uses_vision_model_fallback_and_real_demo_key():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert "AutoModelForVision2Seq as OpenVLAModelClass" in source
    assert "AutoModelForImageTextToText as OpenVLAModelClass" in source
    assert 'model = OpenVLAModelClass.from_pretrained(' in source
    assert 'instructions = data["instructions"]' in source
    assert 'data["instruction"]' not in source


def test_openvla_notebook_supervises_structured_franka_delta_pose_targets():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "02_openvla_qlora_finetune.py"
    )
    source = notebook_path.read_text()

    assert 'FRANKA_ACTION_KEYS = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")' in source
    assert "TRANSLATION_STEP_M = 0.03" in source
    assert "ROTATION_STEP_RAD = 0.05" in source
    assert "def format_franka_action(action):" in source
    assert "def parse_franka_action(text):" in source
    assert 'gripper=open|close' in source
    assert 'outputs = model(**inputs)' in source
    assert 'labels=inputs["input_ids"]' not in source
    assert 'labels[i, :prompt_len] = -100' in source
    assert "franka_action_config.json" in source
    assert "def collate_vla_batch(batch):" in source
