from pathlib import Path


from scripts.run_kaggle_pipeline import (
    build_pipeline_steps,
    cleanup_after_openvla_step,
    parse_steps_arg,
)


ROOT = Path(__file__).resolve().parents[1]


def test_build_pipeline_steps_uses_same_session_working_outputs():
    steps = build_pipeline_steps(ROOT)

    assert len(steps) == 3
    assert steps[0]["script"].name == "01_env_setup_and_demo.py"
    assert steps[1]["script"].name == "02_openvla_qlora_finetune.py"
    assert steps[2]["script"].name == "03_flow_matching_eval.py"
    assert steps[1]["env"]["VLA_DEMO_DIR"] == "/kaggle/working/demos"
    assert steps[1]["post_run"] is cleanup_after_openvla_step
    assert str(steps[1]["outputs"][0]).endswith("/kaggle/working/openvla-finetuned/final")
    assert str(steps[2]["outputs"][0]).endswith("/kaggle/working/results/real_offline_summary.json")


def test_build_pipeline_steps_supports_droid_only_mode():
    steps = build_pipeline_steps(ROOT, droid_only=True)

    assert len(steps) == 3
    assert steps[1]["env"]["VLA_USE_MUJOCO_DEMOS"] == "0"
    assert "VLA_DEMO_DIR" not in steps[1]["env"]


def test_parse_steps_arg_accepts_all_and_subsets():
    assert parse_steps_arg("all", 3) == [0, 1, 2]
    assert parse_steps_arg("1,3", 3) == [0, 2]


def test_cleanup_after_openvla_step_removes_low_disk_targets(tmp_path):
    working_root = tmp_path / "working"
    home_dir = tmp_path / "home"

    demos_dir = working_root / "demos"
    checkpoint_dir = working_root / "openvla-finetuned" / "checkpoint-200"
    final_dir = working_root / "openvla-finetuned" / "final"
    pip_cache_dir = home_dir / ".cache" / "pip"
    datasets_cache_dir = home_dir / ".cache" / "huggingface" / "datasets"

    demos_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    pip_cache_dir.mkdir(parents=True)
    datasets_cache_dir.mkdir(parents=True)

    (demos_dir / "demo_0000.npz").write_bytes(b"demo")
    (checkpoint_dir / "adapter.bin").write_bytes(b"ckpt")
    (final_dir / "adapter_model.bin").write_bytes(b"final")
    (pip_cache_dir / "cache.whl").write_bytes(b"wheel")
    (datasets_cache_dir / "dataset.arrow").write_bytes(b"dataset")

    cleanup_after_openvla_step(working_root=working_root, home_dir=home_dir)

    assert not demos_dir.exists()
    assert not checkpoint_dir.exists()
    assert not pip_cache_dir.exists()
    assert not datasets_cache_dir.exists()
    assert final_dir.exists()
