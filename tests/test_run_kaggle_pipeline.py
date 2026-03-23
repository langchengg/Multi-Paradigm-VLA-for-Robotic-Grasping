from pathlib import Path


from scripts.run_kaggle_pipeline import build_pipeline_steps, parse_steps_arg


ROOT = Path(__file__).resolve().parents[1]


def test_build_pipeline_steps_uses_same_session_working_outputs():
    steps = build_pipeline_steps(ROOT)

    assert len(steps) == 3
    assert steps[0]["script"].name == "01_env_setup_and_demo.py"
    assert steps[1]["script"].name == "02_openvla_qlora_finetune.py"
    assert steps[2]["script"].name == "03_flow_matching_eval.py"
    assert steps[1]["env"]["VLA_DEMO_DIR"] == "/kaggle/working/demos"
    assert str(steps[1]["outputs"][0]).endswith("/kaggle/working/openvla-finetuned/final")
    assert str(steps[2]["outputs"][0]).endswith("/kaggle/working/results/real_offline_summary.json")


def test_parse_steps_arg_accepts_all_and_subsets():
    assert parse_steps_arg("all", 3) == [0, 1, 2]
    assert parse_steps_arg("1,3", 3) == [0, 2]
