from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebook1_collects_franka_demos():
    source = (ROOT / "notebooks" / "01_env_setup_and_demo.py").read_text()

    assert "from envs.franka_grasp_env import FrankaGraspEnv" in source
    assert "from data.collect_demos import collect_demos" in source
    assert 'env = FrankaGraspEnv(image_size=IMAGE_SIZE, camera_name="frontview")' in source
    assert "add_noise=False" in source
    assert "Action format: 7-DOF" in source
    assert "env = SimpleGraspEnv(image_size=IMAGE_SIZE)" not in source


def test_notebook2_uses_native_7d_actions_with_legacy_fallback():
    source = (ROOT / "notebooks" / "02_openvla_qlora_finetune.py").read_text()

    assert "from data.droid_utils import (" in source
    assert "ensure_franka_action_7d," in source
    assert 'def ensure_franka_action_7d(action, source_name="<unknown>"):' not in source
    assert 'action_7d = ensure_franka_action_7d(actions[t], source_name=f)' in source
    assert '"action_7d": action_7d' in source
    assert 'sample["action_4d"]' not in source


def test_notebook3_uses_droid_real_data_for_offline_eval():
    source = (ROOT / "notebooks" / "03_flow_matching_eval.py").read_text()

    assert 'NUMPY_VERSION = "1.26.4"' in source
    assert 'f"numpy=={NUMPY_VERSION}"' in source
    assert "def verify_torch_numpy_bridge():" in source
    assert "torch.tensor([1.0]).numpy()" in source
    assert 'DROID_DATASET_REPO_CANDIDATES = [' in source
    assert '"cadene/droid"' in source
    assert 'DROID_MAX_SAMPLES = 500' in source
    assert 'DROID_EVAL_FRACTION = 0.2' in source
    assert 'DROID_FPS = DROID_DEFAULT_FPS' in source
    assert 'ACTION_HORIZON = 1' in source
    assert "from data.droid_utils import (" in source
    assert "from models.flow_matching_head import FlowMatchingHead" in source
    assert "self.action_dim = action_dim" in source
    assert "self.horizon = horizon" in source
    assert "action_horizon=horizon" in source
    assert "return self.flow_head.sample(features, num_steps=steps)" in source
    assert "def load_real_droid_records(max_samples):" in source
    assert "def split_records_by_episode(records, eval_fraction):" in source
    assert "DROID skip stats:" in source
    assert "class OfflineRealDataEvaluator:" in source
    assert 'train_records, eval_records = split_records_by_episode(all_droid_records, DROID_EVAL_FRACTION)' in source
    assert "class DiffusionVLA(nn.Module):" in source
    assert "class OpenVLAPolicyWrapper:" in source
    assert "VLAMuJoCoEvaluator" not in source
    assert "FrankaGraspEnv" not in source
    assert 'comparison["autoregressive"] = evaluate_policy_offline("autoregressive", openvla_policy, eval_records)' in source
    assert 'comparison["diffusion"] = evaluate_policy_offline("diffusion", diffusion_policy, eval_records)' in source
    assert 'comparison["flow_matching"] = evaluate_policy_offline("flow_matching", flow_policy, eval_records)' in source
    assert 'print(f"   📊 real_offline_metrics.png")' in source
    assert 'print(f"   🖼️ real_offline_examples.png")' in source
    assert 'print(f"   📄 real_offline_summary.json")' in source
