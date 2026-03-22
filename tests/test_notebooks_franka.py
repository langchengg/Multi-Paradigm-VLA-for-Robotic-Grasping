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

    assert 'def ensure_franka_action_7d(action, source_name="<unknown>"):' in source
    assert 'if action.shape[0] == 7:' in source
    assert 'if action.shape[0] == 4:' in source
    assert 'action_7d = ensure_franka_action_7d(actions[t], source_name=f)' in source
    assert '"action_7d": action_7d' in source
    assert 'sample["action_4d"]' not in source


def test_notebook3_uses_franka_env_and_true_7d_sampling():
    source = (ROOT / "notebooks" / "03_flow_matching_eval.py").read_text()

    assert 'NUMPY_VERSION = "1.26.4"' in source
    assert 'f"numpy=={NUMPY_VERSION}"' in source
    assert "def verify_torch_numpy_bridge():" in source
    assert "torch.tensor([1.0]).numpy()" in source
    assert "from envs.franka_grasp_env import FrankaGraspEnv" in source
    assert 'env = FrankaGraspEnv(image_size=256, camera_name="frontview")' in source
    assert "self.action_dim = action_dim" in source
    assert "self.horizon = horizon" in source
    assert "x.reshape(B, self.horizon, self.action_dim)" in source
    assert 'instructions = data["instructions"]' in source
    assert 'data["instruction"]' not in source
    assert "class DiffusionVLA(nn.Module):" in source
    assert "class OpenVLAPolicyWrapper:" in source
    assert "VLAMuJoCoEvaluator" in source
    assert 'comparison["autoregressive"] = evaluate_policy("autoregressive", openvla_policy, env)' in source
    assert 'comparison["diffusion"] = evaluate_policy("diffusion", diffusion_policy, env)' in source
    assert 'comparison["flow_matching"] = evaluate_policy("flow_matching", flow_policy, env)' in source
    assert 'print(f"   🎬 autoregressive/episode_*.gif")' in source
    assert 'print(f"   🎬 diffusion/episode_*.gif")' in source
    assert 'print(f"   🎬 flow_matching/episode_*.gif")' in source
