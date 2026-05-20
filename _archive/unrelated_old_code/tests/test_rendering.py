import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs._rendering import configure_headless_rendering, create_renderer


def test_configure_headless_rendering_accepts_explicit_backend(monkeypatch):
    environ = {"MUJOCO_GL": "egl"}
    calls = []

    monkeypatch.setattr(
        "envs._rendering._ensure_backend_works",
        lambda backend, _environ, _python: calls.append(backend),
    )

    backend = configure_headless_rendering(
        environ=environ,
        platform="linux",
        python_executable="python",
    )

    assert backend == "egl"
    assert environ["PYOPENGL_PLATFORM"] == "egl"
    assert calls == ["egl"]


def test_configure_headless_rendering_uses_requested_pyopengl_platform(monkeypatch):
    environ = {"PYOPENGL_PLATFORM": "osmesa"}
    calls = []

    monkeypatch.setattr(
        "envs._rendering._ensure_backend_works",
        lambda backend, _environ, _python: calls.append(backend),
    )

    backend = configure_headless_rendering(
        environ=environ,
        platform="linux",
        python_executable="python",
    )

    assert backend == "osmesa"
    assert environ["MUJOCO_GL"] == "osmesa"
    assert calls == ["osmesa"]


def test_configure_headless_rendering_auto_falls_back_to_osmesa(monkeypatch):
    environ = {}

    monkeypatch.setattr("envs._rendering._AUTO_BACKEND", None)

    def fake_probe(backend, _environ, _python):
        if backend == "egl":
            return False, "egl failed"
        return True, "ok"

    monkeypatch.setattr("envs._rendering._probe_backend", fake_probe)

    backend = configure_headless_rendering(
        environ=environ,
        platform="linux",
        python_executable="python",
    )

    assert backend == "osmesa"
    assert environ["MUJOCO_GL"] == "osmesa"
    assert environ["PYOPENGL_PLATFORM"] == "osmesa"


def test_configure_headless_rendering_rejects_invalid_headless_backend():
    with pytest.raises(RuntimeError, match="MUJOCO_GL=egl or MUJOCO_GL=osmesa"):
        configure_headless_rendering(
            environ={"MUJOCO_GL": "glfw"},
            platform="linux",
            python_executable="python",
        )


def test_configure_headless_rendering_reports_broken_explicit_backend(monkeypatch):
    monkeypatch.setattr(
        "envs._rendering._probe_backend",
        lambda backend, _environ, _python: (False, f"{backend} broke"),
    )

    with pytest.raises(RuntimeError, match="Configured headless MuJoCo backend 'osmesa'"):
        configure_headless_rendering(
            environ={"MUJOCO_GL": "osmesa"},
            platform="linux",
            python_executable="python",
        )


def test_create_renderer_wraps_backend_errors(monkeypatch):
    class DummyMujoco:
        class Renderer:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("egl init failed")

    monkeypatch.setattr(
        "envs._rendering.configure_headless_rendering",
        lambda **_kwargs: "egl",
    )

    with pytest.raises(RuntimeError, match="MUJOCO_GL='egl'"):
        create_renderer(
            DummyMujoco,
            object(),
            64,
            64,
            environ={},
            platform="linux",
        )


def test_franka_env_smoke_with_fake_renderer(monkeypatch):
    from envs import franka_grasp_env as franka_module

    class FakeRenderer:
        def __init__(self, _mujoco, _model, height, width):
            self.height = height
            self.width = width

        def update_scene(self, _data, camera=None):
            self.camera = camera

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    def fake_create_renderer(mujoco, model, *, height, width):
        return FakeRenderer(mujoco, model, height, width)

    monkeypatch.setattr(franka_module, "create_renderer", fake_create_renderer)

    env = franka_module.FrankaGraspEnv(image_size=32, camera_name="frontview")
    try:
        obs = env.reset(target_object="red_cube", randomize=False)
        assert obs["image"].shape == (32, 32, 3)
        assert obs["image"].dtype == np.uint8

        frame = env.render_frame(camera_name="topdown")
        assert frame.shape == (32, 32, 3)

        obs, reward, done, info = env.step(np.zeros(4))
        assert obs["image"].shape == (32, 32, 3)
        assert isinstance(reward, float)
        assert isinstance(done, (bool, np.bool_))
        assert "success" in info
    finally:
        env.close()


def test_franka_env_cartesian_z_command_moves_down(monkeypatch):
    from envs import franka_grasp_env as franka_module

    class FakeRenderer:
        def __init__(self, _mujoco, _model, height, width):
            self.height = height
            self.width = width

        def update_scene(self, _data, camera=None):
            self.camera = camera

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(
        franka_module,
        "create_renderer",
        lambda mujoco, model, *, height, width: FakeRenderer(mujoco, model, height, width),
    )

    env = franka_module.FrankaGraspEnv(image_size=32, camera_name="frontview")
    try:
        env.reset(target_object="red_cube", randomize=False)
        start = env._get_ee_pos().copy()
        for _ in range(10):
            env.step(np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]))
        delta = env._get_ee_pos() - start
        assert delta[2] < -0.04
    finally:
        env.close()


def test_franka_scripted_rollout_succeeds_deterministically(monkeypatch):
    from data.collect_demos import scripted_grasp_policy
    from envs import franka_grasp_env as franka_module

    class FakeRenderer:
        def __init__(self, _mujoco, _model, height, width):
            self.height = height
            self.width = width

        def update_scene(self, _data, camera=None):
            self.camera = camera

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(
        franka_module,
        "create_renderer",
        lambda mujoco, model, *, height, width: FakeRenderer(mujoco, model, height, width),
    )

    env = franka_module.FrankaGraspEnv(image_size=32, camera_name="frontview")
    try:
        obs = env.reset(target_object="red_cube", randomize=False)
        target_pos = obs["target_pos"].copy()
        phase = 0
        phase_step = 0
        info = {"success": False}

        for _ in range(env._max_steps):
            action, phase, phase_step = scripted_grasp_policy(obs, phase, phase_step, target_pos)
            obs, _reward, done, info = env.step(np.clip(action, -1.0, 1.0))
            if done:
                break

        assert bool(info["success"]) is True
    finally:
        env.close()


def test_envs_package_runs_shared_rendering_config(monkeypatch):
    import envs

    calls = []
    monkeypatch.setattr(
        "envs._rendering.configure_headless_rendering",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    importlib.reload(envs)

    assert calls


def test_envs_package_is_lazy(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    for name in list(sys.modules):
        if name == "envs" or name.startswith("envs."):
            sys.modules.pop(name)

    envs = importlib.import_module("envs")

    assert "envs.simple_grasp_env" not in sys.modules
    assert "envs.franka_grasp_env" not in sys.modules

    _ = envs.SimpleGraspEnv
    assert "envs.simple_grasp_env" in sys.modules


def test_flow_matching_notebook_uses_real_data_offline_eval_path():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "03_flow_matching_eval.py"
    )
    source = notebook_path.read_text()

    assert "notebooks.simplified_env" not in source
    assert "DROID_DATASET_REPO_CANDIDATES" in source
    assert "load_real_droid_records" in source
    assert "OfflineRealDataEvaluator" in source
    assert "from data.droid_utils import (" in source
    assert "from models.flow_matching_head import FlowMatchingHead" in source
    assert "FrankaGraspEnv" not in source
    assert "VLAMuJoCoEvaluator" not in source
    assert "configure_headless_rendering()" not in source
    assert 'os.environ["MUJOCO_GL"] = "osmesa"' not in source


def test_env_setup_notebook_uses_shared_rendering_helper():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "01_env_setup_and_demo.py"
    )
    source = notebook_path.read_text()

    assert "configure_headless_rendering()" in source
    assert 'os.environ["MUJOCO_GL"] = "osmesa"' not in source


def test_readme_uses_module_entrypoint_for_franka_smoke_test():
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    source = readme_path.read_text()

    assert "python -m envs.franka_grasp_env" in source
    assert "python envs/franka_grasp_env.py" not in source
