import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs._rendering import configure_headless_rendering, create_renderer


def test_configure_headless_rendering_defaults_to_osmesa_on_headless_linux():
    environ = {}

    backend = configure_headless_rendering(environ=environ, platform="linux")

    assert backend == "osmesa"
    assert environ["MUJOCO_GL"] == "osmesa"
    assert environ["PYOPENGL_PLATFORM"] == "osmesa"


def test_configure_headless_rendering_preserves_explicit_backend():
    environ = {"MUJOCO_GL": "egl"}

    backend = configure_headless_rendering(environ=environ, platform="linux")

    assert backend == "egl"
    assert environ["MUJOCO_GL"] == "egl"
    assert "PYOPENGL_PLATFORM" not in environ


def test_configure_headless_rendering_skips_non_linux_or_display():
    linux_with_display = {"DISPLAY": ":0"}
    macos = {}

    linux_backend = configure_headless_rendering(
        environ=linux_with_display, platform="linux"
    )
    macos_backend = configure_headless_rendering(environ=macos, platform="darwin")

    assert linux_backend is None
    assert "MUJOCO_GL" not in linux_with_display
    assert macos_backend is None
    assert "MUJOCO_GL" not in macos


def test_create_renderer_rejects_glfw_on_headless_linux():
    class DummyMujoco:
        class Renderer:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("Renderer should not be constructed")

    with pytest.raises(RuntimeError, match="MUJOCO_GL=osmesa or MUJOCO_GL=egl"):
        create_renderer(
            DummyMujoco,
            object(),
            64,
            64,
            environ={"MUJOCO_GL": "glfw"},
            platform="linux",
        )


def test_create_renderer_reports_missing_osmesa_libs(monkeypatch):
    class DummyMujoco:
        class Renderer:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("Renderer should not be constructed")

    monkeypatch.setattr("envs._rendering._missing_osmesa_libraries", lambda: ["OSMesa"])

    with pytest.raises(RuntimeError, match="libosmesa6-dev"):
        create_renderer(DummyMujoco, object(), 64, 64, environ={}, platform="linux")


def test_franka_env_smoke_with_fake_renderer(monkeypatch):
    from envs import franka_grasp_env as franka_module

    class FakeRenderer:
        def __init__(self, _mujoco, _model, height, width):
            self.height = height
            self.width = width
            self.last_camera = None
            self.closed = False

        def update_scene(self, _data, camera=None):
            self.last_camera = camera

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

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


def test_envs_package_runs_shared_rendering_config(monkeypatch):
    import envs

    calls = []
    monkeypatch.setattr(
        "envs._rendering.configure_headless_rendering",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    importlib.reload(envs)

    assert calls


def test_flow_matching_notebook_uses_shared_env_module():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "03_flow_matching_eval.py"
    )
    source = notebook_path.read_text()

    assert "notebooks.simplified_env" not in source
    assert "from envs.simple_grasp_env import SimpleGraspEnv" in source


def test_readme_uses_module_entrypoint_for_franka_smoke_test():
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    source = readme_path.read_text()

    assert "python -m envs.franka_grasp_env" in source
    assert "python envs/franka_grasp_env.py" not in source
