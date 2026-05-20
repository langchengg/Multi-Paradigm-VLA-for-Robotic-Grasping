import numpy as np


def test_synthetic_franka_env_reset_step_with_fake_renderer(monkeypatch):
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

    from src.envs.synthetic_franka_grasp_env import SyntheticFrankaGraspEnv

    # The synthetic wrapper can retain a base class imported before other tests
    # reload envs.* modules, so patch the base module globals as well.
    monkeypatch.setitem(
        SyntheticFrankaGraspEnv.__mro__[1].__init__.__globals__,
        "create_renderer",
        fake_create_renderer,
    )

    env = SyntheticFrankaGraspEnv(image_size=32, camera_name="frontview")
    try:
        obs = env.reset(target_object="red_cube", randomize=False)
        assert obs["image"].shape == (32, 32, 3)
        assert obs["robot_state"]["eef_pos"].shape == (3,)
        assert obs["robot_state"]["eef_quat"].shape == (4,)
        assert obs["robot_state"]["qpos"].shape == (9,)
        assert obs["object_state"]["target_pos"].shape == (3,)
        obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
        assert obs["image"].dtype == np.uint8
        assert isinstance(reward, float)
        assert isinstance(done, (bool, np.bool_))
        assert "success" in info
    finally:
        env.close()
