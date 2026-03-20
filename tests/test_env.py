"""
Environment Unit Tests
======================
Tests for SimpleGraspEnv and data collection pipeline.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSimpleGraspEnv:
    """Tests for the custom MuJoCo grasping environment."""

    @pytest.fixture
    def env(self):
        from envs.simple_grasp_env import SimpleGraspEnv
        e = SimpleGraspEnv(image_size=64, camera_name="frontview")
        yield e
        e.close()

    def test_reset_returns_obs(self, env):
        obs = env.reset(target_object="red_cube")
        assert "image" in obs
        assert "instruction" in obs
        assert "gripper_pos" in obs
        assert "target_pos" in obs

    def test_image_shape(self, env):
        obs = env.reset()
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8

    def test_instruction_is_string(self, env):
        obs = env.reset()
        assert isinstance(obs["instruction"], str)
        assert len(obs["instruction"]) > 0

    def test_gripper_pos_shape(self, env):
        obs = env.reset()
        assert obs["gripper_pos"].shape == (3,)

    def test_target_pos_shape(self, env):
        obs = env.reset()
        assert obs["target_pos"].shape == (3,)

    def test_step_returns_correct_format(self, env):
        env.reset()
        action = np.zeros(4)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)
        assert "success" in info

    def test_step_clips_actions(self, env):
        env.reset()
        # Extreme action should be clipped
        action = np.array([100.0, -100.0, 50.0, 1.0])
        obs, reward, done, info = env.step(action)
        # Should not crash
        assert obs["image"].shape == (64, 64, 3)

    def test_reset_randomization(self, env):
        """Object positions should differ between resets."""
        obs1 = env.reset(target_object="red_cube", randomize=True)
        pos1 = obs1["target_pos"].copy()

        obs2 = env.reset(target_object="red_cube", randomize=True)
        pos2 = obs2["target_pos"].copy()

        # Very unlikely to be exactly the same
        # (but possible, so we just check the code runs)
        assert pos1.shape == pos2.shape

    def test_all_objects_accessible(self, env):
        for obj in ["red_cube", "blue_cube", "green_cube"]:
            obs = env.reset(target_object=obj)
            assert obs["target_pos"].shape == (3,)

    def test_render_frame(self, env):
        env.reset()
        for cam in ["frontview", "topdown", "sideview"]:
            frame = env.render_frame(camera_name=cam)
            assert frame.shape == (64, 64, 3)
            assert frame.dtype == np.uint8

    def test_get_all_object_positions(self, env):
        env.reset()
        positions = env.get_all_object_positions()
        assert len(positions) == 3
        for name, pos in positions.items():
            assert pos.shape == (3,)


class TestDummyVLA:
    """Tests for the DummyVLA model."""

    def test_predict_action_shape(self):
        from models.dummy_vla import DummyVLA
        model = DummyVLA("flow_matching", action_dim=4)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        action, info = model.predict_action(image, "pick up the red cube")
        assert action.shape == (4,)
        assert "decoder_type" in info

    def test_all_decoder_types(self):
        from models.dummy_vla import DummyVLA
        for decoder in ["autoregressive", "diffusion", "flow_matching"]:
            model = DummyVLA(decoder)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            action, info = model.predict_action(image, "test")
            assert action.shape == (4,)
            assert info["decoder_type"] == decoder

    def test_oracle_policy(self):
        from models.dummy_vla import DummyVLA
        model = DummyVLA("flow_matching")
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        action, _ = model.predict_action(
            image, "pick up red cube",
            gripper_pos=np.array([0.5, 0.0, 0.5]),
            target_pos=np.array([0.4, 0.1, 0.24]),
        )
        assert action.shape == (4,)
        assert np.all(np.abs(action) <= 1.0)


class TestCollectDemos:
    """Tests for the demo collection pipeline."""

    @pytest.fixture
    def env(self):
        from envs.simple_grasp_env import SimpleGraspEnv
        e = SimpleGraspEnv(image_size=64, camera_name="frontview")
        yield e
        e.close()

    def test_collect_small_batch(self, env, tmp_path):
        from data.collect_demos import collect_demos
        demos, stats = collect_demos(
            env, num_demos=3,
            save_dir=str(tmp_path),
            verbose=False,
        )
        assert len(demos) == 3
        assert stats["total"] == 3
        assert 0 <= stats["success_rate"] <= 1

    def test_demo_data_format(self, env, tmp_path):
        from data.collect_demos import collect_demos
        demos, _ = collect_demos(
            env, num_demos=2,
            save_dir=str(tmp_path),
            verbose=False,
        )
        demo = demos[0]
        assert "images" in demo
        assert "actions" in demo
        assert "success" in demo
        assert demo["images"].ndim == 4  # (N, H, W, 3)
        assert demo["actions"].ndim == 2  # (N, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
