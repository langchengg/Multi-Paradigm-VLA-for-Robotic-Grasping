"""
Robosuite VLA Wrapper
====================
Wraps robosuite environments (PickPlaceCan, Lift) into a unified interface
suitable for VLA model evaluation with vision + language observations.
"""

import numpy as np

try:
    import robosuite as suite
    from robosuite.wrappers import Wrapper
    HAS_ROBOSUITE = True
except ImportError:
    HAS_ROBOSUITE = False


# Language instruction templates for each robosuite task
TASK_INSTRUCTIONS = {
    "Lift": [
        "pick up the cube",
        "lift the red cube from the table",
        "grasp and raise the cube",
        "grab the cube and lift it up",
    ],
    "PickPlaceCan": [
        "pick up the can and place it in the bin",
        "grasp the can and move it to the target",
        "grab the soda can from the table",
        "pick the can up",
    ],
    "NutAssemblySquare": [
        "pick up the square nut and place it on the peg",
        "assemble the square nut onto the peg",
    ],
    "Stack": [
        "stack the red cube on top of the green cube",
        "place the red block on the green block",
    ],
}


class RobosuiteVLAWrapper:
    """
    Wraps a robosuite environment to provide a VLA-friendly interface.

    Observations:
        - image: (H, W, 3) uint8 RGB from agentview camera
        - wrist_image: (H, W, 3) uint8 RGB from wrist camera (optional)
        - instruction: str, natural language task instruction
        - gripper_pos: (3,) float, end-effector position
        - gripper_quat: (4,) float, end-effector quaternion
        - gripper_state: (2,) float, gripper finger positions

    Actions:
        - (7,) float: [dx, dy, dz, dax, day, daz, gripper] — delta EEF pose + gripper
    """

    def __init__(
        self,
        env_name="Lift",
        robots="Panda",
        image_size=256,
        camera_names=None,
        control_freq=10,
        horizon=200,
        render_offscreen=True,
    ):
        if not HAS_ROBOSUITE:
            raise ImportError(
                "robosuite is required. Install via: pip install robosuite"
            )

        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]

        self.env_name = env_name
        self.image_size = image_size
        self.camera_names = camera_names
        self.horizon = horizon

        # Create robosuite environment
        self.env = suite.make(
            env_name=env_name,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=render_offscreen,
            use_camera_obs=render_offscreen,
            camera_names=camera_names,
            camera_heights=image_size,
            camera_widths=image_size,
            reward_shaping=True,
            control_freq=control_freq,
            horizon=horizon,
        )

        # Store instructions for this task
        self.instructions = TASK_INSTRUCTIONS.get(env_name, ["complete the task"])
        self.current_instruction = None
        self._step_count = 0

        # Get action dimension
        self.action_dim = self.env.action_dim
        print(f"[RobosuiteVLAWrapper] Env: {env_name}, Robot: {robots}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Camera: {camera_names}, resolution: {image_size}x{image_size}")
        print(f"  Control freq: {control_freq}Hz, Horizon: {horizon}")

    def reset(self, instruction=None):
        """
        Reset environment and return initial observation.

        Args:
            instruction: Optional specific instruction string.
                        If None, randomly samples from templates.

        Returns:
            obs_dict with keys: image, instruction, gripper_pos, etc.
        """
        raw_obs = self.env.reset()
        self._step_count = 0

        if instruction is not None:
            self.current_instruction = instruction
        else:
            self.current_instruction = np.random.choice(self.instructions)

        return self._process_obs(raw_obs)

    def step(self, action):
        """
        Step the environment with a (7,) action vector.

        Args:
            action: np.array of shape (action_dim,) — delta EEF + gripper

        Returns:
            obs_dict, reward, done, info
        """
        action = np.clip(action, -1.0, 1.0)
        raw_obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        obs = self._process_obs(raw_obs)

        # Add success info
        info["success"] = self.env._check_success() if hasattr(self.env, '_check_success') else False
        info["step"] = self._step_count

        return obs, reward, done, info

    def _process_obs(self, raw_obs):
        """Convert raw robosuite obs to VLA-friendly format."""
        obs = {
            "instruction": self.current_instruction,
        }

        # Main camera image
        main_cam = self.camera_names[0]
        img_key = f"{main_cam}_image"
        if img_key in raw_obs:
            # robosuite images are (H, W, 3) uint8, but may be flipped
            img = raw_obs[img_key]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            obs["image"] = img[::-1]  # flip vertically (robosuite convention)
        else:
            obs["image"] = np.zeros(
                (self.image_size, self.image_size, 3), dtype=np.uint8
            )

        # Wrist camera image (optional)
        if len(self.camera_names) > 1:
            wrist_cam = self.camera_names[1]
            wrist_key = f"{wrist_cam}_image"
            if wrist_key in raw_obs:
                wrist_img = raw_obs[wrist_key]
                if wrist_img.dtype != np.uint8:
                    wrist_img = (wrist_img * 255).astype(np.uint8)
                obs["wrist_image"] = wrist_img[::-1]

        # Robot proprioception
        if "robot0_eef_pos" in raw_obs:
            obs["gripper_pos"] = raw_obs["robot0_eef_pos"].copy()
        if "robot0_eef_quat" in raw_obs:
            obs["gripper_quat"] = raw_obs["robot0_eef_quat"].copy()
        if "robot0_gripper_qpos" in raw_obs:
            obs["gripper_state"] = raw_obs["robot0_gripper_qpos"].copy()

        return obs

    def render(self, camera_name=None):
        """Render a single frame from the specified camera."""
        if camera_name is None:
            camera_name = self.camera_names[0]

        raw_obs = self.env._get_observations()
        img_key = f"{camera_name}_image"
        if img_key in raw_obs:
            img = raw_obs[img_key]
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img[::-1]
        return None

    def close(self):
        """Close the environment."""
        self.env.close()

    @property
    def observation_space_info(self):
        """Return info about the observation space for documentation."""
        return {
            "image": f"({self.image_size}, {self.image_size}, 3) uint8",
            "instruction": "str",
            "gripper_pos": "(3,) float64",
            "gripper_quat": "(4,) float64",
            "gripper_state": "(2,) float64",
        }


# ─── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing RobosuiteVLAWrapper...")
    print("=" * 60)

    env = RobosuiteVLAWrapper(
        env_name="Lift",
        robots="Panda",
        image_size=128,  # smaller for quick test
        control_freq=10,
    )

    obs = env.reset()
    print(f"\nObservation keys: {list(obs.keys())}")
    print(f"  Image shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
    print(f"  Instruction: '{obs['instruction']}'")
    if "gripper_pos" in obs:
        print(f"  Gripper pos: {obs['gripper_pos']}")

    # Run 10 random steps
    print("\nRunning 10 random steps...")
    for i in range(10):
        action = np.random.uniform(-0.5, 0.5, env.action_dim)
        obs, reward, done, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.3f}, done={done}, "
              f"success={info.get('success', False)}")

    env.close()
    print("\n✅ RobosuiteVLAWrapper test passed!")
