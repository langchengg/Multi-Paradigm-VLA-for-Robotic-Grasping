"""
Simple Grasp Environment
========================
Custom MuJoCo grasping environment with language-conditioned goals.
Features a Panda-style gripper controlled via actuators (not mocap)
with colored cubes on a table.

This demonstrates the ability to design custom simulation environments
for VLA evaluation — a key differentiator for the project.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from envs._rendering import configure_headless_rendering, create_renderer
else:
    from ._rendering import configure_headless_rendering, create_renderer

configure_headless_rendering()

import mujoco
import numpy as np
from PIL import Image


class SimpleGraspEnv:
    """
    Custom MuJoCo tabletop grasping environment.

    Features:
    - Actuator-controlled 3-DOF Cartesian gripper (x, y, z) + 1-DOF gripper open/close
    - Multiple colored objects with free joints
    - Language-conditioned goals ("pick up the red cube")
    - Camera rendering for VLA visual input

    Observation space:
        image: (H, W, 3) uint8 — camera view
        instruction: str — language goal
        gripper_pos: (3,) — current gripper position
        target_pos: (3,) — current target object position

    Action space:
        (4,) float in [-1, 1]: [dx, dy, dz, gripper_cmd]
        gripper_cmd > 0 → close, < 0 → open
    """

    # ──────────────── MuJoCo XML Model ────────────────
    XML = """
    <mujoco model="simple_grasp">
      <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <quality shadowsize="2048"/>
      </visual>

      <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1=".95 .95 .95" rgb2=".8 .8 .8" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
        <material name="red" rgba="0.9 0.15 0.15 1" specular="0.3" shininess="0.5"/>
        <material name="blue" rgba="0.15 0.3 0.9 1" specular="0.3" shininess="0.5"/>
        <material name="green" rgba="0.15 0.8 0.2 1" specular="0.3" shininess="0.5"/>
        <material name="gripper_mat" rgba="0.45 0.45 0.5 0.9" specular="0.5"/>
        <material name="finger_mat" rgba="0.6 0.6 0.65 1" specular="0.5"/>
      </asset>

      <default>
        <joint damping="0.5"/>
        <geom condim="4" friction="1 0.5 0.01"/>
      </default>

      <worldbody>
        <!-- Ground plane -->
        <geom type="plane" size="1 1 0.01" material="grid"/>

        <!-- Table -->
        <body name="table" pos="0.5 0 0.2">
          <geom type="box" size="0.3 0.35 0.02" rgba="0.55 0.4 0.3 1" mass="10"/>
          <!-- Table legs -->
          <geom type="cylinder" size="0.02 0.1" pos="-0.25 -0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
          <geom type="cylinder" size="0.02 0.1" pos="0.25 -0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
          <geom type="cylinder" size="0.02 0.1" pos="-0.25 0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
          <geom type="cylinder" size="0.02 0.1" pos="0.25 0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
        </body>

        <!-- Graspable objects -->
        <body name="red_cube" pos="0.4 0.1 0.24">
          <joint name="red_cube_jnt" type="free"/>
          <geom type="box" size="0.02 0.02 0.02" material="red" mass="0.05"/>
        </body>

        <body name="blue_cube" pos="0.6 -0.1 0.24">
          <joint name="blue_cube_jnt" type="free"/>
          <geom type="box" size="0.02 0.02 0.02" material="blue" mass="0.05"/>
        </body>

        <body name="green_cube" pos="0.5 0.0 0.24">
          <joint name="green_cube_jnt" type="free"/>
          <geom type="box" size="0.015 0.015 0.015" material="green" mass="0.03"/>
        </body>

        <!-- Actuator-controlled gripper (3-DOF Cartesian + 1-DOF finger) -->
        <body name="gripper_base" pos="0.5 0 0.5">
          <!-- Cartesian slides -->
          <joint name="slide_x" type="slide" axis="1 0 0" range="-0.3 0.3" damping="20"/>
          <joint name="slide_y" type="slide" axis="0 1 0" range="-0.3 0.3" damping="20"/>
          <joint name="slide_z" type="slide" axis="0 0 1" range="-0.4 0.2" damping="20"/>
          <geom type="cylinder" size="0.015 0.03" material="gripper_mat" mass="0.5"/>

          <!-- Left finger -->
          <body name="left_finger" pos="0 0.015 -0.04">
            <joint name="finger_left" type="slide" axis="0 1 0" range="-0.005 0.025" damping="5"/>
            <geom type="box" size="0.008 0.003 0.025" material="finger_mat" mass="0.05"/>
          </body>

          <!-- Right finger -->
          <body name="right_finger" pos="0 -0.015 -0.04">
            <joint name="finger_right" type="slide" axis="0 -1 0" range="-0.005 0.025" damping="5"/>
            <geom type="box" size="0.008 0.003 0.025" material="finger_mat" mass="0.05"/>
          </body>
        </body>

        <!-- Cameras -->
        <camera name="frontview" pos="0.5 -0.8 0.7" xyaxes="1 0 0 0 0.6 0.8" fovy="45"/>
        <camera name="topdown" pos="0.5 0 1.2" xyaxes="1 0 0 0 1 0" fovy="45"/>
        <camera name="sideview" pos="1.0 0 0.5" xyaxes="0 1 0 0 0 1" fovy="50"/>

        <!-- Lighting -->
        <light pos="0.5 -0.5 1.5" dir="0 0.5 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
        <light pos="0.5 0.5 1.5" dir="0 -0.5 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
      </worldbody>

      <actuator>
        <!-- Cartesian position control -->
        <position name="act_x" joint="slide_x" kp="200" ctrlrange="-0.3 0.3"/>
        <position name="act_y" joint="slide_y" kp="200" ctrlrange="-0.3 0.3"/>
        <position name="act_z" joint="slide_z" kp="200" ctrlrange="-0.4 0.2"/>
        <!-- Gripper finger control (coupled) -->
        <position name="act_finger_l" joint="finger_left" kp="50" ctrlrange="-0.005 0.025"/>
        <position name="act_finger_r" joint="finger_right" kp="50" ctrlrange="-0.005 0.025"/>
      </actuator>
    </mujoco>
    """

    # ──────────────── Language Instructions ────────────────
    INSTRUCTIONS = {
        "red_cube": [
            "pick up the red cube",
            "grasp the red block",
            "grab the red object from the table",
            "lift the red cube",
        ],
        "blue_cube": [
            "pick up the blue cube",
            "grasp the blue block",
            "grab the blue object from the table",
            "lift the blue cube",
        ],
        "green_cube": [
            "pick up the green cube",
            "grasp the small green block",
            "grab the green object",
            "lift the green cube",
        ],
    }

    OBJECTS = ["red_cube", "blue_cube", "green_cube"]

    def __init__(self, image_size=256, camera_name="frontview"):
        self.model = mujoco.MjModel.from_xml_string(self.XML)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.camera_name = camera_name

        # Create renderer
        self.renderer = create_renderer(
            mujoco,
            self.model,
            height=image_size,
            width=image_size,
        )

        # Cache joint/body ids
        self._body_ids = {}
        for name in self.OBJECTS:
            self._body_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )

        self._gripper_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"
        )

        # Joint indices for each free-joint object (7 DOF each: pos3 + quat4)
        self._obj_qpos_adr = {}
        for name in self.OBJECTS:
            jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_jnt"
            )
            self._obj_qpos_adr[name] = self.model.jnt_qposadr[jnt_id]

        # Actuator indices
        self._act_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_x")
        self._act_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_y")
        self._act_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_z")
        self._act_fl = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_l")
        self._act_fr = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_finger_r")

        # State
        self.target = None
        self.instruction = None
        self._step_count = 0
        self._max_steps = 150

        # Gripper target position (for incremental control)
        self._gripper_target = np.array([0.0, 0.0, 0.0])
        self._finger_target = 0.025  # open

        print(f"[SimpleGraspEnv] image={image_size}x{image_size}, camera={camera_name}")
        print(f"  Objects: {self.OBJECTS}")

    def reset(self, target_object=None, randomize=True):
        """
        Reset environment.

        Args:
            target_object: which object to grasp. If None, random choice.
            randomize: if True, randomize object positions on table.

        Returns:
            obs_dict: {image, instruction, gripper_pos, target_pos}
        """
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Reset gripper to home position above table
        self._gripper_target = np.array([0.0, 0.0, 0.0])  # relative to base at (0.5, 0, 0.5)
        self._finger_target = 0.025  # open

        if randomize:
            for obj_name in self.OBJECTS:
                adr = self._obj_qpos_adr[obj_name]
                # Random position on table surface
                self.data.qpos[adr] = np.random.uniform(0.3, 0.7)  # x
                self.data.qpos[adr + 1] = np.random.uniform(-0.2, 0.2)  # y
                self.data.qpos[adr + 2] = 0.24  # z (on table)
                # Reset orientation to upright
                self.data.qpos[adr + 3] = 1.0  # qw
                self.data.qpos[adr + 4:adr + 7] = 0.0  # qx, qy, qz

        # Set actuator targets
        self.data.ctrl[self._act_x] = self._gripper_target[0]
        self.data.ctrl[self._act_y] = self._gripper_target[1]
        self.data.ctrl[self._act_z] = self._gripper_target[2]
        self.data.ctrl[self._act_fl] = self._finger_target
        self.data.ctrl[self._act_fr] = self._finger_target

        # Step to stabilize
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        # Select target and instruction
        if target_object is None:
            self.target = np.random.choice(self.OBJECTS)
        else:
            self.target = target_object
        self.instruction = np.random.choice(self.INSTRUCTIONS[self.target])

        return self._get_obs()

    def step(self, action):
        """
        Execute action.

        Args:
            action: (7,) array [dx, dy, dz, dax, day, daz, gripper_cmd]
                   OR (4,) array [dx, dy, dz, gripper_cmd] for backward compat
                   dx/dy/dz in [-1, 1] scaled to position deltas
                   dax/day/daz: ignored (3-DOF gripper has no rotation)
                   gripper_cmd > 0 → close, < 0 → open

        Returns:
            obs, reward, done, info
        """
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)
        self._step_count += 1

        # Backward compatibility: 4-DOF → 7-DOF
        if len(action) == 4:
            action = np.array([action[0], action[1], action[2],
                               0.0, 0.0, 0.0, action[3]])

        # Update gripper target position (incremental)
        scale = 0.02  # meters per unit action
        self._gripper_target[0] += action[0] * scale
        self._gripper_target[1] += action[1] * scale
        self._gripper_target[2] += action[2] * scale

        # Clamp to workspace
        self._gripper_target[0] = np.clip(self._gripper_target[0], -0.25, 0.25)
        self._gripper_target[1] = np.clip(self._gripper_target[1], -0.25, 0.25)
        self._gripper_target[2] = np.clip(self._gripper_target[2], -0.35, 0.15)

        # Gripper open/close (index 6 in 7-DOF)
        if action[6] > 0:
            self._finger_target = -0.005  # close (fingers move inward)
        else:
            self._finger_target = 0.025  # open

        # Set actuator controls
        self.data.ctrl[self._act_x] = self._gripper_target[0]
        self.data.ctrl[self._act_y] = self._gripper_target[1]
        self.data.ctrl[self._act_z] = self._gripper_target[2]
        self.data.ctrl[self._act_fl] = self._finger_target
        self.data.ctrl[self._act_fr] = self._finger_target

        # Simulate forward
        for _ in range(20):  # 20 substeps × 0.002s = 0.04s per step → 25 Hz
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        # Compute reward
        target_pos = self._get_object_pos(self.target)
        gripper_pos = self._get_gripper_pos()

        distance = np.linalg.norm(gripper_pos[:3] - target_pos)

        # Success: object lifted above 0.35 (table at 0.22, so 0.13m lift)
        lifted = target_pos[2] > 0.35

        reward = -distance
        if lifted:
            reward += 10.0

        done = lifted or self._step_count >= self._max_steps

        info = {
            "success": lifted,
            "distance": distance,
            "step": self._step_count,
            "target_height": target_pos[2],
        }

        return obs, reward, done, info

    def _get_obs(self):
        """Build observation dictionary."""
        self.renderer.update_scene(self.data, camera=self.camera_name)
        image = self.renderer.render()

        return {
            "image": image.copy(),
            "instruction": self.instruction,
            "gripper_pos": self._get_gripper_pos(),
            "target_pos": self._get_object_pos(self.target),
        }

    def _get_gripper_pos(self):
        """Get gripper end-effector position."""
        return self.data.xpos[self._gripper_body_id].copy()

    def _get_object_pos(self, name):
        """Get object center position."""
        return self.data.xpos[self._body_ids[name]].copy()

    def get_all_object_positions(self):
        """Get positions of all objects."""
        return {name: self._get_object_pos(name) for name in self.OBJECTS}

    def render_frame(self, camera_name=None):
        """Render a single frame from specified camera."""
        cam = camera_name or self.camera_name
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def render_video(self, frames, filename="grasp_demo.gif", fps=10):
        """Save frames as GIF."""
        images = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        images[0].save(
            filename,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )
        print(f"Saved GIF: {filename} ({len(frames)} frames, {fps} fps)")

    def close(self):
        """Clean up renderer."""
        self.renderer.close()


# ─── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing SimpleGraspEnv...")
    print("=" * 60)

    env = SimpleGraspEnv(image_size=128, camera_name="frontview")

    obs = env.reset(target_object="red_cube")
    print(f"\nObservation keys: {list(obs.keys())}")
    print(f"  Image shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
    print(f"  Instruction: '{obs['instruction']}'")
    print(f"  Gripper pos: {obs['gripper_pos']}")
    print(f"  Target pos: {obs['target_pos']}")

    # Save initial frame
    Image.fromarray(obs["image"]).save("/tmp/simple_grasp_test.png")
    print(f"\nSaved test frame to /tmp/simple_grasp_test.png")

    # Run a few steps
    print("\nRunning 20 random steps...")
    frames = [obs["image"]]
    for i in range(20):
        action = np.random.uniform(-0.5, 0.5, 4)
        obs, reward, done, info = env.step(action)
        frames.append(obs["image"])
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}: reward={reward:.3f}, dist={info['distance']:.3f}, "
                  f"height={info['target_height']:.3f}")

    # Save test GIF
    env.render_video(frames, "/tmp/simple_grasp_test.gif", fps=5)

    env.close()
    print("\n✅ SimpleGraspEnv test passed!")
