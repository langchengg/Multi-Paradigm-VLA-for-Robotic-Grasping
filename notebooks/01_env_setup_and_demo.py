#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  Kaggle Notebook 1: MuJoCo Environment Setup + Demo Collection
═══════════════════════════════════════════════════════════════════

Run this on Kaggle (CPU or GPU instance) to:
1. Install MuJoCo + dependencies
2. Set up off-screen rendering (osmesa for Linux)
3. Create custom grasping environment
4. Collect 100 expert demonstrations
5. Save demos as Kaggle Dataset for training

Output: data/demos/*.npz (upload as Kaggle Dataset)

⏱️ Estimated time: 5-10 minutes
💾 Output size: ~200 MB (100 demos with 256×256 images)
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install Dependencies
# ═══════════════════════════════════════════════════════════════

import subprocess
import sys

def install_packages():
    """Install MuJoCo and dependencies for Kaggle."""
    packages = [
        "mujoco>=3.0.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "imageio>=2.30.0",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # Install osmesa for headless rendering on Linux (Kaggle)
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq "
        "libgl1-mesa-glx libgl1-mesa-dev libosmesa6-dev libglew-dev patchelf",
        shell=True, capture_output=True
    )
    print("✅ Packages installed")

install_packages()

# ═══════════════════════════════════════════════════════════════
# Cell 2: Configure Rendering
# ═══════════════════════════════════════════════════════════════

import os

# CRITICAL for Kaggle: Use osmesa (software rendering)
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import mujoco
import numpy as np
from PIL import Image

print(f"✅ MuJoCo {mujoco.__version__} loaded")
print(f"   Rendering backend: {os.environ.get('MUJOCO_GL', 'default')}")

# ═══════════════════════════════════════════════════════════════
# Cell 3: Define Custom Grasping Environment
# ═══════════════════════════════════════════════════════════════

class SimpleGraspEnv:
    """
    Custom MuJoCo tabletop grasping environment for VLA training.

    Features:
    - Actuator-controlled 3-DOF gripper + 2-finger grasp
    - 3 colored objects (red, blue, green cubes)
    - Language-conditioned goals
    - Off-screen camera rendering
    """

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
        <geom type="plane" size="1 1 0.01" material="grid"/>
        <body name="table" pos="0.5 0 0.2">
          <geom type="box" size="0.3 0.35 0.02" rgba="0.55 0.4 0.3 1" mass="10"/>
          <geom type="cylinder" size="0.02 0.1" pos="-0.25 -0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
          <geom type="cylinder" size="0.02 0.1" pos="0.25 -0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
          <geom type="cylinder" size="0.02 0.1" pos="-0.25 0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
          <geom type="cylinder" size="0.02 0.1" pos="0.25 0.3 -0.12" rgba="0.4 0.3 0.25 1"/>
        </body>
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
        <body name="gripper_base" pos="0.5 0 0.5">
          <joint name="slide_x" type="slide" axis="1 0 0" range="-0.3 0.3" damping="20"/>
          <joint name="slide_y" type="slide" axis="0 1 0" range="-0.3 0.3" damping="20"/>
          <joint name="slide_z" type="slide" axis="0 0 1" range="-0.4 0.2" damping="20"/>
          <geom type="cylinder" size="0.015 0.03" material="gripper_mat" mass="0.5"/>
          <body name="left_finger" pos="0 0.015 -0.04">
            <joint name="finger_left" type="slide" axis="0 1 0" range="-0.005 0.025" damping="5"/>
            <geom type="box" size="0.008 0.003 0.025" material="finger_mat" mass="0.05"/>
          </body>
          <body name="right_finger" pos="0 -0.015 -0.04">
            <joint name="finger_right" type="slide" axis="0 -1 0" range="-0.005 0.025" damping="5"/>
            <geom type="box" size="0.008 0.003 0.025" material="finger_mat" mass="0.05"/>
          </body>
        </body>
        <camera name="frontview" pos="0.5 -0.8 0.7" xyaxes="1 0 0 0 0.6 0.8" fovy="45"/>
        <camera name="topdown" pos="0.5 0 1.2" xyaxes="1 0 0 0 1 0" fovy="45"/>
        <camera name="sideview" pos="1.0 0 0.5" xyaxes="0 1 0 0 0 1" fovy="50"/>
        <light pos="0.5 -0.5 1.5" dir="0 0.5 -1" diffuse="0.7 0.7 0.7" specular="0.3 0.3 0.3"/>
        <light pos="0.5 0.5 1.5" dir="0 -0.5 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
      </worldbody>
      <actuator>
        <position name="act_x" joint="slide_x" kp="200" ctrlrange="-0.3 0.3"/>
        <position name="act_y" joint="slide_y" kp="200" ctrlrange="-0.3 0.3"/>
        <position name="act_z" joint="slide_z" kp="200" ctrlrange="-0.4 0.2"/>
        <position name="act_finger_l" joint="finger_left" kp="50" ctrlrange="-0.005 0.025"/>
        <position name="act_finger_r" joint="finger_right" kp="50" ctrlrange="-0.005 0.025"/>
      </actuator>
    </mujoco>
    """

    INSTRUCTIONS = {
        "red_cube": ["pick up the red cube", "grasp the red block",
                     "grab the red object from the table", "lift the red cube"],
        "blue_cube": ["pick up the blue cube", "grasp the blue block",
                      "grab the blue object from the table", "lift the blue cube"],
        "green_cube": ["pick up the green cube", "grasp the small green block",
                       "grab the green object", "lift the green cube"],
    }
    OBJECTS = ["red_cube", "blue_cube", "green_cube"]

    def __init__(self, image_size=256, camera_name="frontview"):
        self.model = mujoco.MjModel.from_xml_string(self.XML)
        self.data = mujoco.MjData(self.model)
        self.image_size = image_size
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)

        self._body_ids = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.OBJECTS}
        self._gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
        self._obj_qpos_adr = {}
        for n in self.OBJECTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_jnt")
            self._obj_qpos_adr[n] = self.model.jnt_qposadr[jid]

        self._act_ids = {k: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, k)
                         for k in ["act_x", "act_y", "act_z", "act_finger_l", "act_finger_r"]}

        self._gripper_target = np.zeros(3)
        self._finger_target = 0.025
        self.target = None
        self.instruction = None
        self._step_count = 0
        self._max_steps = 150

    def reset(self, target_object=None, randomize=True):
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._gripper_target = np.zeros(3)
        self._finger_target = 0.025

        if randomize:
            for n in self.OBJECTS:
                a = self._obj_qpos_adr[n]
                self.data.qpos[a] = np.random.uniform(0.3, 0.7)
                self.data.qpos[a+1] = np.random.uniform(-0.2, 0.2)
                self.data.qpos[a+2] = 0.24
                self.data.qpos[a+3] = 1.0
                self.data.qpos[a+4:a+7] = 0.0

        for k in ["act_x", "act_y", "act_z"]:
            self.data.ctrl[self._act_ids[k]] = 0.0
        self.data.ctrl[self._act_ids["act_finger_l"]] = self._finger_target
        self.data.ctrl[self._act_ids["act_finger_r"]] = self._finger_target

        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self.target = target_object or np.random.choice(self.OBJECTS)
        self.instruction = np.random.choice(self.INSTRUCTIONS[self.target])
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._step_count += 1
        scale = 0.02
        self._gripper_target += action[:3] * scale
        self._gripper_target = np.clip(self._gripper_target, [-0.25,-0.25,-0.35], [0.25,0.25,0.15])
        self._finger_target = -0.005 if action[3] > 0 else 0.025

        self.data.ctrl[self._act_ids["act_x"]] = self._gripper_target[0]
        self.data.ctrl[self._act_ids["act_y"]] = self._gripper_target[1]
        self.data.ctrl[self._act_ids["act_z"]] = self._gripper_target[2]
        self.data.ctrl[self._act_ids["act_finger_l"]] = self._finger_target
        self.data.ctrl[self._act_ids["act_finger_r"]] = self._finger_target

        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        target_pos = self._get_object_pos(self.target)
        gripper_pos = self.data.xpos[self._gripper_body_id]
        distance = np.linalg.norm(gripper_pos - target_pos)
        lifted = target_pos[2] > 0.35
        reward = -distance + (10.0 if lifted else 0.0)
        done = lifted or self._step_count >= self._max_steps
        return obs, reward, done, {"success": lifted, "distance": distance, "step": self._step_count}

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return {
            "image": self.renderer.render().copy(),
            "instruction": self.instruction,
            "gripper_pos": self.data.xpos[self._gripper_body_id].copy(),
            "target_pos": self._get_object_pos(self.target),
        }

    def _get_object_pos(self, name):
        return self.data.xpos[self._body_ids[name]].copy()

    def close(self):
        self.renderer.close()


print("✅ SimpleGraspEnv defined")

# Test environment
env = SimpleGraspEnv(image_size=256)
obs = env.reset(target_object="red_cube")
print(f"   Image: {obs['image'].shape}, Instruction: '{obs['instruction']}'")

# ═══════════════════════════════════════════════════════════════
# Cell 4: Scripted Expert Policy
# ═══════════════════════════════════════════════════════════════

def scripted_grasp_policy(obs, phase, phase_step, target_pos):
    """4-phase grasping: approach → descend → grasp → lift."""
    gripper_pos = obs["gripper_pos"]
    action = np.zeros(4)

    if phase == 0:
        goal = target_pos.copy(); goal[2] += 0.15
        d = goal - gripper_pos
        action[:3] = d * 5.0; action[3] = -1.0
        if np.linalg.norm(d) < 0.02 or phase_step > 30:
            return action, 1, 0
    elif phase == 1:
        goal = target_pos.copy(); goal[2] += 0.02
        d = goal - gripper_pos
        action[:3] = d * 4.0; action[3] = -1.0
        if np.linalg.norm(d) < 0.015 or phase_step > 25:
            return action, 2, 0
    elif phase == 2:
        action[3] = 1.0
        if phase_step > 10:
            return action, 3, 0
    else:
        goal = gripper_pos.copy(); goal[2] = 0.6
        d = goal - gripper_pos
        action[:3] = d * 3.0; action[3] = 1.0

    return np.clip(action, -1, 1), phase, phase_step + 1

# ═══════════════════════════════════════════════════════════════
# Cell 5: Collect Demonstrations
# ═══════════════════════════════════════════════════════════════

NUM_DEMOS = 100
IMAGE_SIZE = 256
SAVE_DIR = "/kaggle/working/demos"
os.makedirs(SAVE_DIR, exist_ok=True)

env = SimpleGraspEnv(image_size=IMAGE_SIZE)
success_count = 0

for i in range(NUM_DEMOS):
    target = np.random.choice(env.OBJECTS)
    obs = env.reset(target_object=target)
    target_pos = obs["target_pos"].copy()

    images, actions, rewards = [], [], []
    phase, phase_step = 0, 0

    for step in range(150):
        images.append(obs["image"])
        action, phase, phase_step = scripted_grasp_policy(obs, phase, phase_step, target_pos)
        action[:3] += np.random.normal(0, 0.03, 3)  # noise augmentation
        action = np.clip(action, -1, 1)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break

    if info["success"]:
        success_count += 1

    np.savez_compressed(
        os.path.join(SAVE_DIR, f"demo_{i:04d}.npz"),
        images=np.array(images),
        actions=np.array(actions),
        instruction=obs["instruction"],
        target_object=target,
        success=info["success"],
    )

    if (i+1) % 20 == 0:
        print(f"  Demo {i+1}/{NUM_DEMOS} | Success: {success_count}/{i+1} ({success_count/(i+1):.0%})")

env.close()
print(f"\n✅ Collected {NUM_DEMOS} demos → {SAVE_DIR}")
print(f"   Success rate: {success_count/NUM_DEMOS:.0%}")
print(f"   Upload this folder as a Kaggle Dataset for Notebook 2!")

# ═══════════════════════════════════════════════════════════════
# Cell 6: Generate Sample GIF
# ═══════════════════════════════════════════════════════════════

# Load a successful demo and save as GIF
for i in range(NUM_DEMOS):
    data = np.load(os.path.join(SAVE_DIR, f"demo_{i:04d}.npz"), allow_pickle=True)
    if data["success"]:
        frames = data["images"][::3]  # subsample
        imgs = [Image.fromarray(f) for f in frames]
        gif_path = "/kaggle/working/expert_demo.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"✅ Saved sample GIF: {gif_path} ({len(frames)} frames)")
        break

print("\n" + "="*60)
print("📋 Next: Run Notebook 2 to fine-tune OpenVLA on this data")
print("="*60)
