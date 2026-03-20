"""
Franka Panda Grasping Environment
==================================
A 7-DOF Franka Emika Panda arm with parallel gripper for tabletop grasping.

Based on the official MuJoCo Menagerie model by Google DeepMind.
Uses Jacobian-based resolved-rate IK for Cartesian velocity control,
converting (dx, dy, dz, gripper) actions to joint-space commands.

Features:
- Realistic 7-DOF Franka Panda arm geometry (based on menagerie)
- 2-finger parallel gripper
- 3 colored objects with randomized positions
- 12 language-conditioned instructions
- 3 camera views (front, top-down, side)
- Cartesian action interface: (dx, dy, dz, gripper) → joint deltas via Jacobian IK
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


class FrankaGraspEnv:
    """
    Franka Panda tabletop grasping environment.

    Action space: (7,) float32 (also accepts (4,) for backward compatibility)
        [dx, dy, dz, dax, day, daz, gripper] in [-1, 1]
        dx/dy/dz: end-effector Cartesian velocity (scaled by 0.015m/step)
        dax/day/daz: end-effector angular velocity (scaled by 0.05rad/step)
        gripper: >0 close, <0 open

        If (4,) action is provided: [dx, dy, dz, gripper] — rotation is set to 0.

    Observation:
        image: (H, W, 3) uint8 — camera RGB
        instruction: str — language goal
        gripper_pos: (3,) — end-effector 3D position
        target_pos: (3,) — target object 3D position
    """

    # ─── Self-contained MJCF model ─────────────────────────────────
    # Simplified Franka Panda based on MuJoCo Menagerie geometry.
    # Uses accurate DH parameters and Panda joint limits.
    XML = """
    <mujoco model="franka_panda_grasp">
      <compiler angle="radian" autolimits="true"/>
      <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"
              cone="elliptic" impratio="10"/>

      <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <quality shadowsize="4096"/>
        <map znear="0.001"/>
      </visual>

      <default>
        <default class="panda">
          <joint damping="100" armature="0.1"/>
          <position kp="500" kv="50"/>
          <geom type="capsule" condim="4" friction="1 0.5 0.01" solref="0.01 1"/>
        </default>
        <default class="finger">
          <joint damping="10" armature="0.01"/>
          <position kp="100" kv="10"/>
        </default>
      </default>

      <asset>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1=".92 .90 .88" rgb2=".78 .76 .74" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
        <material name="panda_white" rgba="0.95 0.95 0.95 1" specular="0.5" shininess="0.6"/>
        <material name="panda_dark" rgba="0.2 0.2 0.22 1" specular="0.3" shininess="0.4"/>
        <material name="panda_hand" rgba="0.85 0.85 0.87 1" specular="0.4" shininess="0.5"/>
        <material name="finger_pad" rgba="0.1 0.1 0.12 1" specular="0.2" shininess="0.3"/>
        <material name="red" rgba="0.9 0.15 0.12 1" specular="0.4" shininess="0.6"/>
        <material name="blue" rgba="0.12 0.28 0.92 1" specular="0.4" shininess="0.6"/>
        <material name="green" rgba="0.15 0.78 0.22 1" specular="0.4" shininess="0.6"/>
        <material name="table_wood" rgba="0.55 0.42 0.32 1" specular="0.2" shininess="0.3"/>
      </asset>

      <worldbody>
        <!-- Ground -->
        <geom type="plane" size="2 2 0.01" material="grid"/>

        <!-- Table -->
        <body name="table" pos="0.25 0 0.2">
          <geom type="box" size="0.45 0.4 0.025" material="table_wood" mass="20"
                contype="1" conaffinity="1"/>
          <!-- Legs (from table bottom to ground) -->
          <geom type="cylinder" size="0.025 0.1" pos="-0.40 -0.35 -0.1"
                rgba="0.45 0.35 0.28 1" contype="0" conaffinity="0"/>
          <geom type="cylinder" size="0.025 0.1" pos="0.40 -0.35 -0.1"
                rgba="0.45 0.35 0.28 1" contype="0" conaffinity="0"/>
          <geom type="cylinder" size="0.025 0.1" pos="-0.40 0.35 -0.1"
                rgba="0.45 0.35 0.28 1" contype="0" conaffinity="0"/>
          <geom type="cylinder" size="0.025 0.1" pos="0.40 0.35 -0.1"
                rgba="0.45 0.35 0.28 1" contype="0" conaffinity="0"/>
        </body>

        <!-- Franka Panda Robot -->
        <body name="panda_base" pos="0 0 0.225" childclass="panda">
          <!-- Base platform -->
          <geom type="cylinder" size="0.06 0.02" material="panda_dark" mass="5"
                contype="0" conaffinity="0"/>

          <!-- Link 0 (base) -->
          <body name="link0" pos="0 0 0.02">
            <geom type="cylinder" size="0.06 0.05" pos="0 0 0.05" material="panda_white"
                  contype="0" conaffinity="0" mass="0.1"/>

            <!-- Link 1 -->
            <body name="link1" pos="0 0 0.333">
              <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.06"
                    material="panda_white" mass="4.97" contype="0" conaffinity="0"/>

              <!-- Link 2 -->
              <body name="link2" pos="0 0 0" quat="0.707 -0.707 0 0">
                <joint name="joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
                <geom type="capsule" fromto="0 0 0 0 -0.316 0" size="0.06"
                      material="panda_white" mass="0.646" contype="0" conaffinity="0"/>

                <!-- Link 3 -->
                <body name="link3" pos="0 -0.316 0" quat="0.707 0.707 0 0">
                  <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                  <geom type="capsule" fromto="0 0 -0.06 0.0825 0 0" size="0.05"
                        material="panda_white" mass="3.228"
                        contype="0" conaffinity="0"/>

                  <!-- Link 4 -->
                  <body name="link4" pos="0.0825 0 0" quat="0.707 0.707 0 0">
                    <joint name="joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
                    <geom type="capsule" fromto="0 0 0 -0.0825 0.384 0" size="0.05"
                          material="panda_white" mass="3.587"
                          contype="0" conaffinity="0"/>

                    <!-- Link 5 -->
                    <body name="link5" pos="-0.0825 0.384 0" quat="0.707 -0.707 0 0">
                      <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                      <geom type="capsule" fromto="0 0 -0.12 0 0 0" size="0.05"
                            material="panda_white" mass="1.225"
                            contype="0" conaffinity="0"/>

                      <!-- Link 6 -->
                      <body name="link6" pos="0 0 0" quat="0.707 0.707 0 0">
                        <joint name="joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.04"
                              material="panda_dark" mass="0.8"
                              contype="0" conaffinity="0"/>
                        <geom type="capsule" fromto="0.088 0 0 0.088 0 -0.12" size="0.04"
                              material="panda_white" mass="0.8"
                              contype="0" conaffinity="0"/>

                        <!-- Link 7 -->
                        <body name="link7" pos="0.088 0 0" quat="0.707 0.707 0 0">
                          <joint name="joint7" type="hinge" axis="0 0 1"
                                 range="-2.8973 2.8973"/>
                          <geom type="cylinder" size="0.04 0.01" material="panda_dark"
                                mass="0.5" contype="0" conaffinity="0"/>

                          <!-- Flange + Hand -->
                          <body name="hand" pos="0 0 0.107" quat="0.924 0 0 -0.383">
                            <geom type="box" size="0.025 0.06 0.02" pos="0 0 0.015"
                                  material="panda_hand"
                                  contype="0" conaffinity="0" mass="0"/>
                            <geom type="box" size="0.025 0.06 0.02" pos="0 0 0.015"
                                  mass="0.73"/>

                            <!-- EE site for IK target -->
                            <site name="ee_site" pos="0 0 0.105" size="0.01"
                                  rgba="1 0 0 0.5" group="5"/>

                            <!-- Left finger -->
                            <body name="left_finger" pos="0 0 0.0584">
                              <joint name="finger_left" type="slide" axis="0 1 0"
                                     range="0.001 0.04" class="finger"/>
                              <geom type="box" size="0.01 0.008 0.03"
                                    pos="0 0.015 0.025" material="panda_hand"
                                    contype="0" conaffinity="0" mass="0"/>
                              <geom type="box" size="0.01 0.008 0.03"
                                    pos="0 0.015 0.025" mass="0.015"/>
                              <!-- Finger pad -->
                              <geom type="box" size="0.012 0.003 0.022"
                                    pos="0 0.005 0.032" material="finger_pad"/>
                            </body>

                            <!-- Right finger -->
                            <body name="right_finger" pos="0 0 0.0584">
                              <joint name="finger_right" type="slide" axis="0 -1 0"
                                     range="0.001 0.04" class="finger"/>
                              <geom type="box" size="0.01 0.008 0.03"
                                    pos="0 -0.015 0.025" material="panda_hand"
                                    contype="0" conaffinity="0" mass="0"/>
                              <geom type="box" size="0.01 0.008 0.03"
                                    pos="0 -0.015 0.025" mass="0.015"/>
                              <!-- Finger pad -->
                              <geom type="box" size="0.012 0.003 0.022"
                                    pos="0 -0.005 0.032" material="finger_pad"/>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- Objects -->
        <body name="red_cube" pos="0.5 0.1 0.24">
          <joint name="red_cube_jnt" type="free"/>
          <geom type="box" size="0.02 0.02 0.02" material="red" mass="0.05"
                contype="1" conaffinity="1"/>
        </body>
        <body name="blue_cube" pos="0.4 -0.15 0.24">
          <joint name="blue_cube_jnt" type="free"/>
          <geom type="box" size="0.02 0.02 0.02" material="blue" mass="0.05"
                contype="1" conaffinity="1"/>
        </body>
        <body name="green_cube" pos="0.55 0.0 0.24">
          <joint name="green_cube_jnt" type="free"/>
          <geom type="box" size="0.015 0.015 0.015" material="green" mass="0.03"
                contype="1" conaffinity="1"/>
        </body>

        <!-- Cameras -->
        <camera name="frontview" pos="0.5 -0.9 0.8" xyaxes="1 0 0 0 0.5 0.85" fovy="45"/>
        <camera name="topdown" pos="0.4 0 1.4" xyaxes="1 0 0 0 1 0" fovy="50"/>
        <camera name="sideview" pos="1.2 0 0.7" xyaxes="0 1 0 -0.3 0 0.95" fovy="50"/>

        <!-- Lighting -->
        <light pos="0.5 -0.5 1.8" dir="0 0.3 -1" diffuse="0.8 0.78 0.75" specular="0.4 0.4 0.4"/>
        <light pos="0.3 0.7 1.5" dir="0 -0.3 -1" diffuse="0.5 0.5 0.52" specular="0.2 0.2 0.2"/>
        <light pos="-0.5 0 2.0" dir="0.3 0 -1" diffuse="0.35 0.35 0.38" specular="0.1 0.1 0.1"/>
      </worldbody>

      <actuator>
        <position name="act_j1" joint="joint1" class="panda" ctrlrange="-2.8973 2.8973"/>
        <position name="act_j2" joint="joint2" class="panda" ctrlrange="-1.7628 1.7628"/>
        <position name="act_j3" joint="joint3" class="panda" ctrlrange="-2.8973 2.8973"/>
        <position name="act_j4" joint="joint4" class="panda" ctrlrange="-3.0718 -0.0698"/>
        <position name="act_j5" joint="joint5" class="panda" ctrlrange="-2.8973 2.8973"/>
        <position name="act_j6" joint="joint6" class="panda" ctrlrange="-0.0175 3.7525"/>
        <position name="act_j7" joint="joint7" class="panda" ctrlrange="-2.8973 2.8973"/>
        <position name="act_finger_l" joint="finger_left" class="finger" ctrlrange="0.001 0.04"/>
        <position name="act_finger_r" joint="finger_right" class="finger" ctrlrange="0.001 0.04"/>
      </actuator>
    </mujoco>
    """

    # Home joint configuration (arm pointing forward and down, ready to grasp)
    HOME_QPOS = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785])

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
        self.renderer = create_renderer(
            mujoco,
            self.model,
            height=image_size,
            width=image_size,
        )

        # Cache body/joint/actuator IDs
        self._body_ids = {}
        for n in self.OBJECTS:
            self._body_ids[n] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        self._ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")

        # Object joint address (free joints, 7 DOF each: pos3 + quat4)
        self._obj_qpos_adr = {}
        for n in self.OBJECTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_jnt")
            self._obj_qpos_adr[n] = self.model.jnt_qposadr[jid]

        # Arm joint IDs (7 joints)
        self._arm_joint_ids = []
        for i in range(1, 8):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            self._arm_joint_ids.append(jid)

        # Actuator IDs
        self._act_ids = {}
        for name in ["act_j1","act_j2","act_j3","act_j4","act_j5","act_j6","act_j7",
                      "act_finger_l","act_finger_r"]:
            self._act_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # State
        self.target = None
        self.instruction = None
        self._step_count = 0
        self._max_steps = 150
        self._finger_target = 0.04  # open

        print(f"[FrankaGraspEnv] image={image_size}x{image_size}, camera={camera_name}")
        print(f"  7-DOF Franka Panda + parallel gripper")
        print(f"  Objects: {self.OBJECTS}")

    def _get_ee_pos(self):
        """Get end-effector position from site."""
        return self.data.site_xpos[self._ee_site_id].copy()

    def _get_ee_jac(self):
        """Get end-effector Jacobian (3×nv for position, 3×nv for rotation)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._ee_site_id)
        # Extract columns for arm joints only
        arm_jacp = np.zeros((3, 7))
        arm_jacr = np.zeros((3, 7))
        for i, jid in enumerate(self._arm_joint_ids):
            dof_adr = self.model.jnt_dofadr[jid]
            arm_jacp[:, i] = jacp[:, dof_adr]
            arm_jacr[:, i] = jacr[:, dof_adr]
        return arm_jacp, arm_jacr

    def _ik_step(self, dx_cart, dx_rot=None):
        """
        Resolved-rate IK: convert Cartesian + angular velocity to joint velocity.

        Uses damped least-squares (DLS) for stability:
            dq = J^T (J J^T + λ²I)^{-1} dx

        Args:
            dx_cart: (3,) Cartesian displacement [dx, dy, dz]
            dx_rot: (3,) angular displacement [dax, day, daz] or None

        Returns:
            dq: (7,) joint displacement
        """
        Jp, Jr = self._get_ee_jac()  # (3, 7) each
        lam = 0.05  # damping factor

        if dx_rot is not None and np.linalg.norm(dx_rot) > 1e-8:
            # Stack position + rotation Jacobians
            J = np.vstack([Jp, Jr])  # (6, 7)
            dx = np.concatenate([dx_cart, dx_rot])  # (6,)
            JJT = J @ J.T + lam**2 * np.eye(6)
        else:
            J = Jp  # (3, 7)
            dx = dx_cart
            JJT = J @ J.T + lam**2 * np.eye(3)

        dq = J.T @ np.linalg.solve(JJT, dx)
        return dq

    def reset(self, target_object=None, randomize=True):
        """Reset environment."""
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._finger_target = 0.04  # open

        # Set arm to home configuration
        for i, jid in enumerate(self._arm_joint_ids):
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = self.HOME_QPOS[i]

        # Set actuator targets to home
        for i in range(7):
            self.data.ctrl[self._act_ids[f"act_j{i+1}"]] = self.HOME_QPOS[i]
        self.data.ctrl[self._act_ids["act_finger_l"]] = self._finger_target
        self.data.ctrl[self._act_ids["act_finger_r"]] = self._finger_target

        # Randomize object positions
        if randomize:
            for n in self.OBJECTS:
                a = self._obj_qpos_adr[n]
                self.data.qpos[a] = np.random.uniform(0.35, 0.6)
                self.data.qpos[a+1] = np.random.uniform(-0.2, 0.2)
                self.data.qpos[a+2] = 0.25
                self.data.qpos[a+3] = 1.0
                self.data.qpos[a+4:a+7] = 0.0

        # Settle the simulation
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        # Select target
        self.target = target_object or np.random.choice(self.OBJECTS)
        self.instruction = np.random.choice(self.INSTRUCTIONS[self.target])
        return self._get_obs()

    def step(self, action):
        """
        Execute action.

        Args:
            action: (7,) [dx, dy, dz, dax, day, daz, gripper] in [-1, 1]
                    OR (4,) [dx, dy, dz, gripper] for backward compatibility

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

        # Convert Cartesian + rotation delta to joint delta via IK
        cart_delta = action[:3] * 0.015   # scale to meters
        rot_delta = action[3:6] * 0.05    # scale to radians
        dq = self._ik_step(cart_delta, rot_delta)

        # Update joint position targets
        for i in range(7):
            qadr = self.model.jnt_qposadr[self._arm_joint_ids[i]]
            current_q = self.data.qpos[qadr]
            target_q = current_q + dq[i]
            # Respect joint limits
            jnt_range = self.model.jnt_range[self._arm_joint_ids[i]]
            target_q = np.clip(target_q, jnt_range[0], jnt_range[1])
            self.data.ctrl[self._act_ids[f"act_j{i+1}"]] = target_q

        # Gripper
        self._finger_target = 0.001 if action[6] > 0 else 0.04
        self.data.ctrl[self._act_ids["act_finger_l"]] = self._finger_target
        self.data.ctrl[self._act_ids["act_finger_r"]] = self._finger_target

        # Simulate
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        # Observations and reward
        obs = self._get_obs()
        target_pos = self._get_object_pos(self.target)
        ee_pos = self._get_ee_pos()
        distance = np.linalg.norm(ee_pos[:2] - target_pos[:2])
        z_distance = abs(ee_pos[2] - target_pos[2])
        lifted = target_pos[2] > 0.35

        reward = -distance - z_distance + (10.0 if lifted else 0.0)
        done = lifted or self._step_count >= self._max_steps

        return obs, float(reward), done, {
            "success": lifted,
            "distance": distance,
            "step": self._step_count,
        }

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return {
            "image": self.renderer.render().copy(),
            "instruction": self.instruction,
            "gripper_pos": self._get_ee_pos(),
            "target_pos": self._get_object_pos(self.target),
        }

    def _get_object_pos(self, name):
        return self.data.xpos[self._body_ids[name]].copy()

    def get_all_object_positions(self):
        return {n: self._get_object_pos(n) for n in self.OBJECTS}

    def render_frame(self, camera_name=None):
        cam = camera_name or self.camera_name
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render().copy()

    def render_video(self, frames, save_path, fps=10):
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(save_path, save_all=True, append_images=imgs[1:],
                     duration=int(1000/fps), loop=0)
        print(f"Saved GIF: {save_path} ({len(frames)} frames, {fps} fps)")

    def close(self):
        self.renderer.close()


# ─── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Franka Panda Grasping Environment...")

    env = FrankaGraspEnv(image_size=256)
    obs = env.reset(target_object="red_cube")

    print(f"  Image: {obs['image'].shape}")
    print(f"  EE pos: {obs['gripper_pos']}")
    print(f"  Target: {obs['target_pos']}")
    print(f"  Instruction: {obs['instruction']}")

    # Save test renders
    for cam in ["frontview", "topdown", "sideview"]:
        frame = env.render_frame(camera_name=cam)
        Image.fromarray(frame).save(f"/tmp/franka_{cam}.png")
        print(f"  Saved /tmp/franka_{cam}.png")

    # Test a few steps with 7-DOF action
    for i in range(5):
        action = np.array([0.1, 0.0, -0.3, 0.0, 0.0, 0.0, -1.0])  # move forward+down, no rotation, open
        obs, r, done, info = env.step(action)
        print(f"  Step {i}: ee={obs['gripper_pos']}, r={r:.2f}, dist={info['distance']:.3f}")

    # Test backward-compatible 4-DOF action
    obs, r, done, info = env.step(np.array([0.1, 0.0, -0.3, -1.0]))
    print(f"  4-DOF compat: ee={obs['gripper_pos']}, r={r:.2f}")

    env.close()
    print("\n✅ Franka Panda env test passed!")
