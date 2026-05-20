from __future__ import annotations

import numpy as np

from envs.franka_grasp_env import FrankaGraspEnv


class SyntheticFrankaGraspEnv(FrankaGraspEnv):
    """Target benchmark wrapper around the existing MuJoCo Franka Panda env.

    The parent environment already implements the important control invariant:
    7D end-effector delta actions are converted through Jacobian IK into MuJoCo
    joint position targets, rather than being treated as joint torques.
    """

    def __init__(self, image_size: int = 128, camera_name: str = "frontview"):
        super().__init__(image_size=image_size, camera_name=camera_name)
        self._finger_joint_ids = [
            self._joint_id("finger_left"),
            self._joint_id("finger_right"),
        ]
        self._object_geom_ids = {
            name: self.model.body_geomadr[self._body_ids[name]]
            for name in self.OBJECTS
        }

    def _joint_id(self, name: str) -> int:
        import mujoco

        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def reset(self, target_object=None, randomize: bool = True):
        obs = super().reset(target_object=target_object, randomize=randomize)
        if randomize:
            self._randomize_object_sizes()
        return self._get_obs() if randomize else obs

    def _randomize_object_sizes(self) -> None:
        """Apply mild per-episode size randomization while preserving color semantics."""
        import mujoco

        for name, geom_id in self._object_geom_ids.items():
            base = 0.015 if name == "green_cube" else 0.02
            size = np.random.uniform(base * 0.85, base * 1.15)
            self.model.geom_size[geom_id, :3] = size
        mujoco.mj_forward(self.model, self.data)

    def _joint_qpos(self, joint_ids: list[int]) -> np.ndarray:
        values = []
        for jid in joint_ids:
            qadr = self.model.jnt_qposadr[jid]
            values.append(self.data.qpos[qadr])
        return np.asarray(values, dtype=np.float32)

    def _joint_qvel(self, joint_ids: list[int]) -> np.ndarray:
        values = []
        for jid in joint_ids:
            dadr = self.model.jnt_dofadr[jid]
            values.append(self.data.qvel[dadr])
        return np.asarray(values, dtype=np.float32)

    def get_robot_state(self) -> dict:
        joint_ids = self._arm_joint_ids + self._finger_joint_ids
        finger_qpos = self._joint_qpos(self._finger_joint_ids)
        return {
            "eef_pos": self._get_ee_pos().astype(np.float32),
            "eef_quat": self.data.xquat[self._hand_body_id].copy().astype(np.float32),
            "gripper": np.asarray([finger_qpos.sum()], dtype=np.float32),
            "qpos": self._joint_qpos(joint_ids),
            "qvel": self._joint_qvel(joint_ids),
        }

    def get_object_state(self, name: str | None = None) -> dict:
        name = name or self.target
        qadr = self._obj_qpos_adr[name]
        return {
            "target_pos": self.data.qpos[qadr:qadr + 3].copy().astype(np.float32),
            "target_quat": self.data.qpos[qadr + 3:qadr + 7].copy().astype(np.float32),
            "target_name": name,
        }

    def _get_obs(self):
        self.renderer.update_scene(self.data, camera=self.camera_name)
        image = self.renderer.render().copy()
        robot_state = self.get_robot_state()
        object_state = self.get_object_state(self.target)
        return {
            "image": image,
            "instruction": self.instruction,
            "robot_state": robot_state,
            "object_state": object_state,
            "gripper_pos": robot_state["eef_pos"],
            "target_pos": object_state["target_pos"],
        }

