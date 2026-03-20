import os
import sys

# Auto-configure headless rendering for Kaggle/Linux environments without a display
if sys.platform.startswith("linux") and "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from .simple_grasp_env import SimpleGraspEnv
from .robosuite_wrapper import RobosuiteVLAWrapper
from .franka_grasp_env import FrankaGraspEnv
