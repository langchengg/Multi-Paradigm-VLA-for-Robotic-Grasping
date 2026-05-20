"""Environment package with lazy imports and shared headless-rendering setup."""

from ._rendering import configure_headless_rendering


configure_headless_rendering()

__all__ = ["FrankaGraspEnv"]


def __getattr__(name):
    if name == "FrankaGraspEnv":
        from .franka_grasp_env import FrankaGraspEnv

        return FrankaGraspEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
