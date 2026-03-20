import ctypes
import os
import sys
from ctypes.util import find_library


HEADLESS_RENDER_BACKENDS = {"egl", "osmesa"}
HEADLESS_APT_INSTALL = (
    "apt-get update -qq && apt-get install -y -qq "
    "libgl1-mesa-glx libgl1-mesa-dev libosmesa6-dev libglew-dev patchelf"
)


def configure_headless_rendering(environ=None, platform=None):
    """Default MuJoCo to OSMesa on headless Linux without overriding explicit backends."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform

    has_display = bool(environ.get("DISPLAY"))
    if not platform.startswith("linux") or has_display:
        return environ.get("MUJOCO_GL")

    if "MUJOCO_GL" not in environ:
        environ["MUJOCO_GL"] = "osmesa"

    if environ.get("MUJOCO_GL") == "osmesa":
        environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

    return environ.get("MUJOCO_GL")


def create_renderer(mujoco, model, height, width, *, environ=None, platform=None):
    """Create a MuJoCo renderer with clearer headless failure modes."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    backend = configure_headless_rendering(environ=environ, platform=platform)
    has_display = bool(environ.get("DISPLAY"))

    if platform.startswith("linux") and not has_display:
        _validate_headless_backend(backend)
        if backend == "osmesa":
            missing = _missing_osmesa_libraries()
            if missing:
                raise RuntimeError(_missing_osmesa_message(missing, environ))

    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        raise RuntimeError(
            _renderer_failure_message(exc, backend=backend, has_display=has_display)
        ) from exc


def _validate_headless_backend(backend):
    if backend in HEADLESS_RENDER_BACKENDS:
        return

    raise RuntimeError(
        "Headless Linux detected but MUJOCO_GL is not configured for offscreen rendering. "
        "Set MUJOCO_GL=osmesa or MUJOCO_GL=egl before importing MuJoCo, "
        f"or install/run with a valid X display. Current MUJOCO_GL={backend!r}."
    )


def _missing_osmesa_libraries():
    missing = []
    if not _can_load_library("OSMesa", "libOSMesa.so", "libOSMesa.so.8", "libOSMesa.so.6"):
        missing.append("OSMesa")
    if not _can_load_library("GLEW", "libGLEW.so", "libGLEW.so.2.2", "libGLEW.so.2.1"):
        missing.append("GLEW")
    return missing


def _can_load_library(*candidates):
    tried = []
    for candidate in candidates:
        resolved = find_library(candidate)
        if resolved:
            tried.append(resolved)
        tried.append(candidate)

    seen = set()
    for name in tried:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _missing_osmesa_message(missing, environ):
    missing_names = ", ".join(missing)
    return (
        "MuJoCo offscreen rendering is configured with MUJOCO_GL=osmesa, "
        f"but the native library setup is incomplete ({missing_names} not found). "
        "This usually happens on Kaggle or headless Linux before the OSMesa packages are installed. "
        f"Install: `{HEADLESS_APT_INSTALL}`. "
        f"Current DISPLAY={environ.get('DISPLAY')!r}, MUJOCO_GL={environ.get('MUJOCO_GL')!r}."
    )


def _renderer_failure_message(exc, *, backend, has_display):
    base = f"MuJoCo renderer initialization failed: {exc}"
    if not has_display and backend == "osmesa":
        return (
            f"{base}. Headless Linux is using OSMesa, so this usually means the native OSMesa/OpenGL "
            f"stack is missing or failed to load. Install: `{HEADLESS_APT_INSTALL}`."
        )
    if not has_display and backend not in HEADLESS_RENDER_BACKENDS:
        return (
            f"{base}. DISPLAY is unset, so MuJoCo cannot use the default GLFW/X11 path. "
            "Set MUJOCO_GL=osmesa or MUJOCO_GL=egl before importing MuJoCo."
        )
    return base
