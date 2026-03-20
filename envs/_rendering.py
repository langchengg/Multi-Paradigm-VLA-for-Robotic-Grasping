import os
import subprocess
import sys


HEADLESS_RENDER_BACKENDS = ("egl", "osmesa")
HEADLESS_APT_INSTALL = (
    "apt-get update -qq && apt-get install -y -qq "
    "libgl1-mesa-glx libgl1-mesa-dev libegl1-mesa-dev "
    "libosmesa6-dev libglew-dev patchelf"
)

_AUTO_BACKEND = None
_PROBE_CACHE = {}


def configure_headless_rendering(environ=None, platform=None, python_executable=None):
    """Pick a working MuJoCo backend before importing mujoco on headless Linux."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    python_executable = sys.executable if python_executable is None else python_executable

    has_display = bool(environ.get("DISPLAY"))
    if not platform.startswith("linux") or has_display:
        return environ.get("MUJOCO_GL")

    requested_backend = _requested_backend(environ)
    if requested_backend is not None:
        _validate_headless_backend(requested_backend)
        _set_pyopengl_platform(environ, requested_backend)
        _ensure_backend_works(requested_backend, environ, python_executable)
        return requested_backend

    requested_pyopengl = _requested_pyopengl_backend(environ)
    if requested_pyopengl is not None:
        _ensure_backend_works(requested_pyopengl, environ, python_executable)
        environ["MUJOCO_GL"] = requested_pyopengl
        _set_pyopengl_platform(environ, requested_pyopengl)
        return requested_pyopengl

    backend = _auto_detect_headless_backend(environ, python_executable)
    environ["MUJOCO_GL"] = backend
    _set_pyopengl_platform(environ, backend)
    return backend


def create_renderer(mujoco, model, height, width, *, environ=None, platform=None):
    """Create a MuJoCo renderer with clearer headless failure modes."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    backend = configure_headless_rendering(environ=environ, platform=platform)
    has_display = bool(environ.get("DISPLAY"))

    if platform.startswith("linux") and not has_display:
        _validate_headless_backend(backend)

    try:
        return mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        raise RuntimeError(
            _renderer_failure_message(exc, backend=backend, has_display=has_display)
        ) from exc


def _requested_backend(environ):
    value = environ.get("MUJOCO_GL")
    if value is None:
        return None
    value = value.strip().lower()
    return value or None


def _requested_pyopengl_backend(environ):
    value = environ.get("PYOPENGL_PLATFORM")
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value not in HEADLESS_RENDER_BACKENDS:
        raise RuntimeError(
            f"Headless Linux detected but PYOPENGL_PLATFORM={value!r} is incompatible. "
            "Use 'egl', 'osmesa', or unset it."
        )
    return value


def _set_pyopengl_platform(environ, backend):
    current = environ.get("PYOPENGL_PLATFORM")
    if current is None:
        environ["PYOPENGL_PLATFORM"] = backend
        return

    normalized = current.strip().lower()
    if normalized != backend:
        raise RuntimeError(
            f"Cannot use MUJOCO_GL={backend!r} because PYOPENGL_PLATFORM is already set to "
            f"{current!r}. Set both to the same value or unset PYOPENGL_PLATFORM."
        )


def _auto_detect_headless_backend(environ, python_executable):
    global _AUTO_BACKEND

    if _AUTO_BACKEND is not None:
        return _AUTO_BACKEND

    failures = []
    for backend in HEADLESS_RENDER_BACKENDS:
        ok, details = _probe_backend(backend, environ, python_executable)
        if ok:
            _AUTO_BACKEND = backend
            return backend
        failures.append((backend, details))

    raise RuntimeError(_no_backend_message(failures, environ))


def _ensure_backend_works(backend, environ, python_executable):
    ok, details = _probe_backend(backend, environ, python_executable)
    if ok:
        return

    raise RuntimeError(
        f"Configured headless MuJoCo backend {backend!r} is not usable in this environment. "
        f"Install native rendering libraries with `{HEADLESS_APT_INSTALL}` or switch to another "
        f"backend. Probe output:\n{details}"
    )


def _probe_backend(backend, environ, python_executable):
    env_key = (
        backend,
        python_executable,
        bool(environ.get("DISPLAY")),
        environ.get("MUJOCO_EGL_DEVICE_ID"),
    )
    cached = _PROBE_CACHE.get(env_key)
    if cached is not None:
        return cached

    probe_env = os.environ.copy()
    probe_env.update(environ)
    probe_env["MUJOCO_GL"] = backend
    probe_env["PYOPENGL_PLATFORM"] = backend

    probe_code = """
import mujoco

model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
renderer = mujoco.Renderer(model, height=4, width=4)
renderer.close()
print('ok')
"""

    try:
        result = subprocess.run(
            [python_executable, "-c", probe_code],
            env=probe_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        outcome = (False, repr(exc))
    else:
        details = (result.stderr or result.stdout or "").strip()
        if result.returncode == 0:
            outcome = (True, details or "ok")
        else:
            outcome = (False, details or f"probe exited with code {result.returncode}")

    _PROBE_CACHE[env_key] = outcome
    return outcome


def _validate_headless_backend(backend):
    if backend in HEADLESS_RENDER_BACKENDS:
        return

    raise RuntimeError(
        "Headless Linux detected but MUJOCO_GL is not configured for offscreen rendering. "
        "Set MUJOCO_GL=egl or MUJOCO_GL=osmesa before importing MuJoCo, "
        f"or install/run with a valid X display. Current MUJOCO_GL={backend!r}."
    )


def _no_backend_message(failures, environ):
    lines = [
        "Headless Linux detected but no working MuJoCo OpenGL backend was found.",
        f"Tried backends: {', '.join(HEADLESS_RENDER_BACKENDS)}.",
        f"Install native rendering libraries with `{HEADLESS_APT_INSTALL}`.",
        f"Current DISPLAY={environ.get('DISPLAY')!r}.",
    ]
    for backend, details in failures:
        lines.append(f"{backend} probe failed:")
        lines.append(details)
    return "\n".join(lines)


def _renderer_failure_message(exc, *, backend, has_display):
    base = f"MuJoCo renderer initialization failed: {exc}"
    if not has_display:
        return (
            f"{base}. Headless Linux selected MUJOCO_GL={backend!r}. "
            f"If this machine changed recently, reinstall native libs with `{HEADLESS_APT_INSTALL}`."
        )
    return base
