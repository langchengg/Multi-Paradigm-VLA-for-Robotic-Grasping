from __future__ import annotations


def try_launch_viewer(env) -> bool:
    try:
        env.launch_viewer()
        return getattr(env, "_viewer", None) is not None
    except Exception as exc:
        print(
            "[viewer] MuJoCo viewer launch failed. On macOS, retry with "
            f"`mjpython -m src.visualization.live_watch ...`. Error: {exc}"
        )
        return False

