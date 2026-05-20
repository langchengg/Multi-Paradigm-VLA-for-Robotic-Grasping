from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio


def save_video(frames, path, fps: int = 10) -> None:
    if not frames:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)

