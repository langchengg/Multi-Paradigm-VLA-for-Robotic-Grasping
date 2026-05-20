"""
Generate Videos & GIFs
======================
Create presentation-quality video assets from evaluation results.
- Individual episode GIFs
- Side-by-side comparison videos
- Annotated frames with overlays
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_comparison_gif(episodes_dict, output_path, fps=8, label_height=30):
    """
    Create a side-by-side comparison GIF of different decoders.

    Args:
        episodes_dict: {"decoder_name": [list of (H,W,3) frames], ...}
        output_path: path to save the output GIF
        fps: frames per second
        label_height: height of text label bar
    """
    names = list(episodes_dict.keys())
    frame_lists = list(episodes_dict.values())

    # Find max episode length
    max_len = max(len(f) for f in frame_lists)

    # Pad shorter episodes by repeating last frame
    for i, frames in enumerate(frame_lists):
        if len(frames) < max_len:
            frame_lists[i] = frames + [frames[-1]] * (max_len - len(frames))

    # Get frame size
    h, w = frame_lists[0][0].shape[:2]
    n_models = len(names)
    canvas_w = w * n_models + (n_models - 1) * 2  # 2px gap between panels
    canvas_h = h + label_height

    composite_frames = []

    for t in range(max_len):
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
        draw = ImageDraw.Draw(canvas)

        for i, (name, frames) in enumerate(zip(names, frame_lists)):
            x_offset = i * (w + 2)

            # Label
            text_x = x_offset + w // 2
            try:
                draw.text((text_x, 5), name, fill="white", anchor="mt")
            except TypeError:
                # Fallback for older Pillow without anchor
                draw.text((x_offset + 5, 5), name, fill="white")

            # Frame
            frame_img = Image.fromarray(frames[t])
            canvas.paste(frame_img, (x_offset, label_height))

        composite_frames.append(canvas)

    # Save
    duration = int(1000 / fps)
    composite_frames[0].save(
        output_path,
        save_all=True,
        append_images=composite_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"Saved comparison GIF: {output_path} "
          f"({max_len} frames, {n_models} models)")


def annotate_frame(frame, text, position="bottom", font_size=14):
    """
    Add text annotation to a frame.

    Args:
        frame: (H, W, 3) uint8 array
        text: annotation string
        position: "top" or "bottom"

    Returns:
        annotated: (H+bar, W, 3) uint8 array
    """
    h, w = frame.shape[:2]
    bar_h = 25
    img = Image.fromarray(frame)

    if position == "bottom":
        canvas = Image.new("RGB", (w, h + bar_h), color=(20, 20, 20))
        canvas.paste(img, (0, 0))
        text_y = h + 4
    else:
        canvas = Image.new("RGB", (w, h + bar_h), color=(20, 20, 20))
        canvas.paste(img, (0, bar_h))
        text_y = 4

    draw = ImageDraw.Draw(canvas)
    draw.text((5, text_y), text, fill=(200, 200, 200))

    return np.array(canvas)


def save_episode_gif(frames, path, fps=10, annotate=True, instruction=None,
                     success=None):
    """
    Save a single episode as GIF with optional annotations.

    Args:
        frames: list of (H, W, 3) uint8 arrays
        path: output path
        fps: frames per second
        annotate: whether to add step/status annotations
        instruction: language instruction to display
        success: whether the episode succeeded
    """
    processed = []
    n = len(frames)

    for i, frame in enumerate(frames):
        if annotate:
            parts = [f"Step {i+1}/{n}"]
            if instruction:
                parts.append(f'"{instruction}"')
            if i == n - 1 and success is not None:
                parts.append("✓ Success!" if success else "✗ Failed")
            text = " | ".join(parts)
            frame = annotate_frame(frame, text, position="bottom")
        processed.append(Image.fromarray(frame))

    if not processed:
        return

    duration = int(1000 / fps)
    processed[0].save(
        path, save_all=True, append_images=processed[1:],
        duration=duration, loop=0,
    )
    print(f"Saved: {path} ({n} frames)")


# ─── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Video Generation")
    print("=" * 60)

    # Create dummy frames
    frames_a = [np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
                for _ in range(20)]
    frames_b = [np.random.randint(50, 150, (128, 128, 3), dtype=np.uint8)
                for _ in range(20)]
    frames_c = [np.random.randint(30, 130, (128, 128, 3), dtype=np.uint8)
                for _ in range(20)]

    # Test single episode GIF
    save_episode_gif(
        frames_a, "/tmp/test_episode.gif",
        instruction="pick up the red cube",
        success=True,
    )

    # Test comparison GIF
    create_comparison_gif(
        {
            "Autoregressive": frames_a,
            "Diffusion": frames_b,
            "Flow-Matching": frames_c,
        },
        "/tmp/test_comparison.gif",
    )

    print("\n✅ Video generation test passed!")
