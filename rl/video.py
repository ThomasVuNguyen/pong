from __future__ import annotations

import os
from typing import Iterable

import numpy as np


def can_write_mp4() -> bool:
    try:
        import imageio.v2 as _imageio  # type: ignore
        import imageio_ffmpeg  # noqa: F401

        return True
    except Exception:
        return False


def write_mp4(path: str, frames: Iterable[np.ndarray], *, fps: int = 30) -> None:
    """
    Write frames (H,W,3) uint8 to an MP4 (H.264) video.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "imageio is required for MP4 writing. Install with: pip install imageio imageio-ffmpeg"
        ) from e

    # Note: imageio uses ffmpeg (via imageio-ffmpeg) for mp4.
    # yuv420p maximizes player compatibility.
    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
    ) as w:
        for f in frames:
            w.append_data(f)


