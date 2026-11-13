"""Image input/output (RGB uint8 convention throughout the package)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_image(path: str | Path) -> NDArray[np.uint8]:
    """Read an image file as an RGB ``(h, w, 3)`` uint8 array."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def save_image(path: str | Path, rgb: NDArray[np.uint8]) -> None:
    """Write an RGB image to disk."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1])):
        raise OSError(f"Cannot write image: {path}")
