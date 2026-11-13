"""Detector interface shared by all landmark back-ends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned face box in pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)

    def clip(self, width: int, height: int) -> BoundingBox:
        """Clip the box to an image of the given size."""
        return BoundingBox(
            max(self.left, 0), max(self.top, 0), min(self.right, width), min(self.bottom, height)
        )

    def as_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom}


@dataclass(frozen=True, slots=True)
class DetectedFace:
    """A detected face with its 68 landmarks ``(68, 2)`` in pixel coordinates."""

    box: BoundingBox
    landmarks: NDArray[np.float64]


@runtime_checkable
class LandmarkDetector(Protocol):
    """Anything that finds faces and returns 68-point landmarks.

    Back-ends with a different landmark scheme (e.g. MediaPipe's 478 points)
    must convert their output to the 68-point dlib scheme, because the
    frontalization and emotion models were trained on dlib geometry.
    """

    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        """Detect faces in an RGB ``(h, w, 3)`` or grayscale ``(h, w)`` image."""
        ...
