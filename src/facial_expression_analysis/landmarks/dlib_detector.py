"""dlib HOG face detector + 68-point shape predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..assets import fetch_dlib_predictor
from ..constants import N_LANDMARKS
from .base import BoundingBox, DetectedFace


class DlibLandmarkDetector:
    """Detect faces with dlib's HOG detector and fit 68 landmarks.

    Parameters
    ----------
    predictor_path
        Path to ``shape_predictor_68_face_landmarks.dat``. Downloaded and cached
        on first use if ``None``.
    upsample
        Number of times to upsample the image before detection. ``1`` finds
        smaller faces at roughly 4x the cost; ``0`` is best for real-time use.
    max_width
        Images wider than this are downscaled for *detection* only (landmarks
        are always fitted at full resolution). ``None`` disables downscaling.
    """

    def __init__(
        self,
        predictor_path: str | Path | None = None,
        *,
        upsample: int = 1,
        max_width: int | None = None,
    ) -> None:
        try:
            import dlib
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "dlib is required for landmark detection: pip install dlib-bin"
            ) from exc
        self._dlib = dlib
        path = Path(predictor_path) if predictor_path is not None else fetch_dlib_predictor()
        if not path.is_file():
            raise FileNotFoundError(f"dlib shape predictor not found: {path}")
        self._detector: Any = dlib.get_frontal_face_detector()
        self._predictor: Any = dlib.shape_predictor(str(path))
        self.upsample = upsample
        self.max_width = max_width

    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        img = np.ascontiguousarray(image)
        if img.dtype != np.uint8 or img.ndim not in (2, 3):
            raise ValueError("Expected a uint8 grayscale (h, w) or RGB (h, w, 3) image")
        gray = img if img.ndim == 2 else _to_gray(img)

        scale = 1.0
        small = gray
        if self.max_width is not None and gray.shape[1] > self.max_width:
            import cv2

            scale = gray.shape[1] / self.max_width
            size = (self.max_width, round(gray.shape[0] / scale))
            small = np.asarray(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), np.uint8)

        faces = []
        for rect in self._detector(small, self.upsample):
            if scale != 1.0:
                rect = self._dlib.rectangle(
                    round(rect.left() * scale),
                    round(rect.top() * scale),
                    round(rect.right() * scale),
                    round(rect.bottom() * scale),
                )
            shape = self._predictor(gray, rect)
            landmarks = np.array(
                [(shape.part(i).x, shape.part(i).y) for i in range(N_LANDMARKS)],
                dtype=np.float64,
            )
            box = BoundingBox(rect.left(), rect.top(), rect.right(), rect.bottom())
            faces.append(DetectedFace(box=box, landmarks=landmarks))
        return faces


def _to_gray(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    weights = np.array([0.299, 0.587, 0.114])
    return np.ascontiguousarray((rgb[..., :3] @ weights).round().astype(np.uint8))
