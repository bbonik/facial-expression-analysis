"""dlib HOG face detector + 68-point shape predictor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..assets import fetch_dlib_predictor
from ..constants import N_LANDMARKS
from .base import BoundingBox, DetectedFace

BoxLike = BoundingBox | Sequence[float] | NDArray[np.float64]

# Median standard deviation of the grey levels inside dlib face boxes on well-lit photos
# (LFW). Dimmer faces are scaled up to it before fitting, see ``DlibLandmarkDetector.fit``.
TARGET_FACE_CONTRAST = 43.7
# Detection images whose 1st-99th percentile grey-level range is narrower than this are
# stretched linearly to 0..255 before running the HOG detector.
MIN_DETECTION_RANGE = 150.0
# Relative box offsets (dx, dy as a fraction of the box width) of the averaged fits.
_FIT_OFFSETS = ((0.0, 0.0), (0.04, 0.0), (-0.04, 0.0), (0.0, 0.04), (0.0, -0.04))


@dataclass(frozen=True, slots=True)
class LandmarkFit:
    """Landmarks fitted inside a given box, with the spread of the averaged fits.

    ``spread`` is the RMS disagreement between fits started from slightly shifted
    boxes, relative to the outer inter-ocular distance. It is small (< 2 %) on faces
    and large (> 3 %) when the box does not contain a face, so trackers use it to
    notice that they have lost the face. It is 0 for a single (non-robust) fit.
    """

    landmarks: NDArray[np.float64]
    spread: float


def _box_array(box: BoxLike) -> NDArray[np.float64]:
    if isinstance(box, BoundingBox):
        return np.array([box.left, box.top, box.right, box.bottom], dtype=np.float64)
    return np.asarray(box, dtype=np.float64).reshape(4)


def stretch_contrast(
    gray: NDArray[np.uint8], min_range: float = MIN_DETECTION_RANGE
) -> NDArray[np.uint8]:
    """Linearly stretch a low-contrast image to 0..255 (no local or non-linear change)."""
    lo, hi = np.percentile(gray[::4, ::4], (1, 99))
    if hi - lo >= min_range:
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0))
    return np.asarray(np.clip(out, 0, 255), dtype=np.uint8)


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
    robust
        Use the low-light robust variant: low-contrast images are stretched before
        detection and landmarks are fitted with :meth:`fit` ``(robust=True)``. Off by
        default so that photo results match the landmarks the models were trained on.
    """

    def __init__(
        self,
        predictor_path: str | Path | None = None,
        *,
        upsample: int = 1,
        max_width: int | None = None,
        robust: bool = False,
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
        self.robust = robust

    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        gray = to_gray(image)
        faces = []
        for box in self.detect_boxes(gray):
            fit = self.fit(gray, box, robust=self.robust)
            faces.append(DetectedFace(box=box, landmarks=fit.landmarks))
        return faces

    def detect_boxes(self, gray: NDArray[np.uint8]) -> list[BoundingBox]:
        """Run the face detector only (boxes in full-resolution pixel coordinates)."""
        scale = 1.0
        small = gray
        if self.max_width is not None and gray.shape[1] > self.max_width:
            import cv2

            scale = gray.shape[1] / self.max_width
            size = (self.max_width, round(gray.shape[0] / scale))
            small = np.asarray(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), np.uint8)
        if self.robust:
            small = stretch_contrast(small)
        return [
            BoundingBox(
                round(r.left() * scale),
                round(r.top() * scale),
                round(r.right() * scale),
                round(r.bottom() * scale),
            )
            for r in self._detector(small, self.upsample)
        ]

    def _fit_once(self, gray: NDArray[np.uint8], box: NDArray[np.float64]) -> NDArray[np.float64]:
        x0, y0, x1, y1 = (round(float(v)) for v in box)
        shape = self._predictor(gray, self._dlib.rectangle(x0, y0, x1, y1))
        return np.array(
            [(shape.part(i).x, shape.part(i).y) for i in range(N_LANDMARKS)], dtype=np.float64
        )

    def fit(self, gray: NDArray[np.uint8], box: BoxLike, *, robust: bool = False) -> LandmarkFit:
        """Fit 68 landmarks inside ``box`` of a grayscale image.

        With ``robust=True`` the face region is prepared and fitted the way that held up
        best in simulated low light (``experiments/13_tracking.py``):

        * a linear contrast gain (never a reduction) brings the face box to the contrast
          of a well-lit face, because dlib's fitter compares raw pixel differences with
          fixed thresholds and loses accuracy on dim, flat faces;
        * a slight Gaussian blur (sigma = box width / 110 px) suppresses sensor noise;
        * the fit is repeated from five slightly shifted boxes and averaged, which halves
          frame-to-frame jitter; their disagreement is returned as ``spread``.
        """
        b = _box_array(box)
        if not robust:
            return LandmarkFit(self._fit_once(gray, b), 0.0)

        import cv2

        w = max(b[2] - b[0], 1.0)
        # work on a crop with a margin around the box: cheaper blur and statistics
        h_img, w_img = gray.shape[:2]
        x0, y0 = int(max(b[0] - 0.6 * w, 0)), int(max(b[1] - 0.6 * w, 0))
        x1, y1 = int(min(b[2] + 0.6 * w, w_img)), int(min(b[3] + 0.6 * w, h_img))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return LandmarkFit(self._fit_once(gray, b), 1.0)
        crop: NDArray[np.float32] = gray[y0:y1, x0:x1].astype(np.float32)
        inner = gray[int(max(b[1], 0)) : int(max(b[3], 1)), int(max(b[0], 0)) : int(max(b[2], 1))]
        if inner.size:
            std = float(inner.std())
            if std < TARGET_FACE_CONTRAST:
                gain = TARGET_FACE_CONTRAST / max(std, 1.0)
                crop = ((crop - float(inner.mean())) * gain + 128.0).astype(np.float32)
        crop = np.asarray(cv2.GaussianBlur(crop, (0, 0), w / 110.0), dtype=np.float32)
        img = np.ascontiguousarray(np.clip(np.round(crop), 0, 255).astype(np.uint8))

        offset = np.array([x0, y0, x0, y0], dtype=np.float64)
        local = b - offset
        fits = np.array(
            [self._fit_once(img, local + w * np.array([dx, dy, dx, dy])) for dx, dy in _FIT_OFFSETS]
        )
        mean = fits.mean(axis=0)
        io = float(np.linalg.norm(mean[36] - mean[45]))
        spread = float(np.sqrt(((fits - mean) ** 2).sum(-1).mean(0)).mean() / max(io, 1e-6))
        return LandmarkFit(mean + offset[:2], spread)


def to_gray(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Validate an RGB ``(h, w, 3)`` or grayscale ``(h, w)`` uint8 image; return grayscale."""
    img = np.ascontiguousarray(image)
    if img.dtype != np.uint8 or img.ndim not in (2, 3):
        raise ValueError("Expected a uint8 grayscale (h, w) or RGB (h, w, 3) image")
    if img.ndim == 2:
        return img
    weights = np.array([0.299, 0.587, 0.114])
    return np.ascontiguousarray((img[..., :3] @ weights).round().astype(np.uint8))
