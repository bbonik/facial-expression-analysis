"""Landmark tracking: start each frame's fit from the previous frame's landmarks.

Re-running the face detector on every video frame makes landmarks jump: dlib's HOG
detector works on an 8-pixel grid and a coarse scale pyramid, so its box moves in steps
even when the face does not, and in low light it often misses the face altogether. The
tracker instead derives the box for the next frame from the current landmarks, with a
small linear model trained to reproduce the box *the detector would have drawn* around
those landmarks. The landmark fitter therefore sees boxes like the ones it was trained
on. A tight rectangle around the landmarks does not work: it is too small and mis-placed,
and the fitter then converges badly (see ``experiments/13_tracking.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import BoundingBox, DetectedFace
from .dlib_detector import DlibLandmarkDetector, to_gray

ASSET = Path(__file__).resolve().parents[1] / "assets" / "box_regressor.npz"


def _normalise(
    landmarks: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Centre and scale landmarks ``(n, 68, 2)``; return centres, scales and features."""
    centre = landmarks.mean(axis=1)
    scale = np.sqrt(((landmarks - centre[:, None]) ** 2).sum(-1).mean(1))
    feats = (landmarks - centre[:, None]) / np.maximum(scale, 1e-9)[:, None, None]
    return centre, scale, feats.reshape(len(landmarks), -1)


@dataclass(frozen=True)
class BoxRegressor:
    """Predict the dlib detector's face box from 68 landmarks.

    Landmarks are centred and scaled to unit RMS radius; a linear model predicts the box
    centre offset and the log of its width and height in those units.
    """

    weights: NDArray[np.float64]  # (136, 4)
    bias: NDArray[np.float64]  # (4,)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        landmarks: NDArray[np.float64],
        boxes: NDArray[np.float64],
        alpha: float = 1e-2,
        metadata: dict[str, Any] | None = None,
    ) -> BoxRegressor:
        """Ridge regression from landmarks ``(n, 68, 2)`` to boxes ``(n, 4)`` (l, t, r, b)."""
        centre, scale, X = _normalise(np.asarray(landmarks, dtype=np.float64))
        b = np.asarray(boxes, dtype=np.float64)
        bc = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
        wh = np.c_[b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
        Y = np.c_[(bc - centre) / scale[:, None], np.log(wh / scale[:, None])]
        xm, ym = X.mean(0), Y.mean(0)
        Xc = X - xm
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (Y - ym))
        return cls(W, ym - xm @ W, dict(metadata or {}))

    def predict(self, landmarks: NDArray[np.float64]) -> NDArray[np.float64]:
        """Boxes ``(n, 4)`` (or ``(4,)`` for a single ``(68, 2)`` shape) as floats."""
        lm = np.asarray(landmarks, dtype=np.float64)
        single = lm.ndim == 2
        centre, scale, X = _normalise(lm[None] if single else lm)
        T = X @ self.weights + self.bias
        c = centre + T[:, :2] * scale[:, None]
        half = np.exp(T[:, 2:]) * scale[:, None] / 2
        out: NDArray[np.float64] = np.c_[c - half, c + half]
        return out[0] if single else out

    def save(self, path: str | Path) -> None:
        import json

        np.savez_compressed(
            path, weights=self.weights, bias=self.bias, metadata=json.dumps(self.metadata)
        )

    @classmethod
    def load(cls, path: str | Path = ASSET) -> BoxRegressor:
        import json

        with np.load(path) as data:
            meta = json.loads(str(data["metadata"])) if "metadata" in data else {}
            return cls(data["weights"], data["bias"], meta)


def _iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union)


@dataclass(eq=False)  # compared by identity
class _Track:
    landmarks: NDArray[np.float64]
    box: NDArray[np.float64]
    misses: int = 0


class LandmarkTracker:
    """Track faces across video frames; a drop-in :class:`LandmarkDetector` for video.

    Each tracked face is re-fitted inside a box predicted from its previous landmarks
    (lightly smoothed over time), using the robust fit of
    :meth:`DlibLandmarkDetector.fit`. The face detector runs only when nothing is
    tracked, when a fit looks doubtful, and every ``redetect_every`` frames to pick up
    new faces. A track is
    dropped when the fit spread exceeds ``max_spread`` on ``max_misses`` consecutive
    frames (the box no longer contains a face) or when its box leaves the image.

    The tracker is stateful: use one instance per video stream, and :meth:`reset` it
    when the stream restarts.
    """

    def __init__(
        self,
        detector: DlibLandmarkDetector | None = None,
        regressor: BoxRegressor | None = None,
        *,
        max_faces: int = 4,
        redetect_every: int = 5,
        box_smoothing: float = 0.5,
        max_spread: float = 0.03,
        max_misses: int = 2,
    ) -> None:
        self.detector = detector or DlibLandmarkDetector(upsample=0, robust=True)
        self.regressor = regressor or BoxRegressor.load()
        self.max_faces = max_faces
        self.redetect_every = redetect_every
        self.box_smoothing = box_smoothing
        self.max_spread = max_spread
        self.max_misses = max_misses
        self.tracks: list[_Track] = []
        self.frame = 0

    def reset(self) -> None:
        self.tracks = []
        self.frame = 0

    def detect(self, image: NDArray[np.uint8]) -> list[DetectedFace]:
        gray = to_gray(image)
        h, w = gray.shape[:2]
        self.frame += 1
        visible: list[_Track] = []
        kept: list[_Track] = []
        doubtful = False
        for track in self.tracks:
            box = self.regressor.predict(track.landmarks)
            box = self.box_smoothing * track.box + (1 - self.box_smoothing) * box
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            if not (0 <= cx < w and 0 <= cy < h) or box[2] - box[0] < 20:
                continue  # face left the image
            fit = self.detector.fit(gray, box, robust=True)
            if fit.spread > self.max_spread:
                track.misses += 1
                doubtful = True
                if track.misses < self.max_misses:
                    kept.append(track)  # keep the old landmarks, show nothing this frame
                continue
            track.landmarks, track.box, track.misses = fit.landmarks, box, 0
            kept.append(track)
            visible.append(track)
        # two tracks that converged on the same face: keep the older one
        self.tracks = []
        for track in kept:
            if all(_iou(track.box, t.box) < 0.5 for t in self.tracks):
                self.tracks.append(track)
        visible = [t for t in visible if t in self.tracks]

        if (
            doubtful
            or not self.tracks
            or (len(self.tracks) < self.max_faces and self.frame % self.redetect_every == 0)
        ):
            for det in self.detector.detect_boxes(gray):
                box = np.array([det.left, det.top, det.right, det.bottom], dtype=np.float64)
                match = [t for t in self.tracks if _iou(box, t.box) > 0.3]
                if match and (match[0].misses == 0 or match[0] in visible):
                    continue  # already tracked
                if not match and len(self.tracks) >= self.max_faces:
                    continue
                # the detector vouches for this face: accept the fit whatever its spread
                fit = self.detector.fit(gray, box, robust=True)
                if match:  # a doubtful track, re-initialised from the detector
                    track = match[0]
                    track.landmarks, track.box, track.misses = fit.landmarks, box, 0
                else:
                    track = _Track(fit.landmarks, box)
                    self.tracks.append(track)
                visible.append(track)

        return [
            DetectedFace(
                box=BoundingBox(*(round(float(v)) for v in t.box)),
                landmarks=t.landmarks,
            )
            for t in visible
        ]
