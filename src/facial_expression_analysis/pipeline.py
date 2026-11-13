"""End-to-end emotion estimation from landmarks or images."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .frontalization import Frontalizer
from .geometry import as_landmark_batch
from .labels import EmotionLabel, describe_emotion
from .model import LinearEmotionModel

if TYPE_CHECKING:
    from .landmarks import BoundingBox, LandmarkDetector


@dataclass(frozen=True, slots=True)
class EmotionEstimate:
    """Dimensional emotion estimate for a single face."""

    arousal: float
    """Activation, from -1 (passive) to 1 (energetic)."""
    valence: float
    """Pleasantness, from -1 (negative) to 1 (positive)."""
    intensity: float
    """Expression intensity, from 0 (neutral) to 1 (intense)."""
    label: EmotionLabel
    landmarks: NDArray[np.float64]
    """Input landmarks ``(68, 2)``, in image coordinates."""
    frontal_landmarks: NDArray[np.float64]
    """Frontalized landmarks ``(68, 2)``, in normalised coordinates."""

    def to_dict(self, *, include_landmarks: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arousal": self.arousal,
            "valence": self.valence,
            "intensity": self.intensity,
            "label": str(self.label),
            "emotion": self.label.emotion,
            "qualifier": self.label.qualifier,
        }
        if include_landmarks:
            out["landmarks"] = self.landmarks.tolist()
            out["frontal_landmarks"] = self.frontal_landmarks.tolist()
        return out


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    """Detection plus emotion estimate for one face in an image."""

    box: BoundingBox
    emotion: EmotionEstimate

    def to_dict(self, *, include_landmarks: bool = False) -> dict[str, Any]:
        emotion = self.emotion.to_dict(include_landmarks=include_landmarks)
        return {"box": self.box.as_dict(), **emotion}


class EmotionEstimator:
    """Estimate arousal, valence and intensity from 68-point facial landmarks.

    Pipeline: landmarks -> Procrustes normalisation -> frontalization ->
    pairwise-distance features -> linear regressor -> clipping and labelling.

    Parameters
    ----------
    model
        Emotion regressor; the bundled model is used if ``None``.
    frontalizer
        Landmark frontalizer; the bundled one is used if ``None``.
    intensity_from_av
        If ``True`` (default) intensity is the norm of ``(arousal, valence)``
        instead of the regressor's own intensity output. This keeps the three
        outputs geometrically consistent with the circumplex model.

    Examples
    --------
    >>> estimator = EmotionEstimator()
    >>> estimate = estimator.estimate(landmarks)  # (68, 2) array
    >>> estimate.valence, str(estimate.label)
    """

    def __init__(
        self,
        model: LinearEmotionModel | None = None,
        frontalizer: Frontalizer | None = None,
        *,
        intensity_from_av: bool = True,
    ) -> None:
        self.model = model if model is not None else LinearEmotionModel.load()
        self.frontalizer = frontalizer if frontalizer is not None else Frontalizer()
        self.intensity_from_av = intensity_from_av

    def predict(self, landmarks: ArrayLike) -> NDArray[np.float64]:
        """Vectorised prediction for ``(n, 68, 2)`` landmarks.

        Returns an ``(n, 3)`` array of clipped ``(arousal, valence, intensity)``.
        """
        batch, _ = as_landmark_batch(landmarks)
        raw = self.model.predict(self.frontalizer(batch))
        return self._postprocess(raw)

    def _postprocess(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty_like(raw)
        out[:, :2] = np.clip(raw[:, :2], -1.0, 1.0)
        if self.intensity_from_av:
            out[:, 2] = np.hypot(out[:, 0], out[:, 1])
        else:
            out[:, 2] = raw[:, 2]
        out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
        return out

    def estimate_batch(self, landmarks: ArrayLike) -> list[EmotionEstimate]:
        """Full estimates (with labels and landmarks) for a batch of faces."""
        batch, _ = as_landmark_batch(landmarks)
        frontal = self.frontalizer(batch)
        avi = self._postprocess(self.model.predict(frontal))
        return [
            EmotionEstimate(
                arousal=float(a),
                valence=float(v),
                intensity=float(i),
                label=describe_emotion(float(a), float(v), float(i)),
                landmarks=raw,
                frontal_landmarks=front,
            )
            for (a, v, i), raw, front in zip(avi, batch, frontal, strict=True)
        ]

    def estimate(self, landmarks: ArrayLike) -> EmotionEstimate:
        """Estimate the emotion of a single face from ``(68, 2)`` landmarks."""
        batch, single = as_landmark_batch(landmarks)
        if not single:
            raise ValueError("estimate() expects a single (68, 2) shape; use estimate_batch()")
        return self.estimate_batch(batch)[0]

    def analyze_image(
        self, image: NDArray[np.uint8], detector: LandmarkDetector
    ) -> list[FaceAnalysis]:
        """Detect all faces in an RGB or grayscale image and estimate emotions."""
        faces = detector.detect(image)
        if not faces:
            return []
        estimates = self.estimate_batch(np.stack([f.landmarks for f in faces]))
        return [FaceAnalysis(f.box, e) for f, e in zip(faces, estimates, strict=True)]


def largest_face(faces: Sequence[FaceAnalysis]) -> FaceAnalysis | None:
    """Return the face with the largest bounding box, if any."""
    return max(faces, key=lambda f: f.box.area, default=None)
