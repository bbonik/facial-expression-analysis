"""Analysis engine of the web demo: detection, both models, fit check, live smoothing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from render import Point

from facial_expression_analysis import DlibLandmarkDetector, EmotionEstimator, Frontalizer
from facial_expression_analysis.labels import describe_emotion
from facial_expression_analysis.model import EmotionModel, KernelEmotionModel, load_model

MODELS = {
    "PLS(31) · default": "Linear, well-calibrated, robust to head pose",
    "Kernel ridge": "Highest offline accuracy, less stable on new faces",
}
DEFAULT_MODEL = next(iter(MODELS))


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]
    landmarks: NDArray[np.float64]
    point: Point
    distance: float | None


@dataclass
class LiveState:
    """Per-browser-session state of the live stream."""

    ema: NDArray[np.float64] | None = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    last_t: float = 0.0
    fps: float = 0.0


class Engine:
    """Loads everything once; thread-safe for concurrent sessions (no mutable shared state)."""

    def __init__(self, max_width: int = 640) -> None:
        frontalizer = Frontalizer()
        self.kernel = KernelEmotionModel.load()
        models: dict[str, EmotionModel] = {
            "PLS(31) · default": load_model(),
            "Kernel ridge": self.kernel,
        }
        self.estimators = {k: EmotionEstimator(m, frontalizer) for k, m in models.items()}
        self.detector = DlibLandmarkDetector(upsample=0, max_width=max_width)
        self.detector_photo = DlibLandmarkDetector(upsample=1, max_width=1280)
        q = self.kernel.metadata.get("basis_distance_quantiles", {})
        self.p50, self.p99 = q.get("0.5"), q.get("0.99")

    def analyze(
        self, rgb: NDArray[np.uint8], model: str, *, photo: bool = False
    ) -> list[FaceResult]:
        estimator = self.estimators.get(model, self.estimators[DEFAULT_MODEL])
        detector = self.detector_photo if photo else self.detector
        faces = estimator.analyze_image(rgb, detector)
        faces.sort(key=lambda f: f.box.left)
        out = []
        for i, f in enumerate(faces):
            e = f.emotion
            feats = self.kernel.features.transform(e.frontal_landmarks[None])
            dist = float(self.kernel.basis_distance(feats)[0])
            b = f.box
            out.append(
                FaceResult(
                    box=(b.left, b.top, b.right, b.bottom),
                    landmarks=e.landmarks,
                    point=Point(
                        e.arousal,
                        e.valence,
                        e.intensity,
                        str(e.label),
                        tag=str(i + 1) if len(faces) > 1 else "",
                    ),
                    distance=dist,
                )
            )
        return out

    def fit(self, distance: float | None) -> tuple[float | None, float | None, float | None]:
        return distance, self.p50, self.p99


def largest(faces: list[FaceResult]) -> FaceResult | None:
    if not faces:
        return None
    return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


def update_live(
    state: LiveState, face: FaceResult | None, model: str, smoothing: float, trail_len: int = 40
) -> tuple[LiveState, Point | None]:
    """Exponential smoothing of (A, V, I) and the trail; resets when the model changes."""
    now = time.perf_counter()
    if state.last_t:
        inst = 1.0 / max(now - state.last_t, 1e-6)
        state.fps = inst if not state.fps else 0.85 * state.fps + 0.15 * inst
    state.last_t = now
    if model != state.model:
        state.ema, state.trail, state.model = None, [], model
    if face is None:
        return state, None
    p = face.point
    current = np.array([p.arousal, p.valence, p.intensity])
    s = float(np.clip(smoothing, 0.0, 0.95))
    state.ema = current if state.ema is None else s * state.ema + (1 - s) * current
    a, v, i = (float(x) for x in state.ema)
    state.trail = [*state.trail, (a, v)][-trail_len:]
    return state, Point(a, v, i, str(describe_emotion(a, v, i)))
