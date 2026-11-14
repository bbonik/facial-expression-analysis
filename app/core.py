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


TRACK_COLORS = ("#a78bfa", "#38bdf8", "#fbbf24", "#f472b6")  # violet, sky, amber, pink
MAX_FACES = len(TRACK_COLORS)


@dataclass
class Track:
    """One face followed across frames, with its own smoothing state and trail."""

    id: int
    ema: NDArray[np.float64]
    center: tuple[float, float]
    size: float
    distance: float | None = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    missed: int = 0

    @property
    def color(self) -> str:
        return TRACK_COLORS[(self.id - 1) % len(TRACK_COLORS)]


@dataclass
class LiveState:
    """Per-browser-session state of the live stream."""

    tracks: list[Track] = field(default_factory=list)
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


def _geometry(face: FaceResult) -> tuple[tuple[float, float], float]:
    x0, y0, x1, y1 = face.box
    return ((x0 + x1) / 2, (y0 + y1) / 2), float(max(x1 - x0, y1 - y0, 1))


def update_tracks(
    state: LiveState,
    faces: list[FaceResult],
    model: str,
    smoothing: float,
    *,
    max_missed: int = 8,
    trail_len: int = 40,
    match_radius: float = 0.75,
) -> tuple[LiveState, list[tuple[Track, FaceResult, Point]]]:
    """Follow faces across frames and smooth each one's (A, V, I) separately.

    Faces are matched greedily to the nearest track (centre distance relative to face
    size). Unmatched faces start new tracks with the smallest free number; tracks unseen
    for ``max_missed`` frames are dropped. Changing the model resets everything.
    Returns the faces seen in this frame as (track, face, smoothed point).
    """
    now = time.perf_counter()
    if state.last_t:
        inst = 1.0 / max(now - state.last_t, 1e-6)
        state.fps = inst if not state.fps else 0.85 * state.fps + 0.15 * inst
    state.last_t = now
    if model != state.model:
        state.tracks, state.model = [], model

    faces = sorted(faces, key=lambda f: -_geometry(f)[1])[:MAX_FACES]
    geo = [_geometry(f) for f in faces]
    candidates = sorted(
        (np.hypot(c[0] - t.center[0], c[1] - t.center[1]) / max(size, t.size), ti, fi)
        for fi, (c, size) in enumerate(geo)
        for ti, t in enumerate(state.tracks)
    )
    assigned: dict[int, Track] = {}
    used: set[int] = set()
    for d, ti, fi in candidates:
        if d > match_radius or ti in used or fi in assigned:
            continue
        assigned[fi] = state.tracks[ti]
        used.add(ti)

    s = float(np.clip(smoothing, 0.0, 0.95))
    seen: list[tuple[Track, FaceResult]] = []
    for fi, face in enumerate(faces):
        current = np.array([face.point.arousal, face.point.valence, face.point.intensity])
        track = assigned.get(fi)
        if track is None:
            taken = {t.id for t in state.tracks}
            new_id = next(k for k in range(1, len(taken) + 2) if k not in taken)
            track = Track(new_id, current.copy(), *geo[fi])
            state.tracks.append(track)
        else:
            track.ema = s * track.ema + (1 - s) * current
            track.center, track.size = geo[fi]
        track.missed = 0
        track.distance = face.distance
        a, v = float(track.ema[0]), float(track.ema[1])
        track.trail = [*track.trail, (a, v)][-trail_len:]
        seen.append((track, face))
    seen_ids = {t.id for t, _ in seen}
    for t in state.tracks:
        if t.id not in seen_ids:
            t.missed += 1
    state.tracks = sorted((t for t in state.tracks if t.missed <= max_missed), key=lambda t: t.id)

    multi = len(seen) > 1
    out = []
    for track, face in sorted(seen, key=lambda x: x[0].id):
        a, v, i = (float(x) for x in track.ema)
        point = Point(
            a, v, i, str(describe_emotion(a, v, i)),
            tag=str(track.id) if multi else "", accent=track.color if multi else "",
        )  # fmt: skip
        out.append((track, face, point))
    return state, out
