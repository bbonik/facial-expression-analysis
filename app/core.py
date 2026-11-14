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
    ema: NDArray[np.float64] | None
    center: tuple[float, float]
    size: float
    distance: float | None = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    missed: int = 0
    raw: NDArray[np.float64] | None = None  # this frame's uncalibrated (A, V, I)

    @property
    def color(self) -> str:
        return TRACK_COLORS[(self.id - 1) % len(TRACK_COLORS)]


@dataclass
class CalibrationRun:
    """A calibration in progress: countdown, then collecting raw neutral predictions."""

    start: float
    samples: dict[int, list[NDArray[np.float64]]] = field(default_factory=dict)
    distances: dict[int, list[float]] = field(default_factory=dict)


@dataclass
class LiveState:
    """Per-browser-session state of the live stream."""

    tracks: list[Track] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    last_t: float = 0.0
    fps: float = 0.0
    # per-person calibration, keyed by (track id, model):
    #   offsets: identity offset (own neutral reading - average neutral reading)
    #   neutral_raw: the person's own raw neutral reading (for "zero at my neutral")
    offsets: dict[tuple[int, str], NDArray[np.float64]] = field(default_factory=dict)
    neutral_raw: dict[tuple[int, str], NDArray[np.float64]] = field(default_factory=dict)
    zero: bool = False
    calibration: CalibrationRun | None = None
    calibrate_clicks: int = 0
    reset_clicks: int = 0
    message: str = ""
    message_kind: str = "info"  # info / ok / warn
    message_until: float = 0.0


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
        # average raw prediction for neutral faces of unseen people (calibration target)
        self.neutral = {
            k: np.asarray(m.metadata.get("neutral_mean", [0.0, 0.0, 0.0]), dtype=np.float64)
            for k, m in models.items()
        }

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
    zero: bool = False,
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
    if model != state.model or zero != state.zero:  # keep identities, restart smoothing
        for t in state.tracks:
            t.ema, t.trail = None, []
        if model != state.model:
            state.calibration = None
        state.model, state.zero = model, zero

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
        raw = np.array([face.point.arousal, face.point.valence, face.point.intensity])
        track = assigned.get(fi)
        if track is None:
            taken = {t.id for t in state.tracks}
            new_id = next(k for k in range(1, len(taken) + 2) if k not in taken)
            track = Track(new_id, None, *geo[fi])
            state.tracks.append(track)
        track.center, track.size = geo[fi]
        track.raw = raw
        current = displayed(state, track.id, raw)
        track.ema = current if track.ema is None else s * track.ema + (1 - s) * current
        track.missed = 0
        track.distance = face.distance
        a, v = float(track.ema[0]), float(track.ema[1])
        track.trail = [*track.trail, (a, v)][-trail_len:]
        seen.append((track, face))
    seen_ids = {t.id for t, _ in seen}
    for t in state.tracks:
        if t.id not in seen_ids:
            t.missed += 1
    dropped = {t.id for t in state.tracks if t.missed > max_missed}
    state.tracks = sorted((t for t in state.tracks if t.missed <= max_missed), key=lambda t: t.id)
    # a dropped face's calibration goes with it (its number may be reused by someone else)
    state.offsets = {k: v for k, v in state.offsets.items() if k[0] not in dropped}
    state.neutral_raw = {k: v for k, v in state.neutral_raw.items() if k[0] not in dropped}

    multi = len(seen) > 1
    out = []
    for track, face in sorted(seen, key=lambda x: x[0].id):
        assert track.ema is not None
        a, v, i = (float(x) for x in track.ema)
        point = Point(
            a, v, i, str(describe_emotion(a, v, i)),
            tag=str(track.id) if multi else "", accent=track.color if multi else "",
        )  # fmt: skip
        out.append((track, face, point))
    return state, out


# ----------------------------------------------------------------------------- calibration
COUNTDOWN_S = 3.0  # time to settle into a neutral face
COLLECT_S = 1.5  # averaging window
MIN_SAMPLES = 6
TIMEOUT_S = 8.0
MAX_OFFSET = 0.45  # larger A/V offsets mean the face was probably not neutral


def apply_offset(
    raw: NDArray[np.float64], offset: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Subtract a person's offset; keep A, V in [-1, 1] and I in [0, 1]."""
    if offset is None:
        return raw
    out = raw - offset
    return np.array([np.clip(out[0], -1, 1), np.clip(out[1], -1, 1), np.clip(out[2], 0, 1)])


def zero_at_neutral(raw: NDArray[np.float64], neutral: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map a person's own neutral reading to exactly (0, 0, 0).

    A and V are shifted; intensity is stretched so that full intensity stays at 1
    (shifting it would compress every expression). Costs accuracy on expressive faces
    (MorphSet, PLS(31): CCC A -0.04, V -0.06, I -0.16; see the model card).
    """
    a, v = raw[0] - neutral[0], raw[1] - neutral[1]
    i = (raw[2] - neutral[2]) / max(1.0 - neutral[2], 0.2)
    return np.array([np.clip(a, -1, 1), np.clip(v, -1, 1), np.clip(i, 0, 1)])


def displayed(state: LiveState, track_id: int, raw: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return what is shown for a face: raw, identity-calibrated, or zeroed at its neutral."""
    key = (track_id, state.model)
    if state.zero and key in state.neutral_raw:
        return zero_at_neutral(raw, state.neutral_raw[key])
    return apply_offset(raw, state.offsets.get(key))


def _say(state: LiveState, text: str, kind: str = "info", seconds: float = 4.0) -> None:
    state.message, state.message_kind = text, kind
    state.message_until = time.perf_counter() + seconds


def handle_buttons(state: LiveState, calibrate_clicks: int, reset_clicks: int) -> None:
    """React to button presses (counters only ever grow; a change means a click)."""
    if reset_clicks != state.reset_clicks:
        state.reset_clicks = reset_clicks
        state.offsets = {k: v for k, v in state.offsets.items() if k[1] != state.model}
        state.neutral_raw = {k: v for k, v in state.neutral_raw.items() if k[1] != state.model}
        state.calibration = None
        state.message = ""
        for t in state.tracks:
            t.ema, t.trail = None, []
    if calibrate_clicks != state.calibrate_clicks:
        state.calibrate_clicks = calibrate_clicks
        state.calibration = CalibrationRun(start=time.perf_counter())


def calibration_status(state: LiveState) -> tuple[str, str] | None:
    """(text, kind) to show now, or None."""
    run = state.calibration
    if run is not None:
        elapsed = time.perf_counter() - run.start
        if elapsed < COUNTDOWN_S:
            return f"calibrating in {int(COUNTDOWN_S - elapsed) + 1}…", "info"
        return "calibrating… keep still", "info"
    if state.message and time.perf_counter() < state.message_until:
        return state.message, state.message_kind
    return None


def countdown_number(state: LiveState) -> int | None:
    """Seconds left in the countdown (0 while collecting), or None if not calibrating."""
    run = state.calibration
    if run is None:
        return None
    elapsed = time.perf_counter() - run.start
    return int(COUNTDOWN_S - elapsed) + 1 if elapsed < COUNTDOWN_S else 0


def calibration_step(
    state: LiveState,
    seen: list[tuple[Track, FaceResult, Point]],
    neutral_mean: NDArray[np.float64],
    p99: float | None,
) -> None:
    """Advance a calibration run with this frame's faces; finish it when ready.

    The offset is identity-only: ``mean(raw neutral) - neutral_mean``. A person's neutral
    face is aligned with the average neutral face of unseen people, which removes their
    resting-face offset without shifting every expression by the model's general bias
    at neutral (measured on MorphSet; see the model card).
    """
    run = state.calibration
    if run is None:
        return
    elapsed = time.perf_counter() - run.start
    if elapsed < COUNTDOWN_S:
        return
    for track, face, _ in seen:
        if track.raw is not None:
            run.samples.setdefault(track.id, []).append(track.raw)
            if face.distance is not None:
                run.distances.setdefault(track.id, []).append(face.distance)
    collected = max((len(v) for v in run.samples.values()), default=0)
    if elapsed < COUNTDOWN_S + COLLECT_S or (collected < MIN_SAMPLES and elapsed < TIMEOUT_S):
        return

    state.calibration = None
    done, rejected = [], []
    for tid, samples in sorted(run.samples.items()):
        if len(samples) < 3:
            continue
        dist = run.distances.get(tid, [])
        if p99 is not None and dist and float(np.median(dist)) > p99:
            rejected.append(f"face {tid}: poor fit")
            continue
        own_neutral = np.mean(samples, axis=0)
        offset = own_neutral - neutral_mean
        if float(np.hypot(offset[0], offset[1])) > MAX_OFFSET:
            rejected.append(f"face {tid}: not neutral")
            continue
        state.offsets[(tid, state.model)] = offset
        state.neutral_raw[(tid, state.model)] = own_neutral
        for t in state.tracks:  # restart smoothing from the calibrated value
            if t.id == tid:
                t.ema, t.trail = None, []
        done.append(tid)
    # success shows as the permanent "calibrated" tag; only problems get a (temporary) tag
    many = len(run.samples) > 1
    if rejected:
        text = "; ".join(r if many else r.split(": ", 1)[1] for r in rejected)
        _say(state, f"calibration failed: {text}, try again", "warn", 5.0)
    elif not done:
        _say(state, "calibration failed: no face, try again", "warn", 5.0)


def is_calibrated(state: LiveState, track_id: int) -> bool:
    return (track_id, state.model) in state.offsets
