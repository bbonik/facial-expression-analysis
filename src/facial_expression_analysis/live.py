"""Lightweight real-time OpenCV demo (the web demo lives in ``app/``)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .labels import describe_emotion
from .landmarks import DlibLandmarkDetector, LandmarkTracker
from .model import EmotionModel, KernelEmotionModel, load_model
from .pipeline import EmotionEstimator, largest_face
from .viz import annotate_image, valence_color

WINDOW = "Facial expression analysis"
PANEL = 300


def _av_panel(
    height: int, trail: deque[tuple[float, float, float]], fps: float
) -> NDArray[np.uint8]:
    """Draw the arousal-valence plane with a fading trail (RGB)."""
    import cv2

    panel = np.full((height, PANEL, 3), 24, dtype=np.uint8)
    size = max(120, min(PANEL - 40, height - 200))
    x0, y0 = 20, 50
    cx, cy = x0 + size // 2, y0 + size // 2
    cv2.circle(panel, (cx, cy), size // 2, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.line(panel, (x0, cy), (x0 + size, cy), (90, 90, 90), 1)
    cv2.line(panel, (cx, y0), (cx, y0 + size), (90, 90, 90), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "Arousal-Valence", (x0, 30), font, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    for text, pos in (
        ("+A", (cx + 4, y0 + 14)), ("-A", (cx + 4, y0 + size - 4)),
        ("-V", (x0 + 2, cy - 6)), ("+V", (x0 + size - 22, cy - 6)),
    ):  # fmt: skip
        cv2.putText(panel, text, pos, font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    def to_px(a: float, v: float) -> tuple[int, int]:
        return round(cx + v * size / 2), round(cy - a * size / 2)

    pts = list(trail)
    for k, (a, v, _) in enumerate(pts):
        alpha = (k + 1) / len(pts)
        color = tuple(round(c * alpha) for c in valence_color(v))
        cv2.circle(panel, to_px(a, v), 2 + round(4 * alpha), color, -1, cv2.LINE_AA)

    y = y0 + size + 40
    if pts:
        a, v, i = pts[-1]
        cv2.putText(panel, str(describe_emotion(a, v, i)), (x0, y), font, 0.6,
                    valence_color(v), 1, cv2.LINE_AA)  # fmt: skip
        for name, val, lo in (("Arousal", a, -1), ("Valence", v, -1), ("Intensity", i, 0)):
            y += 30
            cv2.putText(panel, f"{name:<9} {val:+.2f}", (x0, y), font, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)  # fmt: skip
            bx0, bx1 = x0 + 140, x0 + size
            zero = bx0 if lo == 0 else (bx0 + bx1) // 2
            end = round(bx0 + (val - lo) / (1 - lo) * (bx1 - bx0))
            cv2.rectangle(panel, (bx0, y - 10), (bx1, y), (60, 60, 60), -1)
            cv2.rectangle(panel, (min(zero, end), y - 10), (max(zero, end), y),
                          valence_color(v), -1)  # fmt: skip
    cv2.putText(panel, f"{fps:5.1f} FPS", (PANEL - 85, 30), font, 0.45, (150, 150, 150), 1,
                cv2.LINE_AA)  # fmt: skip
    return panel


def _describe(model: EmotionModel) -> str:
    if isinstance(model, KernelEmotionModel):
        return "kernel ridge"
    return str(model.metadata.get("estimator", "linear")).replace(
        "PLSRegression(n_components=", "PLS("
    )


def _test_ccc(model: EmotionModel) -> str:
    try:
        r = model.metadata["metrics"]["test"]["pipeline"]
        return "/".join(f"{r[t]['ccc']:.2f}" for t in ("arousal", "valence", "intensity"))
    except (KeyError, TypeError):
        return "n/a"


@dataclass
class DemoSettings:
    """Live-adjustable settings of the webcam demo (all shown in the status bar)."""

    camera: int = 0
    smoothing: float = 0.3
    landmarks: bool = True
    upsample: int = 0
    max_width: int = 640
    tracking: bool = True


@dataclass
class FitCheck:
    """Distance of the face to the training faces, with its reference quantiles."""

    distance: float | None = None
    p50: float | None = None
    p99: float | None = None

    @property
    def reliable(self) -> bool | None:
        if self.distance is None or self.p99 is None:
            return None
        return self.distance <= self.p99


STATUS_H = 92
GREEN, ORANGE, GREY, BLUE = (110, 220, 110), (255, 160, 50), (170, 170, 170), (120, 200, 255)


def _status_bar(
    width: int,
    settings: DemoSettings,
    model: EmotionModel,
    fit: FitCheck,
    fps: float,
    n_faces: int,
) -> NDArray[np.uint8]:
    """Draw every setting and live diagnostic under the video (RGB)."""
    import cv2

    bar = np.full((STATUS_H, width, 3), 16, dtype=np.uint8)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.47

    def row(y: int, items: list[tuple[str, tuple[int, int, int]]]) -> None:
        x = 12
        for text, color in items:
            cv2.putText(bar, text, (x, y), font, fs, color, 1, cv2.LINE_AA)
            x += cv2.getTextSize(text, font, fs, 1)[0][0] + 22

    row(22, [
        (f"Model: {_describe(model)} [m]", BLUE),
        (f"test CCC A/V/I {_test_ccc(model)}", GREY),
        (f"Smoothing: {settings.smoothing:.1f} [-/+]", BLUE),
        (f"Landmarks: {'on' if settings.landmarks else 'off'} [l]", BLUE),
    ])  # fmt: skip
    if fit.distance is None:
        fit_text, fit_color = "Fit distance: n/a (no face)", GREY
    else:
        verdict = {True: "OK", False: "UNRELIABLE (unusual face or bad landmark fit)", None: ""}
        fit_text = (
            f"Fit distance: {fit.distance:.1f}  (training median {fit.p50:.1f}, "
            f"99th pct {fit.p99:.1f})  {verdict[fit.reliable]}"
        )
        fit_color = {True: GREEN, False: ORANGE, None: GREY}[fit.reliable]
    row(50, [(fit_text, fit_color)])
    row(78, [
        (f"Camera {settings.camera}", GREY),
        (f"dlib upsample {settings.upsample}, width {settings.max_width}px", GREY),
        (f"tracking {'on' if settings.tracking else 'off'} (t)", GREY),
        (f"Faces: {n_faces}", GREY),
        (f"{fps:.1f} FPS", GREY),
        ("q / Esc: quit", GREY),
    ])  # fmt: skip
    return bar


CAMERA_HELP = """Cannot open camera {camera}.

macOS: the terminal app you run this from needs camera access.
  1. System Settings > Privacy & Security > Camera: enable your terminal
     (Terminal, iTerm, VS Code, ...). If it is not listed, run the demo once more
     and click "Allow" in the dialog.
  2. Quit and reopen the terminal app (permissions apply to new processes only).
Otherwise: close other apps using the camera, or try another index (--camera 1)."""


def _open_camera(camera: int, wait_s: float = 15.0) -> Any:
    """Open a camera, retrying while a macOS permission dialog may be pending."""
    import cv2

    deadline = time.monotonic() + wait_s
    while True:
        cap = cv2.VideoCapture(camera)
        if cap.isOpened():
            return cap
        cap.release()
        if time.monotonic() > deadline:
            raise CameraError(CAMERA_HELP.format(camera=camera))
        time.sleep(1.0)


class CameraError(RuntimeError):
    """The camera could not be opened (permissions, busy, wrong index)."""


def run_webcam(
    camera: int = 0,
    smoothing: float = 0.3,
    model_path: str | Path | None = None,
    upsample: int = 0,
    max_width: int = 640,
    tracking: bool = True,
) -> None:
    """Run the live demo until the user presses q/Esc or closes the window.

    Keys: ``m`` switches between the given (default: PLS(31)) model and the bundled
    kernel model on the same live input; ``-``/``+`` change the smoothing;
    ``l`` toggles the landmark overlay; ``t`` toggles landmark tracking (boxes from
    the previous frame's landmarks and low-light robust fits, see
    :class:`~facial_expression_analysis.landmarks.LandmarkTracker`) versus running the
    face detector on every frame.
    """
    import cv2

    settings = DemoSettings(camera, smoothing, True, upsample, max_width, tracking)
    models: list[EmotionModel] = [load_model(model_path), KernelEmotionModel.load()]
    # The fit check describes the face, not the model: use a kernel model whenever one exists.
    kernel = next((m for m in models if isinstance(m, KernelEmotionModel)), None)
    quantiles = kernel.metadata.get("basis_distance_quantiles", {}) if kernel else {}
    current_model = 0
    estimator = EmotionEstimator(models[current_model])
    detector = DlibLandmarkDetector(upsample=upsample, max_width=max_width)
    tracker = LandmarkTracker(
        DlibLandmarkDetector(upsample=upsample, max_width=max_width, robust=True)
    )
    cap = _open_camera(camera)

    trail: deque[tuple[float, float, float]] = deque(maxlen=30)
    state: NDArray[np.float64] | None = None
    fps, last = 0.0, time.perf_counter()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(cv2.flip(frame, 1)[:, :, ::-1], dtype=np.uint8)
            faces = estimator.analyze_image(rgb, tracker if settings.tracking else detector)
            face = largest_face(faces)
            fit = FitCheck(p50=quantiles.get("0.5"), p99=quantiles.get("0.99"))
            if face is not None:
                if kernel is not None:
                    feats = kernel.features.transform(face.emotion.frontal_landmarks[None])
                    fit.distance = float(kernel.basis_distance(feats)[0])
                e = face.emotion
                current = np.array([e.arousal, e.valence, e.intensity])
                s_ = settings.smoothing
                state = current if state is None else s_ * state + (1 - s_) * current
                trail.append((float(state[0]), float(state[1]), float(state[2])))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now

            video = annotate_image(rgb, faces, landmarks=settings.landmarks)
            top = np.hstack([video, _av_panel(rgb.shape[0], trail, fps)])
            bar = _status_bar(top.shape[1], settings, estimator.model, fit, fps, len(faces))
            cv2.imshow(WINDOW, np.vstack([top, bar])[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("m"):
                current_model = (current_model + 1) % len(models)
                estimator = EmotionEstimator(models[current_model], estimator.frontalizer)
                trail.clear()
                state = None
            elif key in (ord("+"), ord("=")):
                settings.smoothing = min(0.9, round(settings.smoothing + 0.1, 1))
            elif key in (ord("-"), ord("_")):
                settings.smoothing = max(0.0, round(settings.smoothing - 0.1, 1))
            elif key == ord("l"):
                settings.landmarks = not settings.landmarks
            elif key == ord("t"):
                settings.tracking = not settings.tracking
                tracker.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()
