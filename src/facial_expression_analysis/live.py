"""Lightweight real-time OpenCV demo (the web demo lives in ``app/``)."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .labels import describe_emotion
from .landmarks import DlibLandmarkDetector
from .model import load_model
from .pipeline import EmotionEstimator, largest_face
from .viz import annotate_image, valence_color

WINDOW = "Facial expression analysis (q / Esc to quit)"
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


def run_webcam(
    camera: int = 0, smoothing: float = 0.3, model_path: str | Path | None = None
) -> None:
    """Run the live demo until the user presses q/Esc or closes the window."""
    import cv2

    estimator = EmotionEstimator(load_model(model_path))
    detector = DlibLandmarkDetector(upsample=0, max_width=640)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")

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
            faces = estimator.analyze_image(rgb, detector)
            face = largest_face(faces)
            if face is not None:
                e = face.emotion
                current = np.array([e.arousal, e.valence, e.intensity])
                state = current if state is None else smoothing * state + (1 - smoothing) * current
                trail.append((float(state[0]), float(state[1]), float(state[2])))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
            last = now

            canvas = np.hstack([annotate_image(rgb, faces), _av_panel(rgb.shape[0], trail, fps)])
            cv2.imshow(WINDOW, canvas[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
