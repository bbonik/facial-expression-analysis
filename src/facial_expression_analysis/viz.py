"""Drawing helpers (OpenCV for images, matplotlib for figures)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .constants import FACE_CONTOURS

if TYPE_CHECKING:
    from .pipeline import FaceAnalysis

Color = tuple[int, int, int]

# Colour of a face annotation as a function of valence (RGB).
_NEGATIVE: Color = (230, 80, 70)
_NEUTRAL: Color = (240, 240, 240)
_POSITIVE: Color = (70, 200, 120)


def valence_color(valence: float) -> Color:
    """Interpolate red (negative) -> white (neutral) -> green (positive)."""
    v = float(np.clip(valence, -1, 1))
    end = _POSITIVE if v >= 0 else _NEGATIVE
    t = abs(v)
    return tuple(round((1 - t) * n + t * e) for n, e in zip(_NEUTRAL, end, strict=True))  # type: ignore[return-value]


def draw_landmarks(
    image: NDArray[np.uint8], landmarks: NDArray[np.floating], color: Color, thickness: int = 1
) -> None:
    """Draw a 68-point face as poly-lines, in place."""
    import cv2

    pts = np.round(landmarks).astype(np.int32)
    for part, closed in FACE_CONTOURS:
        cv2.polylines(image, [pts[part.start : part.stop]], closed, color, thickness, cv2.LINE_AA)


def annotate_image(
    image: NDArray[np.uint8], faces: list[FaceAnalysis], *, landmarks: bool = True
) -> NDArray[np.uint8]:
    """Return a copy of an RGB image with boxes, landmarks and emotion labels."""
    import cv2

    out = image.copy()
    scale = max(out.shape[:2]) / 1000
    thickness = max(1, round(2 * scale))
    font_scale = max(0.4, 0.6 * scale)
    for face in faces:
        e = face.emotion
        color = valence_color(e.valence)
        b = face.box
        cv2.rectangle(out, (b.left, b.top), (b.right, b.bottom), color, thickness, cv2.LINE_AA)
        if landmarks:
            draw_landmarks(out, e.landmarks, color, max(1, thickness // 2))
        lines = [str(e.label), f"A {e.arousal:+.2f}  V {e.valence:+.2f}  I {e.intensity:.2f}"]
        y = b.bottom
        for text in lines:
            (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(out, (b.left, y), (b.left + w + 8, y + h + base + 8), (20, 20, 20), -1)
            cv2.putText(
                out, text, (b.left + 4, y + h + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA,
            )  # fmt: skip
            y += h + base + 8
    return out


def plot_landmarks(
    landmarks: NDArray[np.floating],
    ax: Any = None,
    color: str = "k",
    title: str | None = None,
) -> Any:
    """Draw a 68-point face shape on a matplotlib axis (created if ``None``)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    for part, closed in FACE_CONTOURS:
        idx = list(part) + ([part.start] if closed else [])
        ax.plot(landmarks[idx, 0], landmarks[idx, 1], color=color, lw=1.5)
    ax.set_aspect("equal")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax
