"""Pure rendering helpers for the web demo: SVG circumplex, HTML cards, image overlay."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from facial_expression_analysis.constants import FACE_CONTOURS
from facial_expression_analysis.labels import _ANGLE_BOUNDS, _EMOTION_NAMES

# --------------------------------------------------------------------------- palette
NEGATIVE = (251, 113, 133)  # rose-400
NEUTRAL = (203, 213, 225)  # slate-300
POSITIVE = (52, 211, 153)  # emerald-400
ACCENT = "#a78bfa"  # violet-400


def valence_rgb(valence: float) -> tuple[int, int, int]:
    """Rose (negative) -> slate (neutral) -> emerald (positive)."""
    v = max(-1.0, min(1.0, float(valence)))
    end = POSITIVE if v >= 0 else NEGATIVE
    t = abs(v) ** 0.8
    return tuple(round((1 - t) * n + t * e) for n, e in zip(NEUTRAL, end, strict=True))  # type: ignore[return-value]


def _hex_rgb(colour: str) -> tuple[int, int, int]:
    c = colour.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def valence_hex(valence: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(*valence_rgb(valence))


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Point:
    """A face's estimate as shown on the circumplex."""

    arousal: float
    valence: float
    intensity: float
    label: str
    tag: str = ""  # face number when several faces are shown
    accent: str = ""  # per-face colour (hex) when several faces are shown


def _sectors() -> list[tuple[str, float, float]]:
    """(word, start, end) in degrees for the 24 circumplex sectors."""
    starts = [_ANGLE_BOUNDS[-1] - 360, *_ANGLE_BOUNDS[:-1]]
    return [
        (_EMOTION_NAMES[i], float(s), float(e))
        for i, (s, e) in enumerate(zip(starts, _ANGLE_BOUNDS, strict=True))
    ]


SECTORS = _sectors()


def _sector_of(angle: float) -> int:
    a = angle % 360
    for i, (_, s, e) in enumerate(SECTORS):
        if s <= a < e or s <= a - 360 < e:
            return i
    return 0


# --------------------------------------------------------------------------- circumplex
def circumplex_svg(
    points: Sequence[Point] = (),
    trails: Sequence[tuple[Sequence[tuple[float, float]], str]] = (),
    *,
    size: int = 420,
) -> str:
    """Russell's circumplex with the 24 emotion words, fading trails and the current points.

    ``trails`` holds one ``((arousal, valence) history, colour)`` per tracked face.

    Coordinates: x = valence, y = -arousal, unit circle = maximal intensity.
    """
    R = 1.0
    parts: list[str] = [
        f'<svg viewBox="-1.62 -1.62 3.24 3.24" width="100%" style="max-width:{size}px" '
        'xmlns="http://www.w3.org/2000/svg" class="fea-circumplex" role="img" '
        'aria-label="Arousal-valence circumplex">',
        "<defs>"
        '<radialGradient id="fea-bg" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#1e1b4b" stop-opacity="0.9"/>'
        '<stop offset="100%" stop-color="#0b1020" stop-opacity="0.2"/></radialGradient>'
        '<linearGradient id="fea-ring" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0%" stop-color="rgb{NEGATIVE}"/>'
        f'<stop offset="50%" stop-color="rgb{NEUTRAL}"/>'
        f'<stop offset="100%" stop-color="rgb{POSITIVE}"/></linearGradient>'
        '<filter id="fea-glow" x="-200%" y="-200%" width="500%" height="500%">'
        '<feGaussianBlur stdDeviation="0.045" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        "</defs>",
        f'<circle r="{R}" fill="url(#fea-bg)"/>',
    ]

    current = points[-1] if len(points) == 1 else None
    active = (
        _sector_of(math.degrees(math.atan2(current.arousal, current.valence)))
        if (current and current.intensity >= 0.1)
        else None
    )

    # highlighted sector of the current estimate
    if active is not None and current is not None:
        _, s, e = SECTORS[active]
        a0, a1 = math.radians(s), math.radians(e)
        x0, y0, x1, y1 = R * math.cos(a0), -R * math.sin(a0), R * math.cos(a1), -R * math.sin(a1)
        parts.append(
            f'<path d="M0 0 L{x0:.4f} {y0:.4f} A{R} {R} 0 0 0 {x1:.4f} {y1:.4f} Z" '
            f'fill="{valence_hex(current.valence)}" fill-opacity="0.13"/>'
        )

    # sector spokes, intensity rings, axes
    for _, s, _e in SECTORS:
        a = math.radians(s)
        parts.append(
            f'<line x1="{0.1 * math.cos(a):.4f}" y1="{-0.1 * math.sin(a):.4f}" '
            f'x2="{R * math.cos(a):.4f}" y2="{-R * math.sin(a):.4f}" '
            'stroke="#94a3b8" stroke-opacity="0.08" stroke-width="0.006"/>'
        )
    for r, name in ((0.325, "slightly"), (0.55, "moderately"), (0.775, "very")):
        parts.append(
            f'<circle r="{r}" fill="none" stroke="#94a3b8" stroke-opacity="0.16" '
            'stroke-width="0.006" stroke-dasharray="0.02 0.03"/>'
            f'<text x="0.02" y="{-r - 0.015:.3f}" font-size="0.05" fill="#94a3b8" '
            f'fill-opacity="0.45">{name}</text>'
        )
    parts.append(
        '<circle r="0.1" fill="#94a3b8" fill-opacity="0.07"/>'
        f'<circle r="{R}" fill="none" stroke="url(#fea-ring)" stroke-width="0.022" '
        'stroke-opacity="0.85"/>'
        '<line x1="-1.05" y1="0" x2="1.05" y2="0" stroke="#cbd5e1" stroke-opacity="0.3" '
        'stroke-width="0.007"/>'
        '<line x1="0" y1="-1.05" x2="0" y2="1.05" stroke="#cbd5e1" stroke-opacity="0.3" '
        'stroke-width="0.007"/>'
    )
    axis = 'font-size="0.068" font-weight="600" fill="#e2e8f0" fill-opacity="0.75"'
    parts.append(
        f'<text x="0" y="-1.5" text-anchor="middle" {axis}>AROUSAL +</text>'
        f'<text x="0" y="1.56" text-anchor="middle" {axis}>AROUSAL −</text>'
        f'<text x="1.5" y="0" text-anchor="middle" {axis} '
        'transform="rotate(90 1.5 0)">VALENCE +</text>'
        f'<text x="-1.5" y="0" text-anchor="middle" {axis} '
        'transform="rotate(-90 -1.5 0)">VALENCE −</text>'
    )

    # emotion words, written radially like a dial
    for i, (word, s, e) in enumerate(SECTORS):
        mid = (s + e) / 2
        a = math.radians(mid)
        r = 1.06
        x, y = r * math.cos(a), -r * math.sin(a)
        left = 90 < mid % 360 < 270
        rot = -mid + (180 if left else 0)
        is_active = i == active
        fill = valence_hex(current.valence) if is_active and current else "#cbd5e1"
        weight = "700" if is_active else "500"
        opacity = "1" if is_active else "0.62"
        parts.append(
            f'<text x="{x:.4f}" y="{y:.4f}" font-size="{0.078 if is_active else 0.064}" '
            f'font-weight="{weight}" fill="{fill}" fill-opacity="{opacity}" '
            f'text-anchor="{"end" if left else "start"}" dominant-baseline="middle" '
            f'transform="rotate({rot:.2f} {x:.4f} {y:.4f})">{word}</text>'
        )

    # trails
    for trail, colour in trails:
        n = len(trail)
        for k in range(1, n):
            (a0_, v0), (a1_, v1) = trail[k - 1], trail[k]
            parts.append(
                f'<line x1="{v0:.4f}" y1="{-a0_:.4f}" x2="{v1:.4f}" y2="{-a1_:.4f}" '
                f'stroke="{colour or ACCENT}" stroke-opacity="{0.08 + 0.5 * k / n:.3f}" '
                'stroke-width="0.016" stroke-linecap="round"/>'
            )

    # current points
    for p in points:
        c = valence_hex(p.valence)
        parts.append(
            f'<circle cx="{p.valence:.4f}" cy="{-p.arousal:.4f}" r="0.075" fill="{c}" '
            'fill-opacity="0.25" filter="url(#fea-glow)"/>'
            f'<circle cx="{p.valence:.4f}" cy="{-p.arousal:.4f}" r="0.04" fill="{c}" '
            f'stroke="{p.accent or "#0b1020"}" stroke-width="{0.018 if p.accent else 0.012}"/>'
        )
        if p.tag:
            parts.append(
                f'<text x="{p.valence + 0.06:.4f}" y="{-p.arousal - 0.06:.4f}" '
                f'font-size="0.08" font-weight="800" fill="{p.accent or c}">'
                f"{html.escape(p.tag)}</text>"
            )
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- cards
def _bar(name: str, value: float, bipolar: bool, color: str) -> str:
    shown = f"{value:+.2f}" if bipolar else f"{value:.2f}"
    if bipolar:
        left = 50 + min(0.0, value) * 50
        width = abs(value) * 50
        centre = '<div class="fea-bar-centre"></div>'
    else:
        left, width, centre = 0.0, max(0.0, value) * 100, ""
    return (
        f'<div class="fea-metric"><span class="fea-metric-name">{name}</span>'
        f'<div class="fea-bar">{centre}<div class="fea-bar-fill" '
        f'style="left:{left:.1f}%;width:{width:.1f}%;background:{color}"></div></div>'
        f'<span class="fea-metric-value">{shown}</span></div>'
    )


def status_banner(text: str, kind: str = "info") -> str:
    """Calibration progress / result shown above the estimates."""
    return f'<div class="fea-status fea-status-{kind}">{html.escape(text)}</div>'


CALIBRATED_CHIP = '<span class="fea-chip">calibrated</span>'


def fit_badge(distance: float | None, p50: float | None, p99: float | None) -> str:
    if distance is None or p99 is None:
        return ""
    ok = distance <= p99
    cls, text = (
        ("ok", "Reliable landmark fit")
        if ok
        else ("warn", "Unusual face or poor landmark fit: treat with caution")
    )
    ref = f"median {p50:.1f}, 99th pct {p99:.1f}" if p50 is not None else f"99th pct {p99:.1f}"
    return (
        f'<div class="fea-badge fea-badge-{cls}"><span class="fea-dot"></span>{text}'
        f'<span class="fea-badge-sub">distance to training faces {distance:.1f} '
        f"({ref})</span></div>"
    )


LIVE_HINT = ("No face detected yet", "Start the camera and face it in good, even light.")
PHOTO_HINT = ("No face to show yet", "Upload a photo or pick one of the examples.")


def empty_card(hint: tuple[str, str] = LIVE_HINT) -> str:
    title, sub = (html.escape(t) for t in hint)
    return (
        '<div class="fea-card fea-empty"><div class="fea-empty-icon">◎</div>'
        f'<div>{title}</div><div class="fea-muted">{sub}</div></div>'
    )


def metrics_card(
    point: Point | None,
    *,
    model_name: str,
    fit: tuple[float | None, float | None, float | None] = (None, None, None),
    footer: str = "",
    hint: tuple[str, str] = LIVE_HINT,
    calibrated: bool = False,
) -> str:
    """Large label, A/V/I bars and the fit badge for the tracked face."""
    if point is None:
        return empty_card(hint)
    color = valence_hex(point.valence)
    bars = (
        _bar("Arousal", point.arousal, True, color)
        + _bar("Valence", point.valence, True, color)
        + _bar("Intensity", point.intensity, False, color)
    )
    return (
        '<div class="fea-card">'
        f'<div class="fea-kicker">{html.escape(model_name)}'
        + (" " + CALIBRATED_CHIP if calibrated else "")
        + "</div>"
        f'<div class="fea-label" style="color:{color}">{html.escape(point.label)}</div>'
        f'<div class="fea-metrics">{bars}</div>'
        f"{fit_badge(*fit)}"
        + (f'<div class="fea-footer">{footer}</div>' if footer else "")
        + "</div>"
    )


def faces_grid(rows: Sequence[tuple[Point, str]]) -> str:
    """One compact card per face (photo mode); ``rows`` = (point, fit badge html)."""
    if not rows:
        return empty_card(PHOTO_HINT)
    cards = []
    for p, badge in rows:
        color = valence_hex(p.valence)
        cards.append(
            '<div class="fea-face">'
            f'<div class="fea-face-tag" style="background:{p.accent or color}">'
            f"{html.escape(p.tag)}</div>"
            f'<div class="fea-face-label" style="color:{color}">{html.escape(p.label)}</div>'
            f'<div class="fea-face-nums">A {p.arousal:+.2f} · V {p.valence:+.2f} · '
            f"I {p.intensity:.2f}</div>{badge}</div>"
        )
    return f'<div class="fea-faces">{"".join(cards)}</div>'


# --------------------------------------------------------------------------- overlay
def draw_countdown(rgb: NDArray[np.uint8], number: int, text: str) -> NDArray[np.uint8]:
    """Dim the frame and draw a large countdown number (0: only the text) in the centre."""
    import cv2

    h, w = rgb.shape[:2]
    out = cv2.addWeighted(rgb, 0.55, np.zeros_like(rgb), 0.45, 0)
    scale = max(h, w) / 640
    if number > 0:
        label = str(number)
        fs = 4.0 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, max(2, int(6 * scale)))
        cv2.putText(out, label, ((w - tw) // 2, (h + th) // 2 - int(20 * scale)),
                    cv2.FONT_HERSHEY_DUPLEX, fs, (196, 181, 253), max(2, int(6 * scale)),
                    cv2.LINE_AA)  # fmt: skip
    fs = 0.75 * scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    y = (h + th) // 2 + int(70 * scale) if number > 0 else (h + th) // 2
    cv2.putText(out, text, ((w - tw) // 2, y), cv2.FONT_HERSHEY_DUPLEX, fs, (240, 244, 255), 1,
                cv2.LINE_AA)  # fmt: skip
    return out


def draw_overlay(
    rgb: NDArray[np.uint8],
    boxes: Sequence[tuple[int, int, int, int]],
    landmarks: Sequence[NDArray[np.floating]],
    points: Sequence[Point],
    *,
    show_landmarks: bool = True,
) -> NDArray[np.uint8]:
    """Draw corner brackets, a landmark mesh and a label pill per face (RGB in and out)."""
    import cv2

    out = rgb.copy()
    scale = max(out.shape[:2]) / 900
    th = max(1, round(2 * scale))
    for (x0, y0, x1, y1), lm, p in zip(boxes, landmarks, points, strict=True):
        color = valence_rgb(p.valence)
        # corner brackets instead of a full rectangle
        L = max(8, int(0.22 * min(x1 - x0, y1 - y0)))
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(out, (cx, cy), (cx + dx * L, cy), color, th + 1, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * L), color, th + 1, cv2.LINE_AA)
        if show_landmarks:
            pts = np.round(lm).astype(np.int32)
            mesh = out.copy()
            for part, closed in FACE_CONTOURS:
                cv2.polylines(mesh, [pts[list(part)]], closed, color, max(1, th - 1), cv2.LINE_AA)
            out = cv2.addWeighted(mesh, 0.55, out, 0.45, 0)
            for x, y in pts:
                cv2.circle(out, (int(x), int(y)), max(1, th - 1), (255, 255, 255), -1, cv2.LINE_AA)
        # label pill above the box
        text = f"{p.tag + '  ' if p.tag else ''}{p.label}"
        fs = max(0.45, 0.62 * scale)
        (tw, tht), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
        pad = int(8 * max(scale, 0.8))
        bx0, by1 = x0, max(tht + 2 * pad, y0 - pad)
        bx1, by0 = x0 + tw + 2 * pad, by1 - tht - 2 * pad
        pill = out.copy()
        cv2.rectangle(pill, (bx0, by0), (bx1, by1), (15, 18, 32), -1, cv2.LINE_AA)
        out = cv2.addWeighted(pill, 0.78, out, 0.22, 0)
        bar = _hex_rgb(p.accent) if p.accent else color
        cv2.rectangle(out, (bx0, by0), (bx0 + max(4, pad // 2 + 2), by1), bar, -1)
        cv2.putText(out, text, (bx0 + pad + 2, by1 - pad), cv2.FONT_HERSHEY_DUPLEX, fs,
                    (240, 244, 255), 1, cv2.LINE_AA)  # fmt: skip
    return out
