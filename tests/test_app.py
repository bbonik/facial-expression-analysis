"""Tests of the web demo's rendering and analysis helpers (no Gradio needed)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from .conftest import IMAGES, requires_dlib_model

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))

from core import FaceResult, LiveState, largest, update_live
from render import (
    SECTORS,
    Point,
    circumplex_svg,
    draw_overlay,
    faces_grid,
    fit_badge,
    metrics_card,
    valence_hex,
)


def test_sectors_cover_the_circle_with_24_words() -> None:
    assert len(SECTORS) == 24
    assert len({w for w, _, _ in SECTORS}) == 24
    assert sum(e - s for _, s, e in SECTORS) == pytest.approx(360)


def test_circumplex_svg_is_valid_xml_and_highlights_the_word() -> None:
    svg = circumplex_svg([Point(0.3, 0.6, 0.67, "Very happy")], [(0.0, 0.0), (0.3, 0.6)])
    root = ET.fromstring(svg)
    texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
    assert "happy" in texts
    bold = [
        t.text for t in root.iter("{http://www.w3.org/2000/svg}text")
        if t.get("font-weight") == "700"
    ]  # fmt: skip
    assert bold == ["happy"]  # the current sector's word is highlighted
    ET.fromstring(circumplex_svg())  # empty state too


def test_valence_colours() -> None:
    assert valence_hex(1.0) == "#34d399"
    assert valence_hex(-1.0) == "#fb7185"
    assert valence_hex(0.0) == "#cbd5e1"


def test_cards() -> None:
    card = metrics_card(Point(0.1, -0.5, 0.51, "Moderately sad"), model_name="PLS(31)",
                        fit=(30.0, 16.0, 26.0))  # fmt: skip
    assert "Moderately sad" in card
    assert "fea-badge-warn" in card
    assert "+0.51" not in card
    assert "0.51" in card
    assert "fea-empty" in metrics_card(None, model_name="x")
    assert "fea-badge-ok" in fit_badge(10.0, 16.0, 26.0)
    assert fit_badge(None, 16.0, 26.0) == ""
    grid = faces_grid([(Point(0.2, 0.7, 0.73, "Very happy", "1"), ""),
                       (Point(0.5, -0.4, 0.64, "Very annoyed", "2"), "")])  # fmt: skip
    assert grid.count('class="fea-face"') == 2
    assert "Upload a photo" in faces_grid([])


def test_overlay_keeps_shape(landmarks: np.ndarray) -> None:
    img = np.zeros((300, 400, 3), np.uint8)
    lm = landmarks[0] - landmarks[0].min(0) + 60
    out = draw_overlay(img, [(50, 50, 250, 260)], [lm], [Point(0.2, 0.5, 0.5, "Happy", "1")])
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.any()


def _face(a: float, v: float, size: int = 100) -> FaceResult:
    return FaceResult((0, 0, size, size), np.zeros((68, 2)), Point(a, v, abs(v), "x"), 10.0)


def test_live_smoothing_trail_and_model_reset() -> None:
    state = LiveState(model="m1")
    state, p = update_live(state, _face(0.0, 0.0), "m1", 0.5)
    state, p = update_live(state, _face(1.0, 1.0), "m1", 0.5)
    assert p is not None
    assert p.arousal == pytest.approx(0.5)
    assert len(state.trail) == 2
    state, p = update_live(state, _face(1.0, 1.0), "m2", 0.5)  # model change resets
    assert p is not None
    assert p.arousal == pytest.approx(1.0)
    assert len(state.trail) == 1
    state, p = update_live(state, None, "m2", 0.5)
    assert p is None
    assert len(state.trail) == 1


def test_largest_face() -> None:
    assert largest([]) is None
    assert largest([_face(0, 0, 50), _face(0, 0, 120)]).box[2] == 120  # type: ignore[union-attr]


@pytest.mark.slow
@requires_dlib_model
def test_engine_on_sample_photo() -> None:
    from core import DEFAULT_MODEL, Engine

    from facial_expression_analysis import load_image

    engine = Engine()
    faces = engine.analyze(load_image(IMAGES / "kids.jpg")[..., :3], DEFAULT_MODEL, photo=True)
    assert len(faces) == 2
    assert [f.point.tag for f in faces] == ["1", "2"]
    assert all(f.distance is not None and f.distance > 0 for f in faces)
    assert engine.p99 is not None
