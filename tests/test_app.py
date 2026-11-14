"""Tests of the web demo's rendering and analysis helpers (no Gradio needed)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from .conftest import IMAGES, requires_dlib_model

sys.path.insert(0, str(Path(__file__).parents[1] / "app"))

from core import FaceResult, LiveState, largest, update_tracks
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
    svg = circumplex_svg([Point(0.3, 0.6, 0.67, "Very happy")], [([(0.0, 0.0), (0.3, 0.6)], "")])
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


def _face(a: float, v: float, size: int = 100, x: int = 0) -> FaceResult:
    return FaceResult((x, 0, x + size, size), np.zeros((68, 2)), Point(a, v, abs(v), "x"), 10.0)


def test_single_face_smoothing_trail_and_model_reset() -> None:
    state = LiveState(model="m1")
    state, seen = update_tracks(state, [_face(0.0, 0.0)], "m1", 0.5)
    state, seen = update_tracks(state, [_face(1.0, 1.0)], "m1", 0.5)
    (track, _, p) = seen[0]
    assert p.arousal == pytest.approx(0.5)  # halfway with smoothing 0.5
    assert p.tag == ""  # a single face carries no number
    assert len(track.trail) == 2
    state, seen = update_tracks(state, [_face(1.0, 1.0)], "m2", 0.5)  # model change resets
    assert seen[0][2].arousal == pytest.approx(1.0)
    assert len(seen[0][0].trail) == 1
    state, seen = update_tracks(state, [], "m2", 0.5)
    assert seen == []
    assert len(state.tracks) == 1  # kept for a few frames in case the face reappears


def test_two_faces_keep_their_identity_when_they_move() -> None:
    state = LiveState()
    left, right = _face(0.0, 0.8, x=0), _face(0.6, -0.6, x=400)
    state, seen = update_tracks(state, [left, right], "m", 0.0)
    ids = {round(p.valence, 1): t.id for t, _, p in seen}
    assert sorted(ids.values()) == [1, 2]
    assert all(p.tag and p.accent for _, _, p in seen)  # numbered and coloured
    # both drift right; listed in the opposite order: identities must follow the faces
    state, seen = update_tracks(state, [_face(0.6, -0.6, x=430), _face(0.0, 0.8, x=30)], "m", 0.0)
    assert {round(p.valence, 1): t.id for t, _, p in seen} == ids
    assert all(len(t.trail) == 2 for t, _, _ in seen)


def test_lost_faces_are_dropped_and_numbers_reused() -> None:
    state = LiveState()
    state, _ = update_tracks(state, [_face(0, 0, x=0), _face(0, 0, x=400)], "m", 0.3)
    for _ in range(9):  # face 1 disappears for longer than max_missed
        state, seen = update_tracks(state, [_face(0, 0, x=400)], "m", 0.3)
    assert [t.id for t in state.tracks] == [2]
    state, seen = update_tracks(state, [_face(0, 0, x=400), _face(0, 0, x=0)], "m", 0.3)
    assert sorted(t.id for t, _, _ in seen) == [1, 2]  # the free number 1 is reused


def test_at_most_four_faces_are_tracked() -> None:
    faces = [_face(0, 0, size=50 + 10 * k, x=200 * k) for k in range(6)]
    _, seen = update_tracks(LiveState(), faces, "m", 0.3)
    assert len(seen) == 4
    assert min(f.box[2] - f.box[0] for _, f, _ in seen) == 70  # the largest ones


def test_circumplex_draws_one_trail_per_face() -> None:
    svg = circumplex_svg(
        [Point(0.2, 0.5, 0.5, "a", "1", "#a78bfa"), Point(-0.3, 0.1, 0.3, "b", "2", "#38bdf8")],
        [([(0, 0), (0.2, 0.5)], "#a78bfa"), ([(0, 0), (-0.3, 0.1)], "#38bdf8")],
    )
    ET.fromstring(svg)
    assert 'stroke="#a78bfa"' in svg
    assert 'stroke="#38bdf8"' in svg


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
