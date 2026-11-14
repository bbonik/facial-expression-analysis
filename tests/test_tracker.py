"""Tests of the landmark tracker, the box regressor and the robust landmark fit."""

from __future__ import annotations

import numpy as np
import pytest

from facial_expression_analysis.landmarks import BoxRegressor
from facial_expression_analysis.landmarks.dlib_detector import stretch_contrast
from facial_expression_analysis.landmarks.tracker import ASSET

from .conftest import IMAGES, random_similarity, requires_dlib_model


def _true_box(shape: np.ndarray) -> np.ndarray:
    """A box that is an exact similarity-invariant function of the shape."""
    radius = np.sqrt(((shape - shape.mean(0)) ** 2).sum(1).mean())
    c = shape.mean(0) + np.array([0.0, -0.1]) * radius
    half = 1.4 * radius
    return np.r_[c - half, c + half]


def test_box_regressor_recovers_a_linear_box_rule(landmarks, rng) -> None:
    # translation and scale only: the box rule is then exactly linear in the features
    shapes = np.array([landmarks[i % len(landmarks)] * rng.uniform(0.5, 2) + rng.uniform(-50, 50, 2)
                       for i in range(200)])  # fmt: skip
    boxes = np.array([_true_box(s) for s in shapes])
    reg = BoxRegressor.fit(shapes, boxes, alpha=1e-6)
    np.testing.assert_allclose(reg.predict(shapes), boxes, atol=1e-3)
    assert reg.predict(shapes[0]).shape == (4,)


def test_box_regressor_save_load_roundtrip(landmarks, tmp_path) -> None:
    boxes = np.array([_true_box(s) for s in landmarks])
    reg = BoxRegressor.fit(landmarks, boxes, metadata={"note": "test"})
    reg.save(tmp_path / "r.npz")
    back = BoxRegressor.load(tmp_path / "r.npz")
    np.testing.assert_array_equal(back.predict(landmarks), reg.predict(landmarks))
    assert back.metadata == {"note": "test"}


def test_shipped_box_regressor_is_plausible_and_pose_equivariant(landmarks, rng) -> None:
    assert ASSET.is_file()
    reg = BoxRegressor.load()
    box = reg.predict(landmarks[0])
    # the box covers the inner face and is roughly square
    inner = landmarks[0][17:]
    assert box[0] < inner[:, 0].min()
    assert box[2] > inner[:, 0].max()
    assert box[1] < inner[:, 1].min()
    assert box[3] > inner[:, 1].mean()
    w, h = box[2] - box[0], box[3] - box[1]
    assert 0.8 < w / h < 1.25
    # translating and scaling the face moves and scales the box with it
    moved = reg.predict(landmarks[0] * 2 + 10)
    np.testing.assert_allclose(moved, box * 2 + 10, rtol=1e-6, atol=1e-6)
    shape = random_similarity(landmarks[0], rng, max_angle=0.2)
    assert np.isfinite(reg.predict(shape)).all()


def test_stretch_contrast_is_linear_and_skips_normal_images() -> None:
    normal = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
    assert stretch_contrast(normal) is normal
    dim = (normal // 4).astype(np.uint8)  # 0..63
    out = stretch_contrast(dim)
    assert out.max() >= 250
    assert out.min() <= 5
    assert np.all(np.diff(out[0].astype(int)) >= 0)  # monotone (linear) mapping


# ------------------------------------------------------------------- with the dlib model
@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    import cv2

    from facial_expression_analysis import load_image

    img = load_image(IMAGES / "pleased.jpg")[..., :3]
    h, w = img.shape[:2]
    return np.ascontiguousarray(cv2.resize(img, (640, round(h * 640 / w))))


@requires_dlib_model
def test_robust_fit_agrees_with_plain_fit_and_reports_spread(frame) -> None:
    from facial_expression_analysis.landmarks import DlibLandmarkDetector
    from facial_expression_analysis.landmarks.dlib_detector import to_gray

    det = DlibLandmarkDetector(upsample=0)
    gray = to_gray(frame)
    (box,) = det.detect_boxes(gray)
    plain = det.fit(gray, box)
    robust = det.fit(gray, box, robust=True)
    iod = np.linalg.norm(plain.landmarks[36] - plain.landmarks[45])
    assert plain.spread == 0.0
    assert np.linalg.norm(robust.landmarks - plain.landmarks, axis=1).mean() < 0.04 * iod
    assert robust.spread < 0.03
    # a box beside the face is not a face: the shifted fits disagree
    w = box.width
    off = [box.left + 1.5 * w, box.top, box.right + 1.5 * w, box.bottom]
    if off[2] < gray.shape[1]:
        assert det.fit(gray, off, robust=True).spread > robust.spread


@requires_dlib_model
def test_tracker_follows_a_moving_face_and_drops_it_when_it_leaves(frame) -> None:
    from facial_expression_analysis.landmarks import DlibLandmarkDetector, LandmarkTracker

    plain = DlibLandmarkDetector(upsample=0)
    tracker = LandmarkTracker(DlibLandmarkDetector(upsample=0, robust=True))
    (first,) = tracker.detect(frame)
    ref = plain.detect(frame)[0].landmarks
    iod = np.linalg.norm(ref[36] - ref[45])
    assert np.linalg.norm(first.landmarks - ref, axis=1).mean() < 0.05 * iod

    # move the face by a few pixels per frame: the tracker follows it
    shift = 0
    for _ in range(8):
        shift += 3
        moved = np.roll(frame, shift, axis=1)
        (face,) = tracker.detect(moved)
        expected = ref + np.array([shift, 0])
        assert np.linalg.norm(face.landmarks - expected, axis=1).mean() < 0.05 * iod
    assert len(tracker.tracks) == 1

    # the face disappears: the track is dropped
    blank = np.full_like(frame, 90)
    results = [tracker.detect(blank) for _ in range(3)]
    assert results[-1] == []
    assert tracker.tracks == []

    tracker.reset()
    assert tracker.frame == 0
    assert tracker.tracks == []
