from __future__ import annotations

import numpy as np
import pytest

from facial_expression_analysis.geometry import (
    as_landmark_batch,
    eye_angle,
    procrustes_normalize,
    shape_scale,
)

from .conftest import random_similarity


def test_normalised_shape_has_zero_mean_unit_scale_level_eyes(landmarks: np.ndarray) -> None:
    out = procrustes_normalize(landmarks)
    np.testing.assert_allclose(out.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(shape_scale(out), 1, atol=1e-12)
    np.testing.assert_allclose(eye_angle(out), 0, atol=1e-12)


def test_invariant_to_similarity_transforms(
    landmarks: np.ndarray, rng: np.random.Generator
) -> None:
    shape = landmarks[0]
    transformed = random_similarity(shape, rng)
    np.testing.assert_allclose(
        procrustes_normalize(transformed), procrustes_normalize(shape), atol=1e-10
    )


def test_batch_matches_single(landmarks: np.ndarray) -> None:
    batch = procrustes_normalize(landmarks[:5])
    for k in range(5):
        np.testing.assert_array_equal(batch[k], procrustes_normalize(landmarks[k]))


def test_vertical_eye_line_does_not_crash(landmarks: np.ndarray) -> None:
    # Regression: the 2021 code raised NameError when the eyes were vertically aligned.
    rot90 = landmarks[0] @ np.array([[0.0, -1.0], [1.0, 0.0]])
    out = procrustes_normalize(rot90)
    np.testing.assert_allclose(out, procrustes_normalize(landmarks[0]), atol=1e-10)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="68, 2"):
        as_landmark_batch(np.zeros((10, 2)))


def test_flags_can_disable_steps(landmarks: np.ndarray) -> None:
    np.testing.assert_array_equal(
        procrustes_normalize(landmarks[0], translate=False, scale=False, rotate=False),
        landmarks[0],
    )
