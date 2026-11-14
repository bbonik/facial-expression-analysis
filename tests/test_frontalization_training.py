from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from facial_expression_analysis.constants import LEFT_EYE, MIRROR_PERMUTATION, RIGHT_EYE
from facial_expression_analysis.frontalization import Frontalizer
from facial_expression_analysis.frontalization_training import (
    MirrorMode,
    PoseDataset,
    align_parts_to_template,
    build_training_pairs,
    fit_frontalization,
    mean_shape,
    mirror_shapes,
)
from facial_expression_analysis.geometry import procrustes_normalize


def test_mirror_permutation_is_an_involution() -> None:
    perm = np.array(MIRROR_PERMUTATION)
    assert sorted(perm) == list(range(68))
    np.testing.assert_array_equal(perm[perm], np.arange(68))


def test_permuted_mirror_keeps_dlib_left_right_convention(landmarks: np.ndarray) -> None:
    shapes = procrustes_normalize(landmarks[:4])
    for mode, left_is_left in ((MirrorMode.PERMUTE, True), (MirrorMode.NAIVE, False)):
        m = mirror_shapes(shapes, mode)
        left = m[:, LEFT_EYE.start : LEFT_EYE.stop, 0].mean(1)
        right = m[:, RIGHT_EYE.start : RIGHT_EYE.stop, 0].mean(1)
        assert np.all((left < right) == left_is_left)


def test_align_parts_moves_anchors_to_template(landmarks: np.ndarray) -> None:
    shapes = procrustes_normalize(landmarks[:3])
    template = mean_shape(landmarks)
    out = align_parts_to_template(shapes, template)
    for anchor in (slice(50, 53), slice(42, 48), slice(36, 42), slice(27, 36), slice(0, 17)):
        np.testing.assert_allclose(
            out[:, anchor].mean(1), np.broadcast_to(template[anchor].mean(0), (3, 2)), atol=1e-12
        )
    # eyebrow-to-eye geometry (expression information) is preserved
    np.testing.assert_allclose(out[:, 17:22] - out[:, 36:37], shapes[:, 17:22] - shapes[:, 36:37])


def _toy_pose_dataset(landmarks: np.ndarray) -> PoseDataset:
    rows, lms = [], []
    rng = np.random.default_rng(0)
    for subj in range(6):
        base = landmarks[subj]
        for yaw in (0, 15, -15):
            rows.append(
                {"subject": subj, "expression": "NE", "yaw": yaw, "pitch": 0, "family": "x"}
            )
            squash = np.array([1 - abs(yaw) / 100, 1.0])  # crude "pose" effect
            lms.append(base * squash + rng.normal(scale=0.3, size=(68, 2)))
    return PoseDataset(pd.DataFrame(rows), np.array(lms))


@pytest.mark.parametrize(("mode", "factor"), [("none", 1), ("naive", 2), ("permute", 2)])
def test_build_training_pairs(landmarks: np.ndarray, mode: str, factor: int) -> None:
    ds = _toy_pose_dataset(landmarks)
    pairs = build_training_pairs(ds, mean_shape(ds.landmarks), MirrorMode(mode))
    assert len(pairs.inputs) == len(ds) * factor
    assert pairs.mirrored.sum() == (len(ds) if factor == 2 else 0)
    # every target is the (identity-normalised) frontal shape of the same subject
    subj = ds.meta.subject.to_numpy()[pairs.row]
    for s in np.unique(subj):
        t = pairs.targets[(subj == s) & ~pairs.mirrored]
        np.testing.assert_allclose(t, np.broadcast_to(t[0], t.shape))


def test_fit_frontalization_recovers_linear_map(rng: np.random.Generator) -> None:
    from facial_expression_analysis.frontalization_training import TrainingPairs

    A = rng.normal(size=(500, 68, 2))
    W = rng.normal(scale=0.05, size=(137, 136))
    flat = np.concatenate([A[:, :, 0], A[:, :, 1], np.ones((500, 1))], axis=1) @ W
    Y = flat.reshape(500, 2, 68).transpose(0, 2, 1)
    pairs = TrainingPairs(A, Y, np.arange(500), np.zeros(500, bool))
    np.testing.assert_allclose(fit_frontalization(pairs), W, atol=1e-8)
    ridge = fit_frontalization(pairs, ridge=1.0)
    assert np.abs(ridge[:-1]).sum() < np.abs(W[:-1]).sum()


@pytest.mark.slow
def test_reproduces_2021_weights() -> None:
    from facial_expression_analysis.assets import POSE_DATASET, cache_dir
    from facial_expression_analysis.frontalization_training import load_pose_dataset

    if not (cache_dir() / POSE_DATASET.name).is_file():
        pytest.skip("pose dataset not cached")
    ds = load_pose_dataset(drop_invalid=False)
    pairs = build_training_pairs(ds, mean_shape(ds.landmarks), MirrorMode.NAIVE)
    np.testing.assert_allclose(
        fit_frontalization(pairs), Frontalizer.from_2021().weights, atol=1e-10
    )
