from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from facial_expression_analysis.constants import STABLE_POINTS
from facial_expression_analysis.features import LandmarkSubset, PairwiseDistanceFeatures

from .conftest import random_similarity


@pytest.mark.parametrize(("subset", "n"), [("all", 2278), ("no_jaw", 1275), ("intraface49", 1176)])
def test_feature_dimensionality(subset: str, n: int) -> None:
    assert PairwiseDistanceFeatures(subset).n_features == n


def test_no_jaw_excludes_jaw_and_keeps_mouth() -> None:
    # Regression: the 2021 template used positions 0..50 instead of landmark ids
    # 17..67, i.e. it kept the jaw and dropped most of the mouth.
    used = np.unique(PairwiseDistanceFeatures("no_jaw").pairs)
    np.testing.assert_array_equal(used, np.arange(17, 68))


def test_intraface49_drops_jaw_and_inner_mouth_corners() -> None:
    used = set(np.unique(PairwiseDistanceFeatures("intraface49").pairs))
    assert used == set(range(17, 68)) - {60, 64}


@pytest.mark.parametrize(
    ("scale", "ref_idx"),
    [("subset", LandmarkSubset.NO_JAW.indices), ("stable", np.array(STABLE_POINTS))],
)
def test_matches_naive_implementation(
    landmarks: np.ndarray, scale: str, ref_idx: np.ndarray
) -> None:
    shape = landmarks[3]
    idx = LandmarkSubset.NO_JAW.indices
    ref = shape[ref_idx]
    s = np.sqrt(np.mean(np.sum((ref - ref.mean(0)) ** 2, axis=1)))
    naive = [np.linalg.norm(shape[a] - shape[b]) / s for a, b in combinations(idx, 2)]
    feats = PairwiseDistanceFeatures("no_jaw", scale=scale)
    np.testing.assert_allclose(feats(shape), naive, rtol=1e-6)
    assert feats.scale_reference == scale


def test_stable_scale_ignores_mouth_opening(landmarks: np.ndarray) -> None:
    """Opening the mouth must not change distances among the eyes and brows."""
    shape = landmarks[0].copy()
    opened = shape.copy()
    opened[[56, 57, 58, 65, 66, 67], 1] += 15.0  # drop the lower lip
    eye_brow = [k for k, (a, b) in enumerate(PairwiseDistanceFeatures().pairs) if a < 48 and b < 48]
    stable = PairwiseDistanceFeatures(scale="stable")
    subset = PairwiseDistanceFeatures(scale="subset")
    np.testing.assert_allclose(stable(opened)[eye_brow], stable(shape)[eye_brow], rtol=1e-6)
    assert not np.allclose(subset(opened)[eye_brow], subset(shape)[eye_brow], rtol=1e-3)


def test_custom_scale_indices() -> None:
    feats = PairwiseDistanceFeatures(scale=[36, 39, 42, 45])
    assert feats.scale_reference is None
    np.testing.assert_array_equal(feats.scale_indices, [36, 39, 42, 45])


def test_similarity_invariance(landmarks: np.ndarray, rng: np.random.Generator) -> None:
    feat = PairwiseDistanceFeatures("all")
    shape = landmarks[0]
    np.testing.assert_allclose(
        feat(random_similarity(shape, rng, max_angle=np.pi)), feat(shape), rtol=1e-5
    )


def test_chunking_is_transparent(landmarks: np.ndarray) -> None:
    feat = PairwiseDistanceFeatures()
    np.testing.assert_array_equal(feat(landmarks, chunk_size=5), feat(landmarks))


def test_feature_names_align_with_pairs() -> None:
    feat = PairwiseDistanceFeatures("no_jaw")
    names = feat.feature_names()
    assert len(names) == feat.n_features
    assert names[0] == "d_17_18"


@pytest.mark.parametrize("bad", [[0, 0, 1], [-1, 2], [3, 68]])
def test_rejects_invalid_indices(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="landmark_indices"):
        PairwiseDistanceFeatures(landmark_indices=bad)
