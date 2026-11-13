from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from facial_expression_analysis.features import LandmarkSubset, PairwiseDistanceFeatures

from .conftest import random_similarity


@pytest.mark.parametrize(("subset", "n"), [("all", 2278), ("no_jaw", 1275)])
def test_feature_dimensionality(subset: str, n: int) -> None:
    assert PairwiseDistanceFeatures(subset).n_features == n


def test_no_jaw_excludes_jaw_and_keeps_mouth() -> None:
    # Regression: the 2021 template used positions 0..50 instead of landmark ids
    # 17..67, i.e. it kept the jaw and dropped most of the mouth.
    used = np.unique(PairwiseDistanceFeatures("no_jaw").pairs)
    np.testing.assert_array_equal(used, np.arange(17, 68))


def test_matches_naive_implementation(landmarks: np.ndarray) -> None:
    shape = landmarks[3]
    idx = LandmarkSubset.NO_JAW.indices
    sub = shape[idx]
    scale = np.sqrt(np.mean(np.sum((sub - sub.mean(0)) ** 2, axis=1)))
    naive = [np.linalg.norm(shape[a] - shape[b]) / scale for a, b in combinations(idx, 2)]
    np.testing.assert_allclose(PairwiseDistanceFeatures("no_jaw")(shape), naive, rtol=1e-6)


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
