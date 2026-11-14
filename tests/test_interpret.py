from __future__ import annotations

import numpy as np
import pytest

from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.interpret import (
    component_region_profile,
    landmark_importance,
    pair_regions,
    pls_vip,
    region_shares,
    standardized_coefficients,
)


def test_pair_regions() -> None:
    pairs = np.array([[17, 37], [48, 54], [30, 60], [36, 45]])
    assert pair_regions(pairs) == ["brows-eyes", "mouth", "mouth-nose", "eyes"]


def test_landmark_importance_conserves_mass(rng: np.random.Generator) -> None:
    pairs = PairwiseDistanceFeatures().pairs
    w = rng.normal(size=len(pairs))
    imp = landmark_importance(w, pairs)
    assert imp.sum() == pytest.approx(np.abs(w).sum())
    assert np.all(imp[:17] == 0)  # no jaw pairs in the default features


def test_region_shares_sum_to_one(rng: np.random.Generator) -> None:
    pairs = PairwiseDistanceFeatures().pairs
    shares = region_shares(rng.normal(size=len(pairs)), pairs)
    assert shares.sum() == pytest.approx(1.0)
    assert shares.is_monotonic_decreasing


def test_component_concentration_detects_single_region() -> None:
    pairs = PairwiseDistanceFeatures().pairs
    mouth_only = np.array([[1.0 if a >= 48 and b >= 48 else 0.0] for a, b in pairs])
    uniform = np.ones((len(pairs), 1))
    prof = component_region_profile(np.hstack([mouth_only, uniform]), pairs)
    assert prof.loc[1, "mouth"] == pytest.approx(1.0)
    assert prof.loc[1, "concentration"] == pytest.approx(1.0)
    assert prof.loc[2, "concentration"] < 0.3


def test_vip_and_standardized_coefficients(rng: np.random.Generator) -> None:
    from sklearn.cross_decomposition import PLSRegression

    X = rng.normal(size=(300, 10))
    y = X[:, :2] @ np.array([[3.0], [-2.0]]) + rng.normal(scale=0.1, size=(300, 1))
    pls = PLSRegression(3).fit(X, y)
    vip = pls_vip(pls)
    assert np.mean(vip**2) == pytest.approx(1.0)  # VIP is normalised so that mean(VIP^2) = 1
    assert set(np.argsort(-vip)[:2]) == {0, 1}
    sc = standardized_coefficients(np.ones((10, 1)), X * np.arange(1, 11))
    assert sc[9, 0] == pytest.approx(10 * X[:, 9].std(), rel=1e-6)


def test_equivalent_movement_recovers_mouth_opening(landmarks: np.ndarray) -> None:
    from facial_expression_analysis.frontalization import Frontalizer
    from facial_expression_analysis.interpret import (
        distance_jacobian,
        equivalent_movement,
        movement_region_shares,
    )

    shape = Frontalizer()(landmarks[0])
    pairs = PairwiseDistanceFeatures().pairs
    move = np.zeros((68, 2))
    move[[56, 57, 58, 65, 66, 67], 1] = 0.05  # lower lip drops
    flat = np.concatenate([move[:, 0], move[:, 1]])
    w = distance_jacobian(shape, pairs) @ flat
    rec = equivalent_movement(w, shape, pairs)
    # the recovered movement produces the same distance changes ...
    rec_flat = np.concatenate([rec[:, 0], rec[:, 1]])
    np.testing.assert_allclose(distance_jacobian(shape, pairs) @ rec_flat, w, atol=1e-8)
    # ... and is concentrated on the mouth
    assert movement_region_shares(rec)["mouth"] > 0.8
