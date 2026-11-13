from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.model import LinearEmotionModel


@pytest.fixture
def small_problem(
    rng: np.random.Generator,
) -> tuple[PairwiseDistanceFeatures, np.ndarray, np.ndarray]:
    feat = PairwiseDistanceFeatures(landmark_indices=np.arange(10))  # 45 features
    X = rng.normal(1, 0.3, size=(500, feat.n_features))
    W = rng.normal(size=(feat.n_features, 3))
    y = X @ W + rng.normal(scale=0.1, size=(500, 3))
    return feat, X, y


@pytest.mark.parametrize("kind", ["pls", "ridge_pipeline"])
def test_from_estimator_reproduces_predictions(
    small_problem: tuple[PairwiseDistanceFeatures, np.ndarray, np.ndarray], kind: str
) -> None:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat, X, y = small_problem
    est = PLSRegression(5) if kind == "pls" else make_pipeline(StandardScaler(), Ridge(1.0))
    est.fit(X, y)
    model = LinearEmotionModel.from_estimator(est, feat)
    np.testing.assert_allclose(model.predict_features(X), est.predict(X), atol=1e-8)


def test_from_estimator_rejects_nonlinear(
    small_problem: tuple[PairwiseDistanceFeatures, np.ndarray, np.ndarray],
) -> None:
    from sklearn.tree import DecisionTreeRegressor

    feat, X, y = small_problem
    tree = DecisionTreeRegressor(max_depth=4).fit(X, y)
    with pytest.raises(ValueError, match="not linear"):
        LinearEmotionModel.from_estimator(tree, feat)


def test_save_load_roundtrip(tmp_path: Path, rng: np.random.Generator) -> None:
    feat = PairwiseDistanceFeatures("no_jaw")
    model = LinearEmotionModel(
        rng.normal(size=(feat.n_features, 3)),
        rng.normal(size=3),
        feat.landmark_indices,
        metadata={"note": "test", "metrics": {"ccc": 0.5}},
    )
    loaded = LinearEmotionModel.load(model.save(tmp_path / "m.npz"))
    np.testing.assert_array_equal(loaded.coef, model.coef)
    np.testing.assert_array_equal(loaded.intercept, model.intercept)
    assert loaded.targets == model.targets
    assert loaded.metadata == model.metadata
    assert loaded.subset is not None
    assert loaded.subset.value == "no_jaw"


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="coef has shape"):
        LinearEmotionModel(np.zeros((10, 3)), np.zeros(3), np.arange(17, 68))


def test_bundled_model_loads_and_is_small() -> None:
    model = LinearEmotionModel.load()
    assert model.subset is not None
    assert model.subset.value == "no_jaw"
    assert model.coef.shape == (1275, 3)
    assert "metrics" in model.metadata


def test_predict_accepts_single_and_batch(landmarks: np.ndarray) -> None:
    from facial_expression_analysis import Frontalizer

    model = LinearEmotionModel.load()
    front = Frontalizer()(landmarks[:4])
    batch = model.predict(front)
    assert batch.shape == (4, 3)
    np.testing.assert_allclose(model.predict(front[0]), batch[0])
