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
        feat.scale_indices,
        metadata={"note": "test", "metrics": {"ccc": 0.5}},
    )
    loaded = LinearEmotionModel.load(model.save(tmp_path / "m.npz"))
    np.testing.assert_array_equal(loaded.coef, model.coef)
    np.testing.assert_array_equal(loaded.intercept, model.intercept)
    assert loaded.targets == model.targets
    assert loaded.metadata == model.metadata
    assert loaded.subset is not None
    assert loaded.subset.value == "no_jaw"
    np.testing.assert_array_equal(loaded.scale_indices, feat.scale_indices)


def test_loads_format_v1_without_scale_indices(tmp_path: Path, rng: np.random.Generator) -> None:
    idx = np.arange(17, 68)
    path = tmp_path / "v1.npz"
    np.savez(
        path,
        coef=rng.normal(size=(1275, 3)),
        intercept=np.zeros(3),
        landmark_indices=idx,
        targets=np.array(["arousal", "valence", "intensity"]),
        metadata=np.array("{}"),
        format_version=np.array(1),
    )
    model = LinearEmotionModel.load(path)
    np.testing.assert_array_equal(model.scale_indices, idx)  # v1 used the subset scale


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="coef has shape"):
        LinearEmotionModel(np.zeros((10, 3)), np.zeros(3), np.arange(17, 68), np.arange(17, 68))


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


# ---------------------------------------------------------------- kernel model
@pytest.fixture(scope="module")
def kernel_problem(landmarks: np.ndarray) -> tuple:
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from facial_expression_analysis import Frontalizer

    rng = np.random.default_rng(0)
    lm = landmarks[rng.integers(0, len(landmarks), 300)] + rng.normal(scale=1.0, size=(300, 68, 2))
    frontal = Frontalizer()(lm)
    feat = PairwiseDistanceFeatures("no_jaw")
    X = feat.transform(frontal).astype(np.float64)
    y = np.tanh(X[:, :3] - X[:, :3].mean(0))
    pipe = make_pipeline(
        StandardScaler(), Nystroem(gamma=0.5 / X.shape[1], n_components=100, random_state=0),
        Ridge(alpha=1.0),
    ).fit(X, y)  # fmt: skip
    return feat, frontal, X, pipe


def test_kernel_model_reproduces_pipeline(kernel_problem: tuple) -> None:
    from facial_expression_analysis.model import KernelEmotionModel

    feat, frontal, X, pipe = kernel_problem
    model = KernelEmotionModel.from_estimator(pipe, feat, frontal, check_features=X)
    np.testing.assert_allclose(model.predict_features(X), pipe.predict(X), atol=1e-8)
    np.testing.assert_allclose(model.predict(frontal[:5]), pipe.predict(X[:5]), atol=1e-8)
    assert model.predict(frontal[0]).shape == (3,)


def test_kernel_model_roundtrip_and_dispatch(kernel_problem: tuple, tmp_path: Path) -> None:
    from facial_expression_analysis.model import KernelEmotionModel, load_model

    feat, frontal, X, pipe = kernel_problem
    model = KernelEmotionModel.from_estimator(pipe, feat, frontal, metadata={"note": "k"})
    path = model.save(tmp_path / "k.npz")
    loaded = load_model(path)
    assert isinstance(loaded, KernelEmotionModel)
    assert loaded.metadata == {"note": "k"}
    np.testing.assert_allclose(loaded.predict_features(X[:20]), model.predict_features(X[:20]))
    with pytest.raises(ValueError, match="kernel model"):
        LinearEmotionModel.load(path)


def test_load_model_dispatches_linear(tmp_path: Path, rng: np.random.Generator) -> None:
    from facial_expression_analysis.model import KernelEmotionModel, load_model

    feat = PairwiseDistanceFeatures("no_jaw")
    lin = LinearEmotionModel(
        rng.normal(size=(feat.n_features, 3)), np.zeros(3), feat.landmark_indices,
        feat.scale_indices,
    )  # fmt: skip
    path = lin.save(tmp_path / "l.npz")
    assert isinstance(load_model(path), LinearEmotionModel)
    with pytest.raises(ValueError, match="linear model"):
        KernelEmotionModel.load(path)


def test_bundled_models(landmarks: np.ndarray) -> None:
    from facial_expression_analysis import Frontalizer
    from facial_expression_analysis.model import KernelEmotionModel, load_model

    default = load_model()
    assert isinstance(default, LinearEmotionModel)  # PLS(31) is the default
    assert default.metadata["config"]["n_components"] == 31
    kernel = KernelEmotionModel.load()
    assert kernel.basis_landmarks.shape[1:] == (68, 2)
    assert "basis_distance_quantiles" in kernel.metadata
    front = Frontalizer()(landmarks[:4])
    for model in (default, kernel):
        assert "metrics" in model.metadata
        pred = model.predict(front)
        assert pred.shape == (4, 3)
        assert np.all(np.isfinite(pred))


def test_kernel_basis_distance_flags_outliers(kernel_problem: tuple) -> None:
    from facial_expression_analysis.model import KernelEmotionModel

    feat, frontal, X, pipe = kernel_problem
    model = KernelEmotionModel.from_estimator(pipe, feat, frontal)
    basis = feat.transform(model.basis_landmarks)
    np.testing.assert_allclose(model.basis_distance(basis[:5]), 0.0, atol=1e-5)
    far = X[:3] * 1.5
    assert np.all(model.basis_distance(far) > 5 * np.median(model.basis_distance(X)))
    # far from all basis faces the prediction tends to the intercept
    np.testing.assert_allclose(model.predict_features(X[:3] * 10), np.tile(model.intercept, (3, 1)))
