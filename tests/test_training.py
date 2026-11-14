from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from facial_expression_analysis import (
    EmotionEstimator,
    LandmarkSubset,
    LinearEmotionModel,
    PairwiseDistanceFeatures,
)
from facial_expression_analysis.data import MorphSet
from facial_expression_analysis.training import (
    Estimator,
    TrainConfig,
    compute_features,
    cross_validate_pls,
    train_model,
    train_pls,
)


@pytest.fixture(scope="module")
def synthetic_morphset(landmarks: np.ndarray) -> MorphSet:
    """Small dataset with targets that are an exact linear function of the features."""
    rng = np.random.default_rng(0)
    n_subjects, per_subject = 40, 8
    idx = rng.integers(0, len(landmarks), size=n_subjects * per_subject)
    lm = landmarks[idx] + rng.normal(scale=1.5, size=(len(idx), 68, 2))
    X = compute_features(lm, PairwiseDistanceFeatures(LandmarkSubset.NO_JAW))
    w = rng.normal(size=(X.shape[1], 3)) / X.shape[1]
    y = (X - X.mean(0)) @ w * 5
    ids = np.repeat(np.arange(n_subjects), per_subject)
    meta = pd.DataFrame({"subject": ids, "person": ids // 2})  # two series per person
    return MorphSet(meta=meta, landmarks=lm, targets=y)


def test_train_pls_end_to_end(synthetic_morphset: MorphSet, tmp_path: Path) -> None:
    result = train_pls(synthetic_morphset, TrainConfig(n_components=10, seed=0))
    test = result.metrics["test"]
    assert test["n_persons"] == 2
    assert test["regressor"]["valence"]["ccc"] > 0.9  # noiseless linear targets
    assert set(test) == {"n_samples", "n_persons", "regressor", "pipeline"}

    path = result.model.save(tmp_path / "model.npz")
    loaded = LinearEmotionModel.load(path)
    assert loaded.metadata["config"]["n_components"] == 10
    assert loaded.metadata["config"]["scale"] == "stable"
    assert "sklearn_version" in loaded.metadata

    est = EmotionEstimator(loaded)
    assert est.predict(synthetic_morphset.landmarks[:3]).shape == (3, 3)


def test_train_kernel_ridge_end_to_end(synthetic_morphset: MorphSet, tmp_path: Path) -> None:
    from facial_expression_analysis.model import KernelEmotionModel, load_model

    config = TrainConfig(n_basis=150, seed=0)
    assert config.estimator is Estimator.KERNEL_RIDGE  # the default
    result = train_model(synthetic_morphset, config)  # export is checked against sklearn
    assert isinstance(result.model, KernelEmotionModel)
    assert result.metrics["test"]["regressor"]["valence"]["ccc"] > 0.8
    loaded = load_model(result.model.save(tmp_path / "k.npz"))
    assert loaded.metadata["config"]["estimator"] == "kernel_ridge"
    est = EmotionEstimator(loaded)
    np.testing.assert_allclose(
        est.predict(synthetic_morphset.landmarks[:3]),
        EmotionEstimator(result.model).predict(synthetic_morphset.landmarks[:3]),
    )


def test_cross_validate_pls_shapes(synthetic_morphset: MorphSet) -> None:
    X = compute_features(synthetic_morphset.landmarks)
    scores = cross_validate_pls(
        X, synthetic_morphset.targets, synthetic_morphset.persons, n_components=5, n_splits=4
    )
    assert scores.shape == (4, 3)
    assert np.all(scores > 0.5)
