"""Training and evaluation of linear emotion models on MorphSet."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .data import MorphSet, split_subjects
from .features import LandmarkSubset, PairwiseDistanceFeatures
from .frontalization import Frontalizer
from .metrics import regression_report
from .model import LinearEmotionModel
from .pipeline import EmotionEstimator

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration of a PLS training run."""

    n_components: int = 31
    subset: LandmarkSubset = LandmarkSubset.NO_JAW
    val_size: float = 0.2
    test_size: float = 0.1
    seed: int = 1


@dataclass
class TrainResult:
    model: LinearEmotionModel
    metrics: dict[str, Any]
    config: TrainConfig
    timings: dict[str, float] = field(default_factory=dict)


def compute_features(
    landmarks: NDArray[np.float64],
    subset: LandmarkSubset | str = LandmarkSubset.NO_JAW,
    frontalizer: Frontalizer | None = None,
) -> NDArray[np.float32]:
    """Frontalize a batch of raw landmarks and compute pairwise-distance features."""
    frontalizer = frontalizer or Frontalizer()
    return PairwiseDistanceFeatures(subset).transform(frontalizer(landmarks))


def evaluate_model(
    model: LinearEmotionModel, data: MorphSet, features: NDArray[np.float32] | None = None
) -> dict[str, Any]:
    """Evaluate a model on a dataset.

    Two views are reported:

    * ``regressor`` - raw linear outputs (comparable to the 2021 results);
    * ``pipeline`` - what users get: arousal/valence clipped to [-1, 1] and
      intensity derived from them.
    """
    if features is None:
        features = compute_features(data.landmarks, landmark_subset(model))
    raw = model.predict_features(features)
    final = EmotionEstimator(model)._postprocess(raw)
    return {
        "n_samples": len(data),
        "n_subjects": int(np.unique(data.subjects).size),
        "regressor": regression_report(data.targets, raw, model.targets),
        "pipeline": regression_report(data.targets, final, model.targets),
    }


def landmark_subset(model: LinearEmotionModel) -> LandmarkSubset:
    if model.subset is None:
        raise ValueError("Model uses a custom landmark set")
    return model.subset


def train_pls(data: MorphSet, config: TrainConfig | None = None) -> TrainResult:
    """Train a PLS regressor with a subject-wise train/validation/test split."""
    from sklearn import __version__ as sklearn_version
    from sklearn.cross_decomposition import PLSRegression

    config = config or TrainConfig()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    features = compute_features(data.landmarks, config.subset)
    timings["features_s"] = time.perf_counter() - t0
    logger.info("Computed %s features in %.1fs", features.shape, timings["features_s"])

    train, val, test = split_subjects(
        data.subjects, val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )

    t0 = time.perf_counter()
    pls = PLSRegression(n_components=config.n_components)
    pls.fit(features[train], data.targets[train])
    timings["fit_s"] = time.perf_counter() - t0
    logger.info("Fitted PLS(%d) in %.1fs", config.n_components, timings["fit_s"])

    feat = PairwiseDistanceFeatures(config.subset)
    metadata = {
        "estimator": f"PLSRegression(n_components={config.n_components})",
        "landmark_subset": config.subset.value,
        "config": {**asdict(config), "subset": config.subset.value},
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "sklearn_version": sklearn_version,
        "python_version": platform.python_version(),
        "n_train": int(train.sum()),
    }
    model = LinearEmotionModel.from_estimator(pls, feat, metadata=metadata)

    metrics = {
        "validation": evaluate_model(model, data.subset(val), features[val]),
        "test": evaluate_model(model, data.subset(test), features[test]),
    }
    model.metadata["metrics"] = metrics
    return TrainResult(model=model, metrics=metrics, config=config, timings=timings)
