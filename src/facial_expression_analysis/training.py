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
from .data import MorphSet, split_by_group
from .features import LandmarkSubset, PairwiseDistanceFeatures, ScaleReference
from .frontalization import Frontalizer
from .metrics import ccc, regression_report
from .model import LinearEmotionModel
from .pipeline import EmotionEstimator

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration of a PLS training run."""

    n_components: int = 31
    subset: LandmarkSubset = LandmarkSubset.NO_JAW
    scale: ScaleReference = ScaleReference.STABLE
    val_size: float = 0.2
    test_size: float = 0.1
    seed: int = 1

    def features(self) -> PairwiseDistanceFeatures:
        return PairwiseDistanceFeatures(self.subset, scale=self.scale)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "subset": self.subset.value, "scale": self.scale.value}


@dataclass
class TrainResult:
    model: LinearEmotionModel
    metrics: dict[str, Any]
    config: TrainConfig
    timings: dict[str, float] = field(default_factory=dict)


def compute_features(
    landmarks: NDArray[np.float64],
    features: PairwiseDistanceFeatures | None = None,
    frontalizer: Frontalizer | None = None,
) -> NDArray[np.float32]:
    """Frontalize a batch of raw landmarks and compute pairwise-distance features."""
    frontalizer = frontalizer or Frontalizer()
    features = features or PairwiseDistanceFeatures()
    return features.transform(frontalizer(landmarks))


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
        features = compute_features(data.landmarks, model.features)
    raw = model.predict_features(features)
    final = EmotionEstimator(model)._postprocess(raw)
    return {
        "n_samples": len(data),
        "n_persons": int(np.unique(data.persons).size),
        "regressor": regression_report(data.targets, raw, model.targets),
        "pipeline": regression_report(data.targets, final, model.targets),
    }


def train_pls(data: MorphSet, config: TrainConfig | None = None) -> TrainResult:
    """Train a PLS regressor with a person-wise train/validation/test split."""
    from sklearn import __version__ as sklearn_version
    from sklearn.cross_decomposition import PLSRegression

    config = config or TrainConfig()
    feat = config.features()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    features = compute_features(data.landmarks, feat)
    timings["features_s"] = time.perf_counter() - t0
    logger.info("Computed %s features in %.1fs", features.shape, timings["features_s"])

    train, val, test = split_by_group(
        data.persons, val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )

    t0 = time.perf_counter()
    pls = PLSRegression(n_components=config.n_components)
    pls.fit(features[train], data.targets[train])
    timings["fit_s"] = time.perf_counter() - t0
    logger.info("Fitted PLS(%d) in %.1fs", config.n_components, timings["fit_s"])

    metadata = {
        "estimator": f"PLSRegression(n_components={config.n_components})",
        "landmark_subset": config.subset.value,
        "scale_reference": config.scale.value,
        "config": config.to_dict(),
        "split": "by person (see facial_expression_analysis.data)",
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


def cross_validate_pls(
    features: NDArray[np.float32],
    targets: NDArray[np.float64],
    groups: NDArray[np.integer],
    n_components: int = 31,
    n_splits: int = 5,
) -> NDArray[np.float64]:
    """Run grouped k-fold CV of PLS; return per-fold CCC of shape ``(n_splits, n_targets)``.

    Folds are formed with :class:`sklearn.model_selection.GroupKFold`, so a
    group (person) never appears in both the training and the test fold.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import GroupKFold

    scores = []
    for train, test in GroupKFold(n_splits=n_splits).split(features, targets, groups):
        pred = (
            PLSRegression(n_components).fit(features[train], targets[train]).predict(features[test])
        )
        scores.append([ccc(targets[test, k], pred[:, k]) for k in range(targets.shape[1])])
    return np.asarray(scores)
