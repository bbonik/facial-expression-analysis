"""Training and evaluation of emotion models on MorphSet."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .data import MorphSet, split_by_group
from .features import LandmarkSubset, PairwiseDistanceFeatures, ScaleReference
from .frontalization import Frontalizer
from .metrics import ccc, regression_report
from .model import EmotionModel, KernelEmotionModel, LinearEmotionModel
from .pipeline import EmotionEstimator

logger = logging.getLogger(__name__)


class Estimator(StrEnum):
    """Regressor families that can be trained and exported."""

    KERNEL_RIDGE = "kernel_ridge"  # RBF kernel ridge on a Nystroem basis (default)
    PLS = "pls"  # partial least squares (linear, interpretable)


@dataclass
class TrainConfig:
    """Configuration of a training run.

    ``n_components`` applies to PLS; ``gamma`` (RBF width, in units of
    ``1 / n_features`` on standardised features), ``alpha`` (ridge penalty) and
    ``n_basis`` (Nystroem basis size) to kernel ridge. The kernel defaults were
    selected by nested CV in ``experiments/08_model_comparison.py``.
    """

    estimator: Estimator = Estimator.KERNEL_RIDGE
    n_components: int = 31
    gamma: float = 0.5
    alpha: float = 1.0
    n_basis: int = 2000
    subset: LandmarkSubset = LandmarkSubset.NO_JAW
    scale: ScaleReference = ScaleReference.STABLE
    val_size: float = 0.2
    test_size: float = 0.1
    seed: int = 1

    def features(self) -> PairwiseDistanceFeatures:
        return PairwiseDistanceFeatures(self.subset, scale=self.scale)

    def describe(self) -> str:
        if self.estimator is Estimator.PLS:
            return f"PLSRegression(n_components={self.n_components})"
        return (
            f"StandardScaler + Nystroem(rbf, gamma={self.gamma}/n_features, "
            f"n_components={self.n_basis}) + Ridge(alpha={self.alpha})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "estimator": self.estimator.value,
            "subset": self.subset.value,
            "scale": self.scale.value,
        }


@dataclass
class TrainResult:
    model: EmotionModel
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
    model: EmotionModel, data: MorphSet, features: NDArray[np.float32] | None = None
) -> dict[str, Any]:
    """Evaluate a model on a dataset.

    Two views are reported:

    * ``regressor`` - raw linear outputs (comparable to the 2021 results);
    * ``pipeline`` - what users get: outputs clipped to their valid ranges.
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


def _fit_and_export(
    config: TrainConfig,
    feat: PairwiseDistanceFeatures,
    frontal: NDArray[np.float64],
    X: NDArray[np.float32],
    y: NDArray[np.float64],
    metadata: dict[str, Any],
) -> EmotionModel:
    from sklearn.cross_decomposition import PLSRegression

    if config.estimator is Estimator.PLS:
        pls = PLSRegression(n_components=config.n_components).fit(X, y)
        return LinearEmotionModel.from_estimator(pls, feat, metadata=metadata)

    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # float64: the Nystroem normalisation K^-1/2 amplifies float32 rounding errors
    X64 = np.asarray(X, dtype=np.float64)
    pipe = make_pipeline(
        StandardScaler(),
        Nystroem(
            gamma=config.gamma / feat.n_features,
            n_components=min(config.n_basis, len(X)),
            random_state=config.seed,
        ),
        Ridge(alpha=config.alpha),
    ).fit(X64, y)
    return KernelEmotionModel.from_estimator(
        pipe, feat, frontal, metadata=metadata, check_features=X64[:256]
    )


def train_model(data: MorphSet, config: TrainConfig | None = None) -> TrainResult:
    """Train an emotion model with a person-wise train/validation/test split."""
    from sklearn import __version__ as sklearn_version

    config = config or TrainConfig()
    feat = config.features()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    frontal = Frontalizer()(data.landmarks)
    features = feat.transform(frontal)
    timings["features_s"] = time.perf_counter() - t0
    logger.info("Computed %s features in %.1fs", features.shape, timings["features_s"])

    train, val, test = split_by_group(
        data.persons, val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )

    metadata = {
        "estimator": config.describe(),
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
    t0 = time.perf_counter()
    model = _fit_and_export(
        config, feat, frontal[train], features[train], data.targets[train], metadata
    )
    timings["fit_s"] = time.perf_counter() - t0
    logger.info("Fitted %s in %.1fs", config.describe(), timings["fit_s"])

    metrics = {
        "validation": evaluate_model(model, data.subset(val), features[val]),
        "test": evaluate_model(model, data.subset(test), features[test]),
    }
    model.metadata["metrics"] = metrics
    if isinstance(model, KernelEmotionModel):
        # reference distribution of the reliability signal (distance to the training faces)
        dist = model.basis_distance(features[val])
        model.metadata["basis_distance_quantiles"] = {
            str(q): float(np.quantile(dist, q)) for q in (0.5, 0.95, 0.99)
        }
    return TrainResult(model=model, metrics=metrics, config=config, timings=timings)


def train_pls(data: MorphSet, config: TrainConfig | None = None) -> TrainResult:
    """Train a PLS model (``train_model`` with ``estimator=PLS``)."""
    return train_model(data, replace(config or TrainConfig(), estimator=Estimator.PLS))


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
