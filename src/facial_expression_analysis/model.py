"""Portable emotion regressors.

Two model families are shipped, both stored as small ``.npz`` files with JSON
metadata so that inference needs neither scikit-learn nor pickle:

* :class:`LinearEmotionModel` - any linear model (PLS, ridge, ...):
  ``y = X @ coef + intercept``. The default model is PLS(31): best calibrated on
  neutral faces, a facial movement moves the same axis for every face, and most
  robust to head pose (``experiments/12_behavioural_tests.py``). Interpretable
  (see :mod:`.interpret`).
* :class:`KernelEmotionModel` - RBF kernel ridge regression on a Nystroem basis:
  ``y = k(z, basis) @ dual_coef + intercept`` on standardised features. Highest
  offline CCC on MorphSet (``experiments/08_model_comparison.py``) and better on
  extreme expressions such as screams, but less well-behaved live.

:func:`load_model` reads either kind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .features import LandmarkSubset, PairwiseDistanceFeatures

TARGETS = ("arousal", "valence", "intensity")
DEFAULT_MODEL = "emotion_model.npz"  # PLS(31): calibrated, decomposed, pose-robust
KERNEL_MODEL = "emotion_model_kernel.npz"  # kernel ridge: highest offline CCC
FORMAT_VERSION = 3  # v2 added scale_indices, v3 the model kind


class EmotionModel(Protocol):
    """What :class:`~.pipeline.EmotionEstimator` needs from a regressor."""

    targets: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def features(self) -> PairwiseDistanceFeatures: ...

    def predict_features(self, X: ArrayLike) -> NDArray[np.float64]: ...

    def predict(self, frontal_landmarks: ArrayLike) -> NDArray[np.float64]: ...

    def save(self, path: str | Path) -> Path: ...


def _bundled(name: str) -> Any:
    return resources.files(__package__).joinpath("assets", name)


def _predict_landmarks(model: EmotionModel, frontal_landmarks: ArrayLike) -> NDArray[np.float64]:
    feats = model.features.transform(frontal_landmarks)
    return model.predict_features(np.atleast_2d(feats)).reshape(
        (*np.shape(feats)[:-1], len(model.targets))
    )


def _check_version(data: Any) -> int:
    version = int(data["format_version"])
    if version > FORMAT_VERSION:
        raise ValueError(f"Unsupported model format version {version}")
    return version


def _kind(data: Any) -> str:
    return str(data["kind"]) if "kind" in data.files else "linear"


def load_model(path: str | Path | None = None) -> LinearEmotionModel | KernelEmotionModel:
    """Load a linear or kernel model; the bundled default model if ``path`` is ``None``."""
    if path is None:
        with resources.as_file(_bundled(DEFAULT_MODEL)) as p:
            return load_model(p)
    with np.load(path, allow_pickle=False) as data:
        kind = _kind(data)
    if kind == "linear":
        return LinearEmotionModel.load(path)
    if kind == "kernel":
        return KernelEmotionModel.load(path)
    raise ValueError(f"Unknown model kind {kind!r}")


class _Predictor(Protocol):
    def predict(self, X: Any) -> Any: ...


@dataclass
class LinearEmotionModel:
    """Linear map from geometric features to ``(arousal, valence, intensity)``.

    Attributes
    ----------
    coef
        ``(n_features, n_targets)`` weights.
    intercept
        ``(n_targets,)`` biases.
    landmark_indices
        Landmarks used to build the pairwise-distance features.
    scale_indices
        Landmarks used for the scale normalisation of the features.
    targets
        Names of the output columns.
    metadata
        Free-form JSON-serialisable information (training config, metrics, ...).
    """

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    landmark_indices: NDArray[np.intp]
    scale_indices: NDArray[np.intp]
    targets: tuple[str, ...] = TARGETS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coef = np.asarray(self.coef, dtype=np.float64)
        self.intercept = np.asarray(self.intercept, dtype=np.float64).reshape(-1)
        self.landmark_indices = np.asarray(self.landmark_indices, dtype=np.intp)
        self.scale_indices = np.asarray(self.scale_indices, dtype=np.intp)
        n_feat = self.features.n_features
        if self.coef.shape != (n_feat, len(self.targets)):
            raise ValueError(
                f"coef has shape {self.coef.shape}, expected {(n_feat, len(self.targets))}"
            )
        if self.intercept.shape != (len(self.targets),):
            raise ValueError(f"intercept must have {len(self.targets)} elements")

    # ------------------------------------------------------------------ build
    @classmethod
    def from_estimator(
        cls,
        estimator: _Predictor,
        features: PairwiseDistanceFeatures,
        *,
        targets: tuple[str, ...] = TARGETS,
        metadata: dict[str, Any] | None = None,
        check: bool = True,
    ) -> LinearEmotionModel:
        """Extract the affine map implemented by any *linear* fitted estimator.

        The map is recovered by probing: ``intercept = f(0)`` and
        ``coef[i] = f(e_i) - f(0)``. This works for scikit-learn pipelines of
        scalers and linear models regardless of their internal parameterisation
        or library version.
        """
        n = features.n_features
        intercept = np.asarray(estimator.predict(np.zeros((1, n))), dtype=np.float64)[0]
        coef = np.asarray(estimator.predict(np.eye(n)), dtype=np.float64) - intercept
        model = cls(
            coef,
            intercept,
            features.landmark_indices,
            features.scale_indices,
            targets,
            dict(metadata or {}),
        )
        if check:
            rng = np.random.default_rng(0)
            probe = rng.normal(1.0, 0.5, size=(16, n))
            expected = np.asarray(estimator.predict(probe), dtype=np.float64)
            if not np.allclose(model.predict_features(probe), expected, rtol=1e-5, atol=1e-6):
                raise ValueError("Estimator is not linear; it cannot be exported")
        return model

    # -------------------------------------------------------------- inference
    @property
    def features(self) -> PairwiseDistanceFeatures:
        return PairwiseDistanceFeatures(
            landmark_indices=self.landmark_indices, scale=self.scale_indices
        )

    @property
    def subset(self) -> LandmarkSubset | None:
        """The named landmark subset, if the indices correspond to one."""
        for s in LandmarkSubset:
            if np.array_equal(s.indices, self.landmark_indices):
                return s
        return None

    def predict_features(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict from a ``(n, n_features)`` feature matrix."""
        return np.asarray(X, dtype=np.float64) @ self.coef + self.intercept

    def predict(self, frontal_landmarks: ArrayLike) -> NDArray[np.float64]:
        """Predict from frontalized landmarks ``(68, 2)`` or ``(n, 68, 2)``."""
        return _predict_landmarks(self, frontal_landmarks)

    # ------------------------------------------------------------- serialise
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coef=self.coef,
            intercept=self.intercept,
            landmark_indices=self.landmark_indices,
            scale_indices=self.scale_indices,
            targets=np.array(self.targets),
            metadata=np.array(json.dumps(self.metadata)),
            format_version=np.array(FORMAT_VERSION),
            kind=np.array("linear"),
        )
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> LinearEmotionModel:
        """Load a linear model; the bundled default PLS(31) model if ``path`` is ``None``."""
        if path is None:
            with resources.as_file(_bundled(DEFAULT_MODEL)) as p:
                return cls.load(p)
        with np.load(path, allow_pickle=False) as data:
            version = _check_version(data)
            if _kind(data) != "linear":
                raise ValueError(f"{path} is a {_kind(data)} model; use load_model()")
            return cls(
                coef=data["coef"],
                intercept=data["intercept"],
                landmark_indices=data["landmark_indices"],
                # format v1 normalised by the feature landmarks themselves
                scale_indices=data["scale_indices"] if version >= 2 else data["landmark_indices"],
                targets=tuple(str(t) for t in data["targets"]),
                metadata=json.loads(str(data["metadata"])),
            )


@dataclass
class KernelEmotionModel:
    """RBF kernel ridge regression on a Nystroem basis of training faces.

    ``z = (x - feature_mean) / feature_scale``;
    ``y = exp(-gamma * ||z - z_basis||^2) @ dual_coef + intercept``.

    Far from every basis face the prediction tends to ``intercept`` (the mean
    expression); :meth:`basis_distance` measures that distance.

    The basis is stored as the *frontalized landmarks* of the basis faces (much
    smaller than their feature vectors); their features are recomputed on load.

    Attributes
    ----------
    basis_landmarks
        ``(m, 68, 2)`` frontalized landmarks of the basis faces.
    dual_coef
        ``(m, n_targets)`` weights of the kernel columns.
    intercept
        ``(n_targets,)`` biases.
    feature_mean, feature_scale
        Standardisation of the features, ``(n_features,)``.
    gamma
        RBF width in the standardised feature space.
    """

    basis_landmarks: NDArray[np.float64]
    dual_coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    gamma: float
    landmark_indices: NDArray[np.intp]
    scale_indices: NDArray[np.intp]
    targets: tuple[str, ...] = TARGETS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.basis_landmarks = np.asarray(self.basis_landmarks, dtype=np.float64)
        self.dual_coef = np.asarray(self.dual_coef, dtype=np.float64)
        self.intercept = np.asarray(self.intercept, dtype=np.float64).reshape(-1)
        self.feature_mean = np.asarray(self.feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(self.feature_scale, dtype=np.float64)
        self.gamma = float(self.gamma)
        self.landmark_indices = np.asarray(self.landmark_indices, dtype=np.intp)
        self.scale_indices = np.asarray(self.scale_indices, dtype=np.intp)
        m, t, n = len(self.basis_landmarks), len(self.targets), self.features.n_features
        if self.dual_coef.shape != (m, t):
            raise ValueError(f"dual_coef has shape {self.dual_coef.shape}, expected {(m, t)}")
        if self.intercept.shape != (t,):
            raise ValueError(f"intercept must have {t} elements")
        if self.feature_mean.shape != (n,) or self.feature_scale.shape != (n,):
            raise ValueError(f"feature_mean and feature_scale must have {n} elements")
        self._basis = self._standardize(self.features.transform(self.basis_landmarks))
        self._basis_sq = (self._basis**2).sum(axis=1)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_estimator(
        cls,
        estimator: Any,
        features: PairwiseDistanceFeatures,
        train_frontal_landmarks: NDArray[np.floating],
        *,
        targets: tuple[str, ...] = TARGETS,
        metadata: dict[str, Any] | None = None,
        check_features: NDArray[np.floating] | None = None,
    ) -> KernelEmotionModel:
        """Export a fitted ``make_pipeline(StandardScaler(), Nystroem(rbf), Ridge())``.

        ``train_frontal_landmarks`` are the frontalized landmarks of the rows the
        pipeline was fitted on; the Nystroem basis rows are taken from them. If
        ``check_features`` is given, the exported model must reproduce the
        pipeline's predictions on them. Fit the pipeline on float64 features: the
        Nystroem normalisation amplifies float32 rounding errors.
        """
        scaler, nys, ridge = (step for _, step in estimator.steps)
        if nys.kernel != "rbf":
            raise ValueError(f"Only the RBF kernel is supported, got {nys.kernel!r}")
        coef = np.atleast_2d(ridge.coef_).T  # (n_basis, n_targets)
        model = cls(
            basis_landmarks=np.asarray(train_frontal_landmarks)[nys.component_indices_],
            dual_coef=nys.normalization_.T @ coef,
            intercept=np.atleast_1d(ridge.intercept_),
            feature_mean=scaler.mean_,
            feature_scale=scaler.scale_,
            gamma=nys.gamma,
            landmark_indices=features.landmark_indices,
            scale_indices=features.scale_indices,
            targets=targets,
            metadata=dict(metadata or {}),
        )
        if check_features is not None:
            expected = np.asarray(estimator.predict(check_features), dtype=np.float64)
            if not np.allclose(model.predict_features(check_features), expected, atol=1e-4):
                raise ValueError("Exported kernel model does not reproduce the estimator")
        return model

    # -------------------------------------------------------------- inference
    @property
    def features(self) -> PairwiseDistanceFeatures:
        return PairwiseDistanceFeatures(
            landmark_indices=self.landmark_indices, scale=self.scale_indices
        )

    def _standardize(self, X: ArrayLike) -> NDArray[np.float64]:
        return (np.asarray(X, dtype=np.float64) - self.feature_mean) / self.feature_scale

    def _sq_distances(self, X: ArrayLike) -> NDArray[np.float64]:
        Z = self._standardize(X)
        d2 = (Z**2).sum(axis=1)[:, None] + self._basis_sq - 2.0 * Z @ self._basis.T
        return np.maximum(d2, 0.0)

    def predict_features(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict from a ``(n, n_features)`` feature matrix."""
        K = np.exp(-self.gamma * self._sq_distances(X))
        return np.asarray(K @ self.dual_coef + self.intercept, dtype=np.float64)

    def basis_distance(self, X: ArrayLike) -> NDArray[np.float64]:
        """Distance of each feature row to its nearest basis face (standardised units).

        Large values flag faces unlike any training face - often a poor landmark
        fit rather than an unusual expression - whose estimates are unreliable.
        """
        return np.asarray(np.sqrt(self._sq_distances(X).min(axis=1)), dtype=np.float64)

    def predict(self, frontal_landmarks: ArrayLike) -> NDArray[np.float64]:
        """Predict from frontalized landmarks ``(68, 2)`` or ``(n, 68, 2)``."""
        return _predict_landmarks(self, frontal_landmarks)

    # ------------------------------------------------------------- serialise
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "basis_landmarks": self.basis_landmarks,
            "dual_coef": self.dual_coef,
            "intercept": self.intercept,
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "gamma": np.array(self.gamma),
            "landmark_indices": self.landmark_indices,
            "scale_indices": self.scale_indices,
            "targets": np.array(self.targets),
            "metadata": np.array(json.dumps(self.metadata)),
            "format_version": np.array(FORMAT_VERSION),
            "kind": np.array("kernel"),
        }
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> KernelEmotionModel:
        """Load a kernel model; the bundled kernel model if ``path`` is ``None``."""
        if path is None:
            with resources.as_file(_bundled(KERNEL_MODEL)) as p:
                return cls.load(p)
        with np.load(path, allow_pickle=False) as data:
            _check_version(data)
            if _kind(data) != "kernel":
                raise ValueError(f"{path} is a {_kind(data)} model; use load_model()")
            return cls(
                basis_landmarks=data["basis_landmarks"],
                dual_coef=data["dual_coef"],
                intercept=data["intercept"],
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                gamma=float(data["gamma"]),
                landmark_indices=data["landmark_indices"],
                scale_indices=data["scale_indices"],
                targets=tuple(str(t) for t in data["targets"]),
                metadata=json.loads(str(data["metadata"])),
            )
