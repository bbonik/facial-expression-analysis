"""Portable linear emotion regressor.

Every model shipped with this package (PLS, ridge, ...) is linear in the
geometric features, so inference reduces to ``y = X @ coef + intercept``. The
weights are stored in a small ``.npz`` file together with JSON metadata, which
removes any dependency on scikit-learn (or its pickle format) at inference
time.
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
DEFAULT_MODEL = "emotion_model.npz"
FORMAT_VERSION = 1


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
    targets
        Names of the output columns.
    metadata
        Free-form JSON-serialisable information (training config, metrics, ...).
    """

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    landmark_indices: NDArray[np.intp]
    targets: tuple[str, ...] = TARGETS
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coef = np.asarray(self.coef, dtype=np.float64)
        self.intercept = np.asarray(self.intercept, dtype=np.float64).reshape(-1)
        self.landmark_indices = np.asarray(self.landmark_indices, dtype=np.intp)
        n_feat = PairwiseDistanceFeatures(landmark_indices=self.landmark_indices).n_features
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
        model = cls(coef, intercept, features.landmark_indices, targets, dict(metadata or {}))
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
        return PairwiseDistanceFeatures(landmark_indices=self.landmark_indices)

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
        feats = self.features.transform(frontal_landmarks)
        return self.predict_features(np.atleast_2d(feats)).reshape(
            (*np.shape(feats)[:-1], len(self.targets))
        )

    # ------------------------------------------------------------- serialise
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coef=self.coef,
            intercept=self.intercept,
            landmark_indices=self.landmark_indices,
            targets=np.array(self.targets),
            metadata=np.array(json.dumps(self.metadata)),
            format_version=np.array(FORMAT_VERSION),
        )
        return path

    @classmethod
    def load(cls, path: str | Path | None = None) -> LinearEmotionModel:
        """Load a model; the bundled default model if ``path`` is ``None``."""
        if path is None:
            ref = resources.files(__package__).joinpath("assets", DEFAULT_MODEL)
            with resources.as_file(ref) as p:
                return cls.load(p)
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version > FORMAT_VERSION:
                raise ValueError(f"Unsupported model format version {version}")
            return cls(
                coef=data["coef"],
                intercept=data["intercept"],
                landmark_indices=data["landmark_indices"],
                targets=tuple(str(t) for t in data["targets"]),
                metadata=json.loads(str(data["metadata"])),
            )
