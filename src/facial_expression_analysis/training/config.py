"""Training configuration (no heavy dependencies, so the CLI can import it cheaply)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from ..features import LandmarkSubset, PairwiseDistanceFeatures, ScaleReference


class Estimator(StrEnum):
    """Regressor families that can be trained and exported."""

    PLS = "pls"  # partial least squares (default: linear, interpretable, well-behaved live)
    KERNEL_RIDGE = "kernel_ridge"  # RBF kernel ridge on a Nystroem basis (highest offline CCC)


@dataclass
class TrainConfig:
    """Configuration of a training run.

    ``n_components`` applies to PLS; ``gamma`` (RBF width, in units of
    ``1 / n_features`` on standardised features), ``alpha`` (ridge penalty) and
    ``n_basis`` (Nystroem basis size) to kernel ridge. The kernel defaults were
    selected by nested CV in ``experiments/08_model_comparison.py``.
    """

    estimator: Estimator = Estimator.PLS
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
