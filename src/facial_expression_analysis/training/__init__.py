"""Training, evaluation and analysis tools (need the ``train`` extra: scikit-learn, pandas).

* :mod:`.data` - MorphSet loading, person ids and person-wise splits
* :mod:`.train` - training and exporting emotion models
* :mod:`.evaluation` - grouped CV, regularisation paths, paired tests, bootstrap CIs
* :mod:`.metrics` - CCC, RMSE, MAE
* :mod:`.frontalization` - retraining the landmark frontalization model
* :mod:`.interpret` - VIP scores, region shares and facial movements of linear models
"""

from typing import TYPE_CHECKING, Any

from .config import Estimator, TrainConfig

if TYPE_CHECKING:
    from .train import (
        TrainResult,
        compute_features,
        cross_validate_pls,
        evaluate_model,
        neutral_mean,
        train_model,
        train_pls,
    )

_LAZY = {
    "TrainResult",
    "compute_features",
    "cross_validate_pls",
    "evaluate_model",
    "neutral_mean",
    "train_model",
    "train_pls",
}


def __getattr__(name: str) -> Any:
    # scikit-learn and pandas are imported only when training code is actually used
    if name in _LAZY:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Estimator",
    "TrainConfig",
    "TrainResult",
    "compute_features",
    "cross_validate_pls",
    "evaluate_model",
    "neutral_mean",
    "train_model",
    "train_pls",
]
