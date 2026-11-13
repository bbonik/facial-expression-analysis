"""Regression metrics for dimensional affect."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def ccc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Lin's concordance correlation coefficient.

    ``2 cov(t, p) / (var(t) + var(p) + (mean(t) - mean(p))**2)`` - penalises
    both decorrelation and systematic bias, and is the standard metric for
    continuous affect recognition.
    """
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    cov = np.mean((t - t.mean()) * (p - p.mean()))
    denom = t.var() + p.var() + (t.mean() - p.mean()) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def regression_report(
    y_true: ArrayLike, y_pred: ArrayLike, target_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """CCC, Pearson r, RMSE, MAE and MSE for each target column."""
    t = np.atleast_2d(np.asarray(y_true, dtype=np.float64))
    p = np.atleast_2d(np.asarray(y_pred, dtype=np.float64))
    if t.shape != p.shape or t.shape[1] != len(target_names):
        raise ValueError("y_true, y_pred and target_names have inconsistent shapes")
    report = {}
    for k, name in enumerate(target_names):
        err = p[:, k] - t[:, k]
        report[name] = {
            "ccc": ccc(t[:, k], p[:, k]),
            "pearson": float(np.corrcoef(t[:, k], p[:, k])[0, 1]),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "mse": float(np.mean(err**2)),
        }
    return report
