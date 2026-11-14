"""Model-evaluation utilities: grouped cross-validation, regularisation paths, statistics.

Faces of the same person are strongly correlated (MorphSet morphs are
interpolations of a handful of photos), so every resampling scheme here works
on *groups* (persons), never on individual rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .metrics import ccc

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- splits
@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train: NDArray[np.intp]
    test: NDArray[np.intp]


def repeated_group_kfold(
    groups: NDArray[np.integer], n_splits: int = 5, n_repeats: int = 1, seed: int = 0
) -> list[Fold]:
    """Group k-fold repeated with different shuffles of the groups.

    Every repeat is a partition of the groups into ``n_splits`` folds; folds of
    different repeats are therefore *paired* when the same seed is used, which
    allows paired comparisons between models.
    """
    from sklearn.model_selection import GroupKFold

    folds = []
    dummy = np.zeros(len(groups))
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (train, test) in enumerate(gkf.split(dummy, groups=groups)):
            folds.append(Fold(r, k, train, test))
    return folds


def leave_one_group_out(labels: NDArray[np.generic]) -> Iterator[tuple[str, NDArray[np.intp]]]:
    """Yield ``(held_out_label, test_mask_indices)`` for every distinct label."""
    for label in pd.unique(labels):
        yield str(label), np.flatnonzero(labels == label)


# ---------------------------------------------------------------- linear model paths
def _standardize(
    X_train: NDArray[np.floating], X_test: NDArray[np.floating]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    mean = X_train.mean(axis=0, dtype=np.float64)
    std = X_train.std(axis=0, ddof=1, dtype=np.float64)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


@dataclass
class LinearPath:
    """Coefficients of a family of linear models ``y = X @ coef[k] + intercept[k]``.

    ``params`` holds the hyper-parameter of each member (e.g. number of PLS
    components or ridge penalty).
    """

    params: NDArray[np.float64]
    coef: NDArray[np.float64]  # (n_params, n_features, n_targets)
    intercept: NDArray[np.float64]  # (n_params, n_targets)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return predictions of every member, shape ``(n_params, n_samples, n_targets)``."""
        out = np.einsum("nf,kft->knt", X.astype(np.float64), self.coef)
        return np.asarray(out + self.intercept[:, None], dtype=np.float64)


def pls_path(X: NDArray[np.floating], y: NDArray[np.floating], max_components: int) -> LinearPath:
    """Fit PLS once and return the models with ``1..max_components`` components.

    PLS (NIPALS) extracts components greedily, so the first ``k`` components of
    a ``K``-component fit are those of a ``k``-component fit. One fit therefore
    yields the whole regularisation path. X and y are standardised exactly as
    :class:`sklearn.cross_decomposition.PLSRegression` does (``scale=True``).
    """
    from sklearn.cross_decomposition import PLSRegression

    Xs, _, x_mean, x_std = _standardize(X, X[:1])
    y = np.asarray(y, dtype=np.float64)
    y_mean, y_std = y.mean(axis=0), y.std(axis=0, ddof=1)
    pls = PLSRegression(n_components=max_components, scale=False).fit(Xs, (y - y_mean) / y_std)
    rot, load = pls.x_rotations_, pls.y_loadings_
    coefs, intercepts = [], []
    for k in range(1, max_components + 1):
        c_std = rot[:, :k] @ load[:, :k].T  # standardised units
        coef = c_std / x_std[:, None] * y_std
        coefs.append(coef)
        intercepts.append(y_mean - x_mean @ coef)
    return LinearPath(
        np.arange(1, max_components + 1, dtype=np.float64), np.stack(coefs), np.stack(intercepts)
    )


def ridge_path(
    X: NDArray[np.floating], y: NDArray[np.floating], alphas: Sequence[float]
) -> LinearPath:
    """Ridge regression on standardised features for many penalties at once.

    Uses one eigen-decomposition of the Gram matrix, so the cost of extra
    penalties is negligible. ``alphas`` are expressed *per sample*: the penalty
    is ``alpha * n_samples * ||w||^2``, which keeps their scale independent of
    the training-set size.
    """
    Xs, _, x_mean, x_std = _standardize(X, X[:1])
    y = np.asarray(y, dtype=np.float64)
    y_mean = y.mean(axis=0)
    gram = Xs.T @ Xs
    xty = Xs.T @ (y - y_mean)
    evals, evecs = np.linalg.eigh(gram)
    proj = evecs.T @ xty
    n = Xs.shape[0]
    coefs, intercepts = [], []
    for a in alphas:
        c_std = evecs @ (proj / (evals + a * n)[:, None])
        coef = c_std / x_std[:, None]
        coefs.append(coef)
        intercepts.append(y_mean - x_mean @ coef)
    return LinearPath(np.asarray(alphas, dtype=np.float64), np.stack(coefs), np.stack(intercepts))


# ---------------------------------------------------------------------- scoring
def score_rows(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    target_names: Sequence[str],
    **keys: object,
) -> list[dict[str, object]]:
    """Long-format CCC / RMSE / MAE rows for each target, tagged with ``keys``."""
    rows: list[dict[str, object]] = []
    for k, name in enumerate(target_names):
        err = y_pred[:, k] - y_true[:, k]
        rows.append(
            {
                **keys,
                "target": name,
                "ccc": ccc(y_true[:, k], y_pred[:, k]),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "mae": float(np.mean(np.abs(err))),
            }
        )
    return rows


def paired_difference(
    scores: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "ccc",
    model_col: str = "model",
) -> pd.DataFrame:
    """Paired per-fold comparison of two models (``a - b``).

    Returns, for each target, the mean difference, a t-based 95% CI (corrected
    for the overlap of training sets with the Nadeau-Bengio factor), and the
    fraction of folds in which ``a`` beats ``b``.
    """
    from scipy import stats

    key = ["repeat", "fold", "target"]
    a = scores[scores[model_col] == model_a].set_index(key)[metric]
    b = scores[scores[model_col] == model_b].set_index(key)[metric]
    diff = (a - b).dropna().rename("diff").reset_index()
    n_folds_per_repeat = int(scores["fold"].max()) + 1
    test_train_ratio = 1 / (n_folds_per_repeat - 1)
    rows = []
    for target, d in diff.groupby("target", sort=False):
        x = d["diff"].to_numpy()
        n = len(x)
        # Nadeau & Bengio (2003) corrected resampled t-test.
        var = x.var(ddof=1) * (1 / n + test_train_ratio) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * np.sqrt(var) if n > 1 else np.nan
        rows.append(
            {
                "target": target,
                "mean_diff": x.mean(),
                "ci_low": x.mean() - half,
                "ci_high": x.mean() + half,
                "win_rate": float((x > 0).mean()),
                "n_folds": n,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    groups: NDArray[np.integer],
    metric: Callable[[NDArray[np.floating], NDArray[np.floating]], float] = ccc,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of a metric, resampling *groups* with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        stats_[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats_, [(1 - level) / 2, 1 - (1 - level) / 2])
    return metric(y_true, y_pred), float(lo), float(hi)
