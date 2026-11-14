"""Interpretation of linear emotion models built on pairwise-distance features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .constants import LEFT_BROW, LEFT_EYE, NOSE_BASE, NOSE_BRIDGE, RIGHT_BROW, RIGHT_EYE

FloatArray = NDArray[np.float64]

REGIONS: dict[str, tuple[int, ...]] = {
    "brows": (*LEFT_BROW, *RIGHT_BROW),
    "eyes": (*LEFT_EYE, *RIGHT_EYE),
    "nose": (*NOSE_BRIDGE, *NOSE_BASE),
    "mouth": tuple(range(48, 68)),
    "jaw": tuple(range(17)),
}
_REGION_OF = {lm: name for name, ids in REGIONS.items() for lm in ids}


def region_of(landmark: int) -> str:
    return _REGION_OF[int(landmark)]


def pair_regions(pairs: NDArray[np.integer]) -> list[str]:
    """Region label of each landmark pair, e.g. ``"brows-eyes"`` or ``"mouth"``."""
    out = []
    for a, b in pairs:
        ra, rb = sorted((region_of(a), region_of(b)))
        out.append(ra if ra == rb else f"{ra}-{rb}")
    return out


def standardized_coefficients(coef: FloatArray, X: NDArray[np.floating]) -> FloatArray:
    """Coefficients times feature standard deviation: effect of a 1-sd change of each feature."""
    return np.asarray(coef * X.std(axis=0, dtype=np.float64)[:, None], dtype=np.float64)


def pls_vip(pls: Any) -> FloatArray:
    """Variable importance in projection of a fitted ``PLSRegression``.

    ``VIP_j = sqrt(p * sum_a SS_a (w_ja / ||w_a||)^2 / sum_a SS_a)`` with
    ``SS_a = ||q_a||^2 t_a' t_a`` the variance of Y explained by component a.
    Features with VIP > 1 are more important than average.
    """
    w, t, q = pls.x_weights_, pls.x_scores_, pls.y_loadings_
    ss = (q**2).sum(axis=0) * (t**2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.asarray(np.sqrt(w.shape[0] * (wn**2 @ ss) / ss.sum()), dtype=np.float64)


def landmark_importance(weights: FloatArray, pairs: NDArray[np.integer], n: int = 68) -> FloatArray:
    """Distribute |weight| of each pair equally to its two landmarks."""
    imp = np.zeros(n)
    np.add.at(imp, pairs[:, 0], np.abs(weights) / 2)
    np.add.at(imp, pairs[:, 1], np.abs(weights) / 2)
    return imp


def region_shares(weights: FloatArray, pairs: NDArray[np.integer]) -> pd.Series:
    """Fraction of total |weight| carried by each region pair, sorted descending."""
    s = pd.Series(np.abs(weights), index=pair_regions(pairs)).groupby(level=0).sum()
    return (s / s.sum()).sort_values(ascending=False)


def component_region_profile(rotations: FloatArray, pairs: NDArray[np.integer]) -> pd.DataFrame:
    """Share of each PLS component's |rotation| mass per *landmark region*.

    Pair weights are split equally between the two landmarks' regions. The
    ``concentration`` column is ``1 - H / log(n_regions)`` (normalised entropy):
    1 means the component involves a single region, 0 a uniform spread.
    """
    regions = [r for r in REGIONS if r != "jaw"]
    ra = [region_of(a) for a in pairs[:, 0]]
    rb = [region_of(b) for b in pairs[:, 1]]
    rows = []
    for k in range(rotations.shape[1]):
        w = np.abs(rotations[:, k])
        mass = dict.fromkeys(regions, 0.0)
        for wi, a, b in zip(w, ra, rb, strict=True):
            mass[a] += wi / 2
            mass[b] += wi / 2
        total = sum(mass.values())
        p = np.array([mass[r] / total for r in regions])
        entropy = -(p[p > 0] * np.log(p[p > 0])).sum()
        rows.append(
            {"component": k + 1, **dict(zip(regions, p, strict=True)),
             "concentration": 1 - entropy / np.log(len(regions))}
        )  # fmt: skip
    return pd.DataFrame(rows).set_index("component")


def plot_pair_weights(
    ax: Any,
    shape: FloatArray,
    pairs: NDArray[np.integer],
    weights: FloatArray,
    top: int = 60,
    title: str | None = None,
) -> Any:
    """Draw the ``top`` strongest landmark pairs on a face shape.

    Red: the distance increases the output; blue: it decreases it. Line width
    and opacity scale with |weight|.
    """
    from .viz import plot_landmarks

    plot_landmarks(shape, ax=ax, color="0.75")
    order = np.argsort(-np.abs(weights))[:top]
    wmax = np.abs(weights[order]).max()
    for k in order[::-1]:
        a, b = pairs[k]
        mag = abs(weights[k]) / wmax
        ax.plot(
            shape[[a, b], 0], shape[[a, b], 1],
            color="#c0392b" if weights[k] > 0 else "#2e6fba",
            lw=0.5 + 2.5 * mag, alpha=0.25 + 0.75 * mag, solid_capstyle="round",
        )  # fmt: skip
    if title:
        ax.set_title(title)
    return ax


def distance_jacobian(shape: FloatArray, pairs: NDArray[np.integer]) -> FloatArray:
    """Jacobian of the pairwise distances w.r.t. the landmark coordinates.

    Returns ``(n_pairs, 136)`` in the ``[x_0..x_67, y_0..y_67]`` layout: a small
    landmark displacement ``d`` changes the distances by ``J @ d``.
    """
    n = shape.shape[0]
    a, b = pairs[:, 0], pairs[:, 1]
    diff = shape[a] - shape[b]
    unit = diff / np.linalg.norm(diff, axis=1, keepdims=True)
    J = np.zeros((len(pairs), 2 * n))
    rows = np.arange(len(pairs))
    for axis in (0, 1):
        J[rows, axis * n + a] += unit[:, axis]
        J[rows, axis * n + b] -= unit[:, axis]
    return J


def equivalent_movement(
    feature_direction: FloatArray, shape: FloatArray, pairs: NDArray[np.integer]
) -> FloatArray:
    """Smallest landmark displacement whose effect on the distances best matches a direction.

    Solves ``min ||d||`` s.t. ``J d ~ w`` (least squares) at the given face shape,
    which turns a direction in feature space (a PLS component, a coefficient
    vector) into a facial movement ``(68, 2)``. Rigid motions do not change
    distances, so they are excluded from the minimum-norm solution.
    """
    J = distance_jacobian(shape, pairs)
    d = np.linalg.lstsq(J, feature_direction, rcond=None)[0]
    n = shape.shape[0]
    return np.stack([d[:n], d[n:]], axis=1)


def movement_region_shares(movement: FloatArray) -> pd.Series:
    """Share of squared displacement per facial region."""
    energy = (movement**2).sum(axis=1)
    s = pd.Series({r: energy[list(ids)].sum() for r, ids in REGIONS.items()})
    return s / s.sum()


def plot_movement(
    ax: Any, shape: FloatArray, movement: FloatArray, title: str | None = None
) -> Any:
    """Draw a facial movement as arrows on a face shape."""
    from .viz import plot_landmarks

    plot_landmarks(shape, ax=ax, color="0.75")
    mag = np.linalg.norm(movement, axis=1)
    scale = 0.12 / (float(np.max(mag)) + 1e-12)
    keep = mag > 0.15 * float(np.max(mag))
    ax.quiver(
        shape[keep, 0], shape[keep, 1], movement[keep, 0] * scale, movement[keep, 1] * scale,
        angles="xy", scale_units="xy", scale=1, color="#c0392b", width=0.006,
    )  # fmt: skip
    if title:
        ax.set_title(title)
    return ax
