"""Geometric features: normalised distances between all pairs of landmarks."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import JAW, N_LANDMARKS
from .geometry import as_landmark_batch, shape_scale


class LandmarkSubset(StrEnum):
    """Which landmarks contribute to the pairwise-distance features."""

    ALL = "all"
    """All 68 landmarks -> C(68, 2) = 2278 features."""
    NO_JAW = "no_jaw"
    """Exclude the 17 jaw-line landmarks -> C(51, 2) = 1275 features."""

    @property
    def indices(self) -> NDArray[np.intp]:
        if self is LandmarkSubset.ALL:
            return np.arange(N_LANDMARKS)
        return np.arange(JAW.stop, N_LANDMARKS)


class PairwiseDistanceFeatures:
    """Euclidean distances between every pair of selected landmarks.

    Distances are divided by the RMS radius of the selected landmarks, which
    makes the features invariant to translation, rotation and scale. See
    Vonikakis & Winkler, *Efficient Facial Expression Analysis for Dimensional
    Affect Recognition Using Geometric Features* (2021).

    Parameters
    ----------
    subset
        Landmarks to use.
    landmark_indices
        Explicit landmark indices; overrides ``subset``. Mainly useful for
        experiments and for reproducing legacy models.
    """

    def __init__(
        self,
        subset: LandmarkSubset | str = LandmarkSubset.NO_JAW,
        *,
        landmark_indices: ArrayLike | None = None,
    ) -> None:
        if landmark_indices is None:
            self.subset: LandmarkSubset | None = LandmarkSubset(subset)
            idx = self.subset.indices
        else:
            self.subset = None
            idx = np.asarray(landmark_indices, dtype=np.intp)
            if idx.ndim != 1 or len(np.unique(idx)) != len(idx):
                raise ValueError("landmark_indices must be a 1-D array of unique indices")
            if np.min(idx) < 0 or np.max(idx) >= N_LANDMARKS:
                raise ValueError(f"landmark_indices must lie in [0, {N_LANDMARKS - 1}]")
        self.landmark_indices = idx
        i, j = np.triu_indices(len(idx), k=1)
        # Store *landmark ids*, not positions within the subset.
        self.pairs: NDArray[np.intp] = np.stack([idx[i], idx[j]], axis=1)

    @property
    def n_features(self) -> int:
        return len(self.pairs)

    def feature_names(self) -> list[str]:
        """Human-readable names such as ``"d_17_48"``."""
        return [f"d_{a}_{b}" for a, b in self.pairs]

    def __call__(self, landmarks: ArrayLike, chunk_size: int = 8192) -> NDArray[np.float32]:
        return self.transform(landmarks, chunk_size=chunk_size)

    def transform(self, landmarks: ArrayLike, chunk_size: int = 8192) -> NDArray[np.float32]:
        """Compute features for ``(68, 2)`` or ``(n, 68, 2)`` landmarks.

        Returns ``(n_features,)`` or ``(n, n_features)`` float32 features. Large
        batches are processed in chunks to bound peak memory.
        """
        batch, single = as_landmark_batch(landmarks)
        out = np.empty((batch.shape[0], self.n_features), dtype=np.float32)
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        for start in range(0, batch.shape[0], chunk_size):
            chunk = batch[start : start + chunk_size]
            dist = np.linalg.norm(chunk[:, a] - chunk[:, b], axis=-1)
            scale = shape_scale(chunk[:, self.landmark_indices])
            out[start : start + chunk_size] = dist / scale[:, np.newaxis]
        return out[0] if single else out
