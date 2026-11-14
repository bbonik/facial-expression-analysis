"""Geometric features: normalised distances between all pairs of landmarks."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import INTRAFACE_49, JAW, N_LANDMARKS, STABLE_POINTS
from .geometry import as_landmark_batch, shape_scale


class LandmarkSubset(StrEnum):
    """Which landmarks contribute to the pairwise-distance features."""

    ALL = "all"
    """All 68 landmarks -> C(68, 2) = 2278 features."""
    NO_JAW = "no_jaw"
    """The 51 non-jaw landmarks -> C(51, 2) = 1275 features."""
    INTRAFACE_49 = "intraface49"
    """The 49-point IntraFace scheme used in the original MATLAB system: no jaw
    and no inner-mouth corners (dlib 60, 64) -> C(49, 2) = 1176 features."""

    @property
    def indices(self) -> NDArray[np.intp]:
        if self is LandmarkSubset.ALL:
            return np.arange(N_LANDMARKS)
        if self is LandmarkSubset.INTRAFACE_49:
            return np.array(INTRAFACE_49, dtype=np.intp)
        return np.arange(JAW.stop, N_LANDMARKS)


class ScaleReference(StrEnum):
    """Landmarks used to estimate the face scale that normalises distances."""

    STABLE = "stable"
    """Nose and eye corners, which barely move with expressions. An open mouth
    or raised eyebrows do not inflate the scale estimate."""
    SUBSET = "subset"
    """The same landmarks as the features (behaviour of the 2021 model)."""


def _validate_indices(indices: ArrayLike, name: str) -> NDArray[np.intp]:
    idx = np.asarray(indices, dtype=np.intp)
    if idx.ndim != 1 or len(idx) < 2 or len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} must be a 1-D array of at least 2 unique indices")
    if np.min(idx) < 0 or np.max(idx) >= N_LANDMARKS:
        raise ValueError(f"{name} must lie in [0, {N_LANDMARKS - 1}]")
    return idx


class PairwiseDistanceFeatures:
    """Euclidean distances between every pair of selected landmarks.

    Each distance is divided by the RMS radius of a set of *reference*
    landmarks, which makes the features invariant to translation, rotation and
    scale. See Vonikakis & Winkler, *Efficient Facial Expression Analysis for
    Dimensional Affect Recognition Using Geometric Features* (2021).

    Parameters
    ----------
    subset
        Landmarks whose pairwise distances are computed.
    scale
        Landmarks used for the scale estimate: a :class:`ScaleReference` or an
        explicit array of landmark indices.
    landmark_indices
        Explicit landmark indices; overrides ``subset``.
    """

    def __init__(
        self,
        subset: LandmarkSubset | str = LandmarkSubset.NO_JAW,
        *,
        scale: ScaleReference | str | ArrayLike = ScaleReference.STABLE,
        landmark_indices: ArrayLike | None = None,
    ) -> None:
        if landmark_indices is None:
            self.subset: LandmarkSubset | None = LandmarkSubset(subset)
            idx = self.subset.indices
        else:
            self.subset = None
            idx = _validate_indices(landmark_indices, "landmark_indices")
        self.landmark_indices = idx

        if isinstance(scale, str):
            ref = ScaleReference(scale)
            scale_idx = (
                np.array(STABLE_POINTS, dtype=np.intp) if ref is ScaleReference.STABLE else idx
            )
        else:
            scale_idx = _validate_indices(scale, "scale")
        self.scale_indices: NDArray[np.intp] = scale_idx

        i, j = np.triu_indices(len(idx), k=1)
        # Pairs hold *landmark ids*, not positions within the subset (the 2021
        # code, ported from a 49-point IntraFace system, mixed the two up).
        self.pairs: NDArray[np.intp] = np.stack([idx[i], idx[j]], axis=1)

    @property
    def n_features(self) -> int:
        return len(self.pairs)

    @property
    def scale_reference(self) -> ScaleReference | None:
        """The named scale reference, if the scale indices correspond to one."""
        if np.array_equal(self.scale_indices, STABLE_POINTS):
            return ScaleReference.STABLE
        if np.array_equal(self.scale_indices, self.landmark_indices):
            return ScaleReference.SUBSET
        return None

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
            scale = shape_scale(chunk[:, self.scale_indices])
            out[start : start + chunk_size] = dist / scale[:, np.newaxis]
        return out[0] if single else out
