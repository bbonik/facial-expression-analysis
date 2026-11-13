"""Linear frontalization of 68-point facial landmarks."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import N_LANDMARKS
from .geometry import as_landmark_batch, procrustes_normalize

DEFAULT_WEIGHTS = "frontalization_dlib68.npy"


class Frontalizer:
    """Estimate how a (possibly non-frontal) face shape looks from the front.

    The model is a linear regression from Procrustes-normalised landmarks to
    frontal landmarks, as described in Vonikakis & Winkler (ICIP 2020). The
    bundled weights were learnt on dlib landmarks; for other landmark engines
    the model must be retrained (see
    https://github.com/bbonik/facial-landmark-frontalization).

    Parameters
    ----------
    weights
        Array of shape ``(2N + 1, 2N)`` mapping ``[x_1..x_N, y_1..y_N, 1]`` to
        ``[x'_1..x'_N, y'_1..y'_N]``. Loaded from the bundled file if ``None``.
    """

    def __init__(self, weights: ArrayLike | None = None) -> None:
        if weights is None:
            weights = self._load_bundled()
        w = np.asarray(weights, dtype=np.float64)
        expected = (2 * N_LANDMARKS + 1, 2 * N_LANDMARKS)
        if w.shape != expected:
            raise ValueError(f"Frontalization weights must have shape {expected}, got {w.shape}")
        self.weights = w

    @classmethod
    def from_file(cls, path: str | Path) -> Frontalizer:
        """Load weights from a ``.npy`` file."""
        return cls(np.load(path, allow_pickle=False))

    @staticmethod
    def _load_bundled() -> NDArray[np.float64]:
        ref = resources.files(__package__).joinpath("assets", DEFAULT_WEIGHTS)
        with resources.as_file(ref) as path:
            return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)

    def __call__(self, landmarks: ArrayLike) -> NDArray[np.float64]:
        return self.frontalize(landmarks)

    def frontalize(self, landmarks: ArrayLike) -> NDArray[np.float64]:
        """Frontalize ``(68, 2)`` or ``(n, 68, 2)`` landmarks.

        Returns frontal landmarks in the normalised (Procrustes) coordinate
        frame, with the same rank as the input.
        """
        batch, single = as_landmark_batch(landmarks)
        normalised = procrustes_normalize(batch)
        n = normalised.shape[0]
        design = np.concatenate([normalised[:, :, 0], normalised[:, :, 1], np.ones((n, 1))], axis=1)
        frontal = (design @ self.weights).reshape(n, 2, N_LANDMARKS).transpose(0, 2, 1)
        return frontal[0] if single else frontal
