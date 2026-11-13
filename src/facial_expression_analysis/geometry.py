"""Vectorised shape-normalisation utilities for 68-point facial landmarks.

All functions accept either a single shape of size ``(68, 2)`` or a batch of
shapes of size ``(n, 68, 2)`` and return arrays of the same rank.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import LEFT_EYE, N_LANDMARKS, RIGHT_EYE

FloatArray = NDArray[np.floating]


def as_landmark_batch(landmarks: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    """Convert input to a float64 batch ``(n, 68, 2)``.

    Returns the batch and a flag telling whether the input was a single shape.
    """
    arr = np.asarray(landmarks, dtype=np.float64)
    single = arr.ndim == 2
    if single:
        arr = arr[np.newaxis]
    if arr.ndim != 3 or arr.shape[1:] != (N_LANDMARKS, 2):
        raise ValueError(
            f"Expected landmarks of shape (68, 2) or (n, 68, 2), got {np.shape(landmarks)}"
        )
    return arr, single


def shape_scale(landmarks: FloatArray) -> FloatArray:
    """Root-mean-square distance of the landmarks to their centroid.

    Accepts ``(..., k, 2)`` and returns ``(...)``.
    """
    centred = landmarks - landmarks.mean(axis=-2, keepdims=True)
    return np.sqrt(np.mean(np.sum(centred**2, axis=-1), axis=-1))


def eye_angle(landmarks: FloatArray) -> FloatArray:
    """Angle (radians) of the line joining the two eye centres, ``(...)``."""
    left = landmarks[..., LEFT_EYE.start : LEFT_EYE.stop, :].mean(axis=-2)
    right = landmarks[..., RIGHT_EYE.start : RIGHT_EYE.stop, :].mean(axis=-2)
    delta = right - left
    return np.arctan2(delta[..., 1], delta[..., 0])


def procrustes_normalize(
    landmarks: ArrayLike,
    *,
    translate: bool = True,
    scale: bool = True,
    rotate: bool = True,
) -> NDArray[np.float64]:
    """Remove translation, scale and in-plane rotation from facial shapes.

    The shape is centred on its centroid, divided by its RMS radius and rotated
    so that the line joining the eye centres is horizontal. See
    Vonikakis & Winkler, *Identity-Invariant Facial Landmark Frontalization for
    Facial Expression Analysis*, ICIP 2020.

    Parameters
    ----------
    landmarks
        ``(68, 2)`` or ``(n, 68, 2)`` array of ``(x, y)`` coordinates.
    translate, scale, rotate
        Which components of the similarity transform to compensate.

    Returns
    -------
    numpy.ndarray
        Normalised landmarks, same shape as the input.
    """
    batch, single = as_landmark_batch(landmarks)
    out = batch.copy()

    if translate:
        out -= out.mean(axis=1, keepdims=True)
    if scale:
        out /= shape_scale(out)[:, np.newaxis, np.newaxis]
    if rotate:
        theta = eye_angle(out)
        cos, sin = np.cos(theta), np.sin(theta)
        # Row-vector convention: p' = p @ R with R = [[c, -s], [s, c]], which
        # rotates by -theta and therefore levels the eyes.
        rot = np.stack([np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=-2)
        out = out @ rot

    return out[0] if single else out
