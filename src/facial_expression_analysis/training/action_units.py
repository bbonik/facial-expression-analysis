"""Geometric templates of facial action units (FACS) on the 68-point dlib face.

Each template is a symmetric displacement pattern of the landmarks that a muscle action
moves, written as ``landmark -> (outward, down)`` weights for the subject's left side and
mirrored to the right. ``outward`` points away from the vertical midline of the face and
``down`` follows image coordinates (y down). The templates are approximations drawn from the
FACS descriptions of each action unit (Ekman, Friesen & Hager, 2002); they describe the
direction of the movement, not its size. Used in ``experiments/14_pls_dimensionality.py`` to
relate PLS components to action units.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..constants import MIRROR_PERMUTATION

_LEFT: Final[dict[str, dict[int, tuple[float, float]]]] = {
    "AU1 inner brow raiser": {21: (0, -1), 20: (0, -0.6), 19: (0, -0.25)},
    "AU2 outer brow raiser": {17: (0, -1), 18: (0, -0.8), 19: (0, -0.4)},
    "AU4 brow lowerer": {21: (-0.5, 0.8), 20: (-0.3, 0.6), 19: (0, 0.4), 18: (0, 0.2)},
    "AU5 upper lid raiser": {37: (0, -1), 38: (0, -1)},
    "AU6 cheek raiser": {40: (0, -1), 41: (0, -1), 36: (0, -0.3)},
    "AU7 lid tightener": {40: (0, -1), 41: (0, -1), 37: (0, 0.4), 38: (0, 0.4)},
    "AU9 nose wrinkler": {
        31: (0, -1),
        32: (0, -0.6),
        33: (0, -0.4),
        21: (0, 0.4),
        50: (0, -0.4),
        51: (0, -0.3),
    },
    "AU10 upper lip raiser": {
        50: (0, -1),
        51: (0, -0.8),
        49: (0, -0.5),
        61: (0, -0.7),
        62: (0, -0.6),
    },
    "AU12 lip corner puller": {48: (0.7, -0.7), 60: (0.5, -0.5), 49: (0.2, -0.3), 59: (0.2, -0.2)},
    "AU15 lip corner depressor": {48: (0, 1), 60: (0, 0.7), 59: (0, 0.3)},
    "AU17 chin raiser": {57: (0, -1), 58: (0, -0.7), 66: (0, -0.8), 67: (0, -0.6)},
    "AU20 lip stretcher": {48: (1, 0), 60: (0.8, 0), 49: (0.3, 0.1), 59: (0.3, 0)},
    "AU23 lip tightener": {
        50: (0, 0.6),
        51: (0, 0.6),
        49: (0, 0.3),
        58: (0, -0.6),
        57: (0, -0.6),
        59: (0, -0.3),
        48: (-0.3, 0),
    },
    "AU25 lips part": {
        61: (0, -0.3),
        62: (0, -0.4),
        67: (0, 0.8),
        66: (0, 1),
        58: (0, 0.3),
        57: (0, 0.4),
    },
    "AU26 jaw drop": {
        57: (0, 1),
        58: (0, 0.8),
        59: (0, 0.5),
        66: (0, 1),
        67: (0, 0.9),
        48: (0, 0.3),
    },
    "AU27 mouth stretch": {
        57: (0, 1),
        58: (0, 0.9),
        59: (0, 0.7),
        66: (0, 1),
        67: (0, 0.9),
        48: (-0.4, 0.4),
        60: (-0.4, 0.3),
        62: (0, -0.2),
    },
}

ACTION_UNITS: Final[tuple[str, ...]] = tuple(_LEFT)


def _bilateral(left: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
    perm = np.asarray(MIRROR_PERMUTATION)
    out = dict(left)
    for i, w in left.items():
        out.setdefault(int(perm[i]), w)  # midline landmarks map to themselves
    return out


def action_unit_field(name: str, face: NDArray[np.floating]) -> NDArray[np.float64]:
    """Unit-norm displacement field ``(68, 2)`` of an action unit on a given face shape."""
    if name not in _LEFT:
        raise KeyError(f"unknown action unit {name!r}; choose from {ACTION_UNITS}")
    shape = np.asarray(face, dtype=np.float64)
    midline = shape[27:31, 0].mean()  # nose bridge
    field = np.zeros((68, 2))
    for i, (outward, down) in _bilateral(_LEFT[name]).items():
        side = np.sign(shape[i, 0] - midline) or 1.0
        field[i] = (outward * side, down)
    return np.asarray(field / np.linalg.norm(field), dtype=np.float64)
