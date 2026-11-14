"""Training and evaluation of the landmark frontalization model.

Ported from https://github.com/bbonik/facial-landmark-frontalization
(``estimate_mean_face.py`` and ``fill_matrices.py``), following Vonikakis &
Winkler, *Identity-Invariant Facial Landmark Frontalization for Facial
Expression Analysis*, ICIP 2020.

For every (subject, expression) of a multi-pose dataset, each posed shape is
paired with the frontal shape of the same subject and expression. Inputs are
Procrustes-normalised shapes; targets are Procrustes-normalised frontal shapes
whose facial parts are additionally moved to the positions of a mean face
("identity invariance"), except for the eyebrows which follow their eye. A
linear map (least squares, optionally ridge) from inputs to targets is the
frontalization model.

The training data (``landmark_dataset.csv``) is anonymised. Its subjects fall
into three families with distinct camera grids and expression sets, which
identify the source databases:

=========  ======================================  ==============================
Family     Yaw / pitch grid                        Expressions
=========  ======================================  ==============================
multipie   0, ±15, ±30, ±45 (±60, ±90 rare); ±20   neutral, smile, surprise,
                                                   squint (SQ), disgust, scream (SC)
kdef       0, ±45; 0                               the 7 basic expressions
caspeal    0, ±22, ±45, ±67; ±20                   neutral, HA, SL (likely CAS-PEAL)
=========  ======================================  ==============================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .assets import fetch_pose_dataset
from .constants import MIRROR_PERMUTATION, N_LANDMARKS
from .data import LANDMARK_COLUMNS
from .geometry import procrustes_normalize

FloatArray = NDArray[np.float64]


class MirrorMode(StrEnum):
    """How mirrored copies are added to the frontalization training set."""

    NONE = "none"
    """No mirrored copies."""
    NAIVE = "naive"
    """Negate x only (2021 behaviour): left/right landmark ids end up swapped,
    i.e. the mirrored shapes do not follow the dlib convention."""
    PERMUTE = "permute"
    """Negate x and permute the ids, so mirrored shapes follow the dlib convention."""


@dataclass(frozen=True)
class PoseDataset:
    """Multi-pose landmark dataset.

    ``meta`` columns: ``subject``, ``expression``, ``yaw``, ``pitch``, ``family``.
    ``landmarks``: ``(n, 68, 2)`` image coordinates.
    """

    meta: pd.DataFrame
    landmarks: FloatArray

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, mask: NDArray[np.bool_]) -> PoseDataset:
        return PoseDataset(self.meta.loc[mask].reset_index(drop=True), self.landmarks[mask])


def _family(group: pd.DataFrame) -> str:
    yaws = set(group["yaw"].dropna().abs().astype(int))
    if set(group["expression"]) & {"AF", "AN", "SA"}:
        return "kdef"
    if yaws & {22, 67}:
        return "caspeal"
    return "multipie"


def load_pose_dataset(path: str | Path | None = None, *, drop_invalid: bool = True) -> PoseDataset:
    """Load the multi-pose dataset.

    ``drop_invalid`` removes the 12 rows with missing pose (two of them carry
    the malformed expression codes ``"V."`` and ``"H."``). Keep them only to
    reproduce the 2021 weights exactly.
    """
    df = pd.read_csv(path or fetch_pose_dataset())
    meta = df[["subject_id", "expression", "pose_yaw", "pose_pitch"]].rename(
        columns={"subject_id": "subject", "pose_yaw": "yaw", "pose_pitch": "pitch"}
    )
    families = {subject: _family(group) for subject, group in meta.groupby("subject")}
    meta["family"] = meta["subject"].map(families)
    coords = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float64)
    landmarks = coords.reshape(-1, 2, N_LANDMARKS).transpose(0, 2, 1).copy()
    ds = PoseDataset(meta, landmarks)
    if drop_invalid:
        ds = ds.subset((meta["yaw"].notna() & meta["pitch"].notna()).to_numpy())
    return ds


def mean_shape(landmarks: FloatArray) -> FloatArray:
    """Mean of the Procrustes-normalised shapes (the identity template)."""
    return np.asarray(procrustes_normalize(landmarks).mean(axis=0), dtype=np.float64)


# (anchor landmarks, landmarks moved with that anchor) for identity normalisation.
_PARTS: tuple[tuple[slice, tuple[slice, ...]], ...] = (
    (slice(50, 53), (slice(48, 68),)),  # mouth, anchored at the upper lip
    (slice(42, 48), (slice(42, 48), slice(22, 27))),  # right eye + its eyebrow
    (slice(36, 42), (slice(36, 42), slice(17, 22))),  # left eye + its eyebrow
    (slice(27, 36), (slice(27, 36),)),  # nose
    (slice(0, 17), (slice(0, 17),)),  # jaw
)


def align_parts_to_template(shapes: FloatArray, template: FloatArray) -> FloatArray:
    """Move each facial part so that its anchor matches the template (identity invariance).

    Eyebrows move with their eye, so eyebrow-to-eye distances - which convey
    expression - are preserved. Accepts ``(68, 2)`` or ``(n, 68, 2)`` normalised shapes.
    """
    out = np.array(shapes, dtype=np.float64, copy=True)
    for anchor, parts in _PARTS:
        shift = template[anchor].mean(axis=0) - out[..., anchor, :].mean(axis=-2)
        for part in parts:
            out[..., part, :] += shift[..., np.newaxis, :]
    return out


def mirror_shapes(shapes: FloatArray, mode: MirrorMode) -> FloatArray:
    """Horizontally mirror centred shapes ``(n, 68, 2)``."""
    flipped = np.asarray(shapes * np.array([-1.0, 1.0]), dtype=np.float64)
    if mode is MirrorMode.PERMUTE:
        flipped = flipped[:, list(MIRROR_PERMUTATION)]
    return flipped


@dataclass
class TrainingPairs:
    inputs: FloatArray  # (m, 68, 2) normalised posed shapes
    targets: FloatArray  # (m, 68, 2) identity-normalised frontal shapes
    row: NDArray[np.intp]  # source row of each pair in the dataset
    mirrored: NDArray[np.bool_]


def build_training_pairs(
    data: PoseDataset, template: FloatArray, mirror: MirrorMode = MirrorMode.PERMUTE
) -> TrainingPairs:
    """Pair every shape with the frontal shape of the same subject and expression.

    As in the 2021 code, the first frontal row (yaw = pitch = 0) of each group is
    the target, and the frontal row itself is also used as an input.
    """
    meta = data.meta.reset_index(drop=True)
    frontal = (meta["yaw"] == 0) & (meta["pitch"] == 0)
    key = meta["subject"].astype(str) + "|" + meta["expression"].astype(str)
    first_frontal = meta.index.to_series()[frontal].groupby(key[frontal]).first()
    target_row = key.map(first_frontal)
    valid = target_row.notna().to_numpy()
    rows = np.flatnonzero(valid)
    tgt = target_row[valid].to_numpy(dtype=np.intp)

    inputs = procrustes_normalize(data.landmarks[rows])
    targets = align_parts_to_template(procrustes_normalize(data.landmarks[tgt]), template)
    mirrored = np.zeros(len(rows), dtype=bool)
    if mirror is not MirrorMode.NONE:
        inputs = np.concatenate([inputs, mirror_shapes(inputs, mirror)])
        targets = np.concatenate([targets, mirror_shapes(targets, mirror)])
        rows = np.concatenate([rows, rows])
        mirrored = np.concatenate([mirrored, ~mirrored])
    return TrainingPairs(inputs, targets, rows, mirrored)


def _flatten(shapes: FloatArray) -> FloatArray:
    """``(m, 68, 2)`` -> ``(m, 136)`` in the ``[x..., y...]`` layout of the weights."""
    return np.concatenate([shapes[:, :, 0], shapes[:, :, 1]], axis=1)


def fit_frontalization(pairs: TrainingPairs, ridge: float = 0.0) -> FloatArray:
    """Least-squares (or ridge) map from inputs to targets, shape ``(137, 136)``.

    ``ridge`` is a per-sample penalty on the weights (not on the intercept).
    """
    A = _flatten(pairs.inputs)
    Y = _flatten(pairs.targets)
    if ridge == 0.0:
        design = np.hstack([A, np.ones((len(A), 1))])
        weights, *_ = np.linalg.lstsq(design, Y, rcond=None)
        return np.asarray(weights)
    a_mean, y_mean = A.mean(0), Y.mean(0)
    Ac, Yc = A - a_mean, Y - y_mean
    W = np.linalg.solve(Ac.T @ Ac + ridge * len(A) * np.eye(A.shape[1]), Ac.T @ Yc)
    b = y_mean - a_mean @ W
    return np.vstack([W, b])


def frontalization_error(frontal_pred: FloatArray, frontal_true: FloatArray) -> NDArray[np.float64]:
    """Mean Euclidean landmark error per face, in normalised (RMS-radius = 1) units."""
    return np.asarray(np.linalg.norm(frontal_pred - frontal_true, axis=-1).mean(axis=-1))


def assign_pose_persons(data: PoseDataset) -> NDArray[np.int64]:
    """Group subject ids that are (very likely) the same person.

    Subject ids are recording sessions, so one person can have several ids. The
    rules come from the Procrustes distances between frontal neutral shapes:

    * **kdef** - exactly two sessions per person; each id's nearest neighbour is
      almost always its twin, so ids are paired greedily by distance (< 0.35).
    * **multipie** - a person's sessions have nearby ids, but the signal is weaker;
      ids are linked generously when ``|id_a - id_b| <= 3`` and distance < 0.45.
    * **caspeal** - one id per person (no near-duplicates).

    Over-merging only makes cross-validation groups coarser, which is the safe
    direction for leakage. Returns a person id for every row.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial.distance import cdist

    meta = data.meta
    subjects = np.unique(meta["subject"])
    index = {s: k for k, s in enumerate(subjects)}
    ref = meta[(meta["yaw"] == 0) & (meta["pitch"] == 0) & (meta["expression"] == "NE")]
    ref = ref.drop_duplicates("subject")
    edges: list[tuple[int, int]] = []
    for family, grp in ref.groupby("family"):
        ids = grp["subject"].to_numpy()
        shapes = procrustes_normalize(data.landmarks[grp.index.to_numpy()]).reshape(len(grp), -1)
        dist = cdist(shapes, shapes)
        np.fill_diagonal(dist, np.inf)
        if family == "kdef":
            paired: set[int] = set()
            for flat in np.argsort(dist, axis=None):
                a, b = np.unravel_index(flat, dist.shape)
                if dist[a, b] >= 0.35:
                    break
                if a not in paired and b not in paired:
                    paired |= {int(a), int(b)}
                    edges.append((index[ids[a]], index[ids[b]]))
        elif family == "multipie":
            close = (dist < 0.45) & (np.abs(ids[:, None] - ids[None, :]) <= 3)
            i, j = np.nonzero(np.triu(close, k=1))
            edges += [(index[ids[a]], index[ids[b]]) for a, b in zip(i, j, strict=True)]
    n = len(subjects)
    r, c = (list(x) for x in zip(*edges, strict=True)) if edges else ([], [])
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels[[index[s] for s in meta["subject"]]], dtype=np.int64)
