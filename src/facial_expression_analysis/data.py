"""Loading and splitting of the MorphSet dataset.

MorphSet (Vonikakis et al., ICIP 2021) contains dlib landmarks of face images
morphed between a neutral face and categorical expressions, with dimensional
labels (arousal, valence, intensity) derived from the morphing coefficients.

Subjects vs. persons
--------------------
The ``Subject`` column is *not* a person identifier. Some source databases
photograph each person several times, and every photo series got its own
subject id:

====  ========  ==========================================  ========
Code  Database  Series per person                           Persons
====  ========  ==========================================  ========
R     RaFD      3 consecutive ids (gaze directions)         67
K     KDEF      2 consecutive ids (sessions A and B)        70
F     FACES     2 consecutive ids (picture sets a and b)    171
W     WSEFEP    1 id                                        29
====  ========  ==========================================  ========

This was established from the data (the nearest neighbour of a subject's
neutral face shape lies in the same id group 83-94% of the time, against ~1%
by chance) and matches the person counts of the source databases. Splits must
therefore be made by :attr:`MorphSet.persons`, otherwise the same person
appears in training and test data.

Mirrored copies
---------------
Every image exists twice: ``..._1.jpg`` is the original and ``..._2.jpg`` its
horizontal mirror, with landmarks re-detected by dlib on the flipped image
(left/right landmark semantics are therefore preserved; the median residual
between a face and its un-mirrored copy is 1.8 px, i.e. detector noise). A copy
shares subject, person and targets with its original, so person-wise splits
keep both together; the dataset holds 83,191 distinct images.

All faces are frontal and eye-aligned (inter-ocular distance ~150 px).
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .assets import fetch_morphset
from .constants import N_LANDMARKS
from .model import TARGETS

META_COLUMNS = ["Filename", "Dataset", "Subject", "Emotion Label", "Agreement"]
TARGET_COLUMNS = ["Arousal", "Valence", "Intensity"]
LANDMARK_COLUMNS = [f"x{i}" for i in range(N_LANDMARKS)] + [f"y{i}" for i in range(N_LANDMARKS)]

# dataset code -> (first subject id, consecutive subject ids per person)
_PERSON_GROUPING = {"R": (1, 3), "K": (202, 2), "W": (342, 1), "F": (371, 2)}


def assign_person_ids(meta: pd.DataFrame) -> NDArray[np.int64]:
    """Map MorphSet ``(dataset, subject)`` pairs to person ids (see module docs)."""
    person = np.full(len(meta), -1, dtype=np.int64)
    offset = 0
    for code, (first, per_person) in _PERSON_GROUPING.items():
        mask = (meta["dataset"] == code).to_numpy()
        if not mask.any():
            continue
        local = (meta.loc[mask, "subject"].to_numpy() - first) // per_person
        if local.min() < 0:
            raise ValueError(f"Unexpected subject ids for dataset {code!r}")
        person[mask] = offset + local
        offset += int(local.max()) + 1
    if (person < 0).any():
        unknown = sorted(set(meta.loc[person < 0, "dataset"].astype(str)))
        raise ValueError(f"Unknown dataset codes: {unknown}")
    return person


@dataclass(frozen=True)
class MorphSet:
    """In-memory MorphSet.

    Attributes
    ----------
    meta
        One row per face: file name, source dataset, subject (photo series),
        person, emotion label, agreement and whether the face is a mirrored copy.
    landmarks
        ``(n, 68, 2)`` float64 dlib landmarks in image coordinates.
    targets
        ``(n, 3)`` float64 ``(arousal, valence, intensity)``.
    """

    meta: pd.DataFrame
    landmarks: NDArray[np.float64]
    targets: NDArray[np.float64]

    target_names: tuple[str, ...] = TARGETS

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def subjects(self) -> NDArray[np.int64]:
        """Photo-series ids as provided in the CSV (not unique per person)."""
        return self.meta["subject"].to_numpy()

    @property
    def persons(self) -> NDArray[np.int64]:
        """Person ids: use these as groups for splitting and cross-validation."""
        return self.meta["person"].to_numpy()

    def subset(self, mask: NDArray[np.bool_]) -> MorphSet:
        return MorphSet(
            self.meta.loc[mask].reset_index(drop=True), self.landmarks[mask], self.targets[mask]
        )


def load_morphset(path: str | Path | None = None) -> MorphSet:
    """Load MorphSet from the zipped CSV (downloaded on first use)."""
    zpath = fetch_morphset(path)
    with zipfile.ZipFile(zpath) as zf, zf.open("Morphset.csv") as fh:
        df = pd.read_csv(fh, engine="pyarrow")

    meta = df[META_COLUMNS].rename(
        columns={
            "Filename": "filename",
            "Dataset": "dataset",
            "Subject": "subject",
            "Emotion Label": "label",
            "Agreement": "agreement",
        }
    )
    meta["subject"] = meta["subject"].astype(np.int64)
    meta["person"] = assign_person_ids(meta)
    meta["mirrored"] = meta["filename"].str.endswith("_2.jpg")
    meta["dataset"] = meta["dataset"].astype("category")
    meta["label"] = meta["label"].astype("category")

    coords = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float64)
    landmarks = coords.reshape(-1, 2, N_LANDMARKS).transpose(0, 2, 1).copy()
    targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return MorphSet(meta=meta, landmarks=landmarks, targets=targets)


def split_by_group(
    groups: NDArray[np.integer],
    *,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 1,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Split rows into train/validation/test so that no group spans two splits.

    MorphSet has many faces per person; splitting by row would leak identity
    information between splits and inflate scores. Pass person ids as groups.

    Returns boolean row masks ``(train, val, test)``.
    """
    if not 0 <= val_size < 1 or not 0 <= test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in [0, 1) and sum to < 1")
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = round(len(unique) * (1 - val_size - test_size))
    n_val = round(len(unique) * val_size)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    return (
        np.isin(groups, train_ids),
        np.isin(groups, val_ids),
        np.isin(groups, test_ids),
    )
