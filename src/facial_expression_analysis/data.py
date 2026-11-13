"""Loading and splitting of the MorphSet dataset.

MorphSet (Vonikakis et al., ICIP 2021) contains dlib landmarks of face images
morphed between a neutral face and categorical expressions, with dimensional
labels (arousal, valence, intensity) derived from the morphing coefficients.
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


@dataclass(frozen=True)
class MorphSet:
    """In-memory MorphSet.

    Attributes
    ----------
    meta
        One row per face: file name, source dataset, subject, label, agreement.
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
        return self.meta["subject"].to_numpy()

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
    meta["dataset"] = meta["dataset"].astype("category")
    meta["label"] = meta["label"].astype("category")

    coords = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float64)
    landmarks = coords.reshape(-1, 2, N_LANDMARKS).transpose(0, 2, 1).copy()
    targets = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return MorphSet(meta=meta, landmarks=landmarks, targets=targets)


def split_subjects(
    subjects: NDArray[np.integer],
    *,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 1,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    """Split rows into train/validation/test so that no subject spans two splits.

    MorphSet has many expressions per subject; splitting by row would leak
    identity information between splits and inflate scores.

    Returns boolean row masks ``(train, val, test)``.
    """
    if not 0 <= val_size < 1 or not 0 <= test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be in [0, 1) and sum to < 1")
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = round(len(unique) * (1 - val_size - test_size))
    n_val = round(len(unique) * val_size)
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    return (
        np.isin(subjects, train_ids),
        np.isin(subjects, val_ids),
        np.isin(subjects, test_ids),
    )
