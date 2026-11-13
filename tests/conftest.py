from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
IMAGES = Path(__file__).parents[1] / "data" / "images"


@pytest.fixture(scope="session")
def legacy_reference() -> dict[str, np.ndarray]:
    """Outputs of the original 2021 implementation on 32 MorphSet faces.

    Generated with the v1.0.0 code in its original environment (Python 3.6,
    scikit-learn 0.23.1). Keys: ``landmarks`` (32, 68, 2) raw input,
    ``frontal`` (32, 68, 2), ``raw_predictions`` (32, 3) and ``labels`` (32,).
    """
    with np.load(DATA / "legacy_reference.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def legacy_weights() -> dict[str, np.ndarray]:
    """Affine weights extracted from the 2021 PLS(30) model (coef, intercept)."""
    with np.load(DATA / "legacy_pls30_linear.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def landmarks(legacy_reference: dict[str, np.ndarray]) -> np.ndarray:
    return legacy_reference["landmarks"]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def random_similarity(
    shape: np.ndarray, rng: np.random.Generator, max_angle: float = 0.6
) -> np.ndarray:
    """Apply a random rotation, scale and translation to a (68, 2) shape."""
    theta = rng.uniform(-max_angle, max_angle)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return shape @ rot.T * rng.uniform(0.3, 3.0) + rng.uniform(-200, 200, size=2)


def dlib_predictor_available() -> bool:
    try:
        from facial_expression_analysis.assets import DLIB_PREDICTOR, cache_dir

        return (cache_dir() / DLIB_PREDICTOR.name).is_file()
    except Exception:
        return False


requires_dlib_model = pytest.mark.skipif(
    not dlib_predictor_available(), reason="dlib model not cached (run `fea download`)"
)
