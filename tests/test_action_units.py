from __future__ import annotations

import numpy as np
import pytest

from facial_expression_analysis import Frontalizer
from facial_expression_analysis.constants import MIRROR_PERMUTATION
from facial_expression_analysis.training.action_units import ACTION_UNITS, action_unit_field


@pytest.fixture(scope="module")
def face(landmarks) -> np.ndarray:
    return Frontalizer()(landmarks).mean(0)


def test_fields_are_unit_norm_symmetric_and_jaw_free(face) -> None:
    perm = np.array(MIRROR_PERMUTATION)
    for name in ACTION_UNITS:
        f = action_unit_field(name, face)
        assert np.linalg.norm(f) == pytest.approx(1.0)
        assert not f[:17].any()  # no jaw landmarks: the features do not use them
        # mirror symmetry: the field on the right side is the mirrored field on the left
        np.testing.assert_allclose(f[perm, 0], -f[:, 0], atol=1e-12)
        np.testing.assert_allclose(f[perm, 1], f[:, 1], atol=1e-12)


def test_directions_match_the_facs_descriptions(face) -> None:
    up = lambda name, i: action_unit_field(name, face)[i, 1] < 0  # noqa: E731
    assert up("AU1 inner brow raiser", 21)
    assert not up("AU4 brow lowerer", 21)
    assert up("AU12 lip corner puller", 48)
    assert not up("AU15 lip corner depressor", 48)
    f = action_unit_field("AU20 lip stretcher", face)
    assert f[48, 0] < 0 < f[54, 0]  # both corners move away from the midline


def test_unknown_action_unit(face) -> None:
    with pytest.raises(KeyError):
        action_unit_field("AU99", face)
