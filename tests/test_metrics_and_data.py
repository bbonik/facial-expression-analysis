from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from facial_expression_analysis.data import assign_person_ids, split_by_group
from facial_expression_analysis.metrics import ccc, regression_report


def test_ccc_perfect_and_biased(rng: np.random.Generator) -> None:
    t = rng.normal(size=1000)
    t -= t.mean()  # so that -t has the same mean and CCC(t, -t) is exactly -1
    assert ccc(t, t) == pytest.approx(1.0)
    # A constant offset keeps Pearson r = 1 but must lower the CCC.
    assert ccc(t, t + 1.0) < 0.7
    assert ccc(t, -t) == pytest.approx(-1.0)


def test_ccc_constant_input_is_nan() -> None:
    assert np.isnan(ccc(np.zeros(5), np.zeros(5)))


def test_regression_report_keys(rng: np.random.Generator) -> None:
    y = rng.normal(size=(100, 2))
    report = regression_report(y, y + 0.1, ["a", "b"])
    assert set(report) == {"a", "b"}
    assert report["a"]["mae"] == pytest.approx(0.1)
    assert report["a"]["rmse"] == pytest.approx(0.1)


def test_subject_split_is_disjoint_and_complete() -> None:
    subjects = np.repeat(np.arange(1, 101), 7)
    train, val, test = split_by_group(subjects, val_size=0.2, test_size=0.1, seed=0)
    assert np.all(train.astype(int) + val + test == 1)
    sets = [set(subjects[m]) for m in (train, val, test)]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert [len(s) for s in sets] == [70, 20, 10]
    # Regression: the 2021 split silently dropped the highest subject id.
    assert 100 in set().union(*sets)


def test_subject_split_is_reproducible() -> None:
    subjects = np.repeat(np.arange(50), 3)
    a = split_by_group(subjects, seed=3)
    b = split_by_group(subjects, seed=3)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_subject_split_validates_sizes() -> None:
    with pytest.raises(ValueError, match="sum"):
        split_by_group(np.arange(10), val_size=0.6, test_size=0.5)


def test_assign_person_ids_groups_photo_series() -> None:
    meta = pd.DataFrame(
        {
            "dataset": ["R", "R", "R", "R", "K", "K", "K", "W", "W", "F", "F", "F"],
            "subject": [1, 2, 3, 4, 202, 203, 204, 342, 343, 371, 372, 373],
        }
    )
    person = assign_person_ids(meta)
    # R: triplets; K, F: pairs; W: one id per person. Ids are unique across datasets.
    assert person.tolist() == [0, 0, 0, 1, 2, 2, 3, 4, 5, 6, 6, 7]


def test_assign_person_ids_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        assign_person_ids(pd.DataFrame({"dataset": ["X"], "subject": [1]}))
