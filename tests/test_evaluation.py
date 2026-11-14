from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from facial_expression_analysis.evaluation import (
    bootstrap_ci,
    leave_one_group_out,
    paired_difference,
    pls_path,
    repeated_group_kfold,
    ridge_path,
    score_rows,
)


@pytest.fixture
def problem(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(400, 30)) * rng.uniform(0.5, 3, size=30) + rng.normal(size=30)
    W = rng.normal(size=(30, 3))
    y = X @ W + rng.normal(scale=2.0, size=(400, 3))
    return X, y


@pytest.mark.parametrize("k", [1, 4, 9])
def test_pls_path_matches_individual_fits(problem: tuple[np.ndarray, np.ndarray], k: int) -> None:
    from sklearn.cross_decomposition import PLSRegression

    X, y = problem
    path = pls_path(X, y, max_components=10)
    expected = PLSRegression(n_components=k).fit(X, y).predict(X)
    np.testing.assert_allclose(path.predict(X)[k - 1], expected, atol=1e-8)


def test_ridge_path_matches_closed_form(problem: tuple[np.ndarray, np.ndarray]) -> None:
    from sklearn.linear_model import Ridge

    X, y = problem
    alphas = [1e-4, 1e-2, 1.0]
    path = ridge_path(X, y, alphas)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    for k, a in enumerate(alphas):
        expected = Ridge(alpha=a * len(X)).fit(Xs, y).predict(Xs)
        np.testing.assert_allclose(path.predict(X)[k], expected, atol=1e-8)


def test_repeated_group_kfold_respects_groups() -> None:
    groups = np.repeat(np.arange(30), 4)
    folds = repeated_group_kfold(groups, n_splits=5, n_repeats=2, seed=0)
    assert len(folds) == 10
    for f in folds:
        assert not set(groups[f.train]) & set(groups[f.test])
    for r in range(2):
        tests = np.concatenate([f.test for f in folds if f.repeat == r])
        assert sorted(tests) == list(range(len(groups)))
    first = {tuple(f.test) for f in folds if f.repeat == 0}
    second = {tuple(f.test) for f in folds if f.repeat == 1}
    assert first != second  # repeats reshuffle the groups


def test_leave_one_group_out() -> None:
    labels = np.array(["a", "b", "a", "c"])
    out = dict(leave_one_group_out(labels))
    assert set(out) == {"a", "b", "c"}
    np.testing.assert_array_equal(out["a"], [0, 2])


def test_paired_difference_detects_better_model(rng: np.random.Generator) -> None:
    rows = []
    for r in range(2):
        for f in range(5):
            base = rng.normal(0.6, 0.02)
            for model, delta in (("good", 0.05), ("base", 0.0)):
                rows.append(
                    {
                        "model": model,
                        "repeat": r,
                        "fold": f,
                        "target": "valence",
                        "ccc": base + delta,
                    }
                )
    res = paired_difference(pd.DataFrame(rows), "good", "base")
    row = res.iloc[0]
    assert row.mean_diff == pytest.approx(0.05)
    assert row.ci_low > 0
    assert row.win_rate == 1.0


def test_bootstrap_ci_contains_estimate(rng: np.random.Generator) -> None:
    groups = np.repeat(np.arange(40), 10)
    t = rng.normal(size=400)
    p = t + rng.normal(scale=0.5, size=400)
    est, lo, hi = bootstrap_ci(t, p, groups, n_boot=200)
    assert lo < est < hi


def test_score_rows() -> None:
    y = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    rows = score_rows(y, y, ["a", "b"], model="m")
    assert {r["target"] for r in rows} == {"a", "b"}
    assert all(r["ccc"] == pytest.approx(1.0) and r["model"] == "m" for r in rows)
