"""Nested grouped CV for linear models: PLS (n_components) and ridge (alpha).

Outer loop: 5-fold x 2 repeats grouped by person -> unbiased performance.
Inner loop: 3-fold grouped CV on each outer training set -> hyper-parameter,
chosen with the one-standard-error rule (the most regularised model whose
inner score is within one SE of the best).

Also compares three ways of producing intensity on the outer predictions:
  regressor  - the model's own intensity output
  hypot      - sqrt(arousal^2 + valence^2)   (labels satisfy I = hypot(A, V) exactly)
  rescaled   - (A, V) rescaled so that its norm equals the regressor intensity

Outputs: outputs/06_linear_scores.csv, outputs/06_inner_curves.csv, outputs/06_choices.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.evaluation import (
    pls_path,
    repeated_group_kfold,
    ridge_path,
    score_rows,
)
from facial_expression_analysis.metrics import ccc
from facial_expression_analysis.training import compute_features

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
MAX_K = 60
ALPHAS = np.logspace(-7, -1, 25)
TARGETS = ("arousal", "valence", "intensity")


def inner_curves(X, y, groups, fit_path, seed):
    """Mean and SE (over inner folds) of the mean-over-targets CCC along a path."""
    per_fold = []
    for f in repeated_group_kfold(groups, n_splits=3, seed=seed):
        pred = fit_path(X[f.train], y[f.train]).predict(X[f.test])  # (P, n, 3)
        yt = y[f.test]
        per_fold.append([np.mean([ccc(yt[:, t], p[:, t]) for t in range(3)]) for p in pred])
    a = np.asarray(per_fold)
    return a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))


def one_se(mean, se, params, more_regularised_first):
    best = np.argmax(mean)
    ok = np.flatnonzero(mean >= mean[best] - se[best])
    # PLS: fewer components = more regularised; ridge: larger alpha = more regularised
    return params[ok.min()] if more_regularised_first else params[ok.max()], params[best]


def intensity_variants(raw):
    a, v = np.clip(raw[:, 0], -1, 1), np.clip(raw[:, 1], -1, 1)
    i_reg = np.clip(raw[:, 2], 0, 1)
    norm = np.hypot(a, v)
    scale = np.divide(i_reg, norm, out=np.zeros_like(norm), where=norm > 1e-9)
    return {
        "regressor": np.column_stack([a, v, i_reg]),
        "hypot": np.column_stack([a, v, np.clip(norm, 0, 1)]),
        "rescaled": np.column_stack([a * scale, v * scale, i_reg]),
    }


def main():
    ds = load_morphset()
    X = compute_features(ds.landmarks)  # default: 51 non-jaw landmarks, stable scale
    y, groups = ds.targets, ds.persons
    scores, curves, choices = [], [], []
    for f in repeated_group_kfold(groups, n_splits=5, n_repeats=2, seed=0):
        t0 = time.time()
        Xtr, ytr, gtr = X[f.train], y[f.train], groups[f.train]
        families = {
            "pls": (lambda A, b: pls_path(A, b, MAX_K), np.arange(1, MAX_K + 1), True),
            "ridge": (lambda A, b: ridge_path(A, b, ALPHAS), ALPHAS, False),
        }
        for name, (fit, params, reg_first) in families.items():
            mean, se = inner_curves(Xtr, ytr, gtr, fit, seed=100 + 10 * f.repeat + f.fold)
            curves += [
                {
                    "model": name,
                    "repeat": f.repeat,
                    "fold": f.fold,
                    "param": p,
                    "inner_ccc": m,
                    "inner_se": s,
                }
                for p, m, s in zip(params, mean, se, strict=True)
            ]
            chosen, best = one_se(mean, se, params, reg_first)
            choices.append(
                {"model": name, "repeat": f.repeat, "fold": f.fold, "one_se": chosen, "best": best}
            )
            path = fit(Xtr, ytr)
            idx = int(np.flatnonzero(params == chosen)[0])
            raw = path.predict(X[f.test])[idx]
            for variant, pred in intensity_variants(raw).items():
                scores += score_rows(
                    y[f.test],
                    pred,
                    TARGETS,
                    model=f"{name}/{variant}",
                    repeat=f.repeat,
                    fold=f.fold,
                )
            # Fixed 2021 configuration for reference (PLS, 31 components)
            if name == "pls":
                for variant, pred in intensity_variants(path.predict(X[f.test])[30]).items():
                    scores += score_rows(
                        y[f.test],
                        pred,
                        TARGETS,
                        model=f"pls31/{variant}",
                        repeat=f.repeat,
                        fold=f.fold,
                    )
        print(
            f"repeat {f.repeat} fold {f.fold}: {time.time() - t0:.0f}s  "
            f"choices={[(c['model'], c['one_se'], c['best']) for c in choices[-2:]]}",
            flush=True,
        )
        pd.DataFrame(scores).to_csv(OUT / "06_linear_scores.csv", index=False)
        pd.DataFrame(curves).to_csv(OUT / "06_inner_curves.csv", index=False)
        pd.DataFrame(choices).to_csv(OUT / "06_choices.csv", index=False)

    s = pd.DataFrame(scores)
    print(s.groupby(["model", "target"]).ccc.agg(["mean", "std"]).unstack().round(4).to_string())


if __name__ == "__main__":
    main()
