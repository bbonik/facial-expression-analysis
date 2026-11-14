"""Compare regressors on five axes: accuracy, generalisation, jitter, pose, external validity.

1. In-distribution: 5-fold CV grouped by person (paired folds across models).
2. Generalisation: leave-one-database-out (RaFD / KDEF / WSEFEP / FACES).
3. Jitter: Gaussian noise (sigma = 1 px per coordinate, matching dlib's measured
   repeatability: 1.8 px mean residual between a face and its re-detected mirror)
   is added to the raw landmarks of the test faces; we report mean |delta| of the
   arousal/valence outputs between clean and noisy inputs (5 noise draws).
4. Pose invariance: on the multi-pose dataset (Multi-PIE and CAS-PEAL persons only,
   since KDEF persons are also in MorphSet), mean |delta| between the prediction for
   a posed face and the frontal face of the same subject and expression.
5. External validity: on the same Multi-PIE / CAS-PEAL persons (real, not morphed,
   expressions, including extremes such as Multi-PIE's scream), the prediction for
   an expressive face minus that for the subject's neutral face at the same pose.
   Expected signs: happy -> valence up; disgust -> valence down; surprise, scream ->
   arousal up. Reported: share of correctly signed contrasts and mean contrast.

``--models a,b`` evaluates only those models and replaces their rows in the
existing output files (the hybrid ``pls31_kr`` was added after the first run,
when the kernel model failed on out-of-distribution faces).

PLS / ridge hyper-parameters come from experiment 06 (median one-SE choice).
Non-linear models get light tuning on one inner split of fold 0's training set.

Outputs: outputs/08_scores.csv, 08_jitter.csv, 08_pose.csv, 08_external.csv, 08_tuning.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.evaluation import repeated_group_kfold, score_rows
from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.frontalization import Frontalizer
from facial_expression_analysis.frontalization_training import load_pose_dataset
from facial_expression_analysis.metrics import ccc

OUT = Path("outputs")
TARGETS = ("arousal", "valence", "intensity")
JITTER_SIGMA = 1.0
N_JITTER = 5
RNG = np.random.default_rng(0)

FRONT = Frontalizer()
FEATS = PairwiseDistanceFeatures()


def features(landmarks):
    return FEATS(FRONT(landmarks))


class Float64Output(RegressorMixin, BaseEstimator):
    """Wrap a regressor so that predictions are float64 arrays."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est_ = clone(self.est).fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.est_.predict(X), dtype=np.float64)


class ResidualKernel(RegressorMixin, BaseEstimator):
    """Linear model plus a kernel model of its residuals.

    Far from the training data the kernel term decays to ~0, so predictions fall
    back to the linear model instead of collapsing to the mean.
    """

    def __init__(self, linear, kernel):
        self.linear = linear
        self.kernel = kernel

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.linear_ = clone(self.linear).fit(X, y)
        self.kernel_ = clone(self.kernel).fit(X, y - self.linear_.predict(X))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.linear_.predict(X) + self.kernel_.predict(X)


def candidate_models(pls_k: int, ridge_alpha: float, n_train: int):
    std = StandardScaler
    hgb = lambda leaves: MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.1,
            max_leaf_nodes=leaves,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=0,
        )
    )
    return {
        "pls": [PLSRegression(pls_k)],
        "pls31": [PLSRegression(31)],  # the 2021 choice (generalisation-driven)
        "ridge": [make_pipeline(std(), Ridge(alpha=ridge_alpha * n_train))],
        "pca_ridge": [
            make_pipeline(std(), PCA(n, random_state=0), Ridge(alpha=1.0)) for n in (30, 100)
        ],
        "kernel_ridge": [
            make_pipeline(
                std(), Nystroem(gamma=g / 1275, n_components=2000, random_state=0), Ridge(alpha=a)
            )
            for g in (0.5, 1.0, 2.0)
            for a in (1e-2, 1.0)
        ],
        "pls31_kr": [
            ResidualKernel(
                PLSRegression(31),
                make_pipeline(
                    std(),
                    Nystroem(gamma=g / 1275, n_components=2000, random_state=0),
                    Ridge(alpha=a),
                ),
            )
            for g in (0.25, 0.5, 1.0)
            for a in (1.0, 10.0)
        ],
        "hgb": [hgb(31), hgb(63)],
        "mlp": [
            make_pipeline(
                std(),
                MLPRegressor(
                    hidden_layer_sizes=(256, 128),
                    alpha=a,
                    early_stopping=True,
                    max_iter=200,
                    random_state=0,
                ),
            )
            for a in (1e-4, 1e-2)
        ],
    }


def mean_ccc(y, p):
    return float(np.mean([ccc(y[:, k], p[:, k]) for k in range(y.shape[1])]))


def tune(models, X, y, groups):
    """Pick the best configuration per family on one grouped inner split."""
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, groups))
    chosen, log = {}, []
    for name, configs in models.items():
        best, best_score = configs[0], -np.inf
        if len(configs) > 1:
            for k, est in enumerate(configs):
                t0 = time.time()
                score = mean_ccc(y[va], Float64Output(est).fit(X[tr], y[tr]).predict(X[va]))
                log.append(
                    {
                        "model": name,
                        "config": k,
                        "params": str(est),
                        "inner_ccc": score,
                        "seconds": time.time() - t0,
                    }
                )
                print(f"  tune {name}[{k}] ccc={score:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if score > best_score:
                    best, best_score = est, score
        chosen[name] = best
    return chosen, log


def jitter_rows(model, landmarks_test, X_clean_pred, tags):
    deltas = []
    for _ in range(N_JITTER):
        noisy = landmarks_test + RNG.normal(scale=JITTER_SIGMA, size=landmarks_test.shape)
        deltas.append(np.abs(model.predict(features(noisy))[:, :2] - X_clean_pred[:, :2]).mean(0))
    d = np.mean(deltas, axis=0)
    return {**tags, "jitter_dA": d[0], "jitter_dV": d[1]}


def pose_rows(model, pose, tags):
    meta = pose.meta
    frontal = (meta.yaw == 0) & (meta.pitch == 0)
    key = meta.subject.astype(str) + "|" + meta.expression
    ref = key.map(meta.index.to_series()[frontal].groupby(key[frontal]).first())
    ok = ref.notna() & ~frontal & (meta.pitch == 0) & meta.yaw.abs().isin([15, 22, 30, 45])
    ok &= meta.family != "kdef"
    posed, refs = np.flatnonzero(ok), ref[ok].to_numpy(dtype=int)
    p_pose = np.clip(model.predict(features(pose.landmarks[posed]))[:, :2], -1, 1)
    p_ref = np.clip(model.predict(features(pose.landmarks[refs]))[:, :2], -1, 1)
    d = np.abs(p_pose - p_ref)
    yaw = meta.yaw.abs().to_numpy()[posed]
    return [
        {**tags, "abs_yaw": y, "pose_dA": d[yaw == y, 0].mean(), "pose_dV": d[yaw == y, 1].mean()}
        for y in np.unique(yaw)
    ]


EXPECTED = {"HA": (1, +1), "DI": (1, -1), "SU": (0, +1), "SC": (0, +1)}  # (target, sign)


def external_rows(model, pose, tags):
    meta = pose.meta
    ok = (meta.family != "kdef") & (meta.pitch == 0) & (meta.yaw.abs() <= 45)
    key = meta.subject.astype(str) + "|" + meta.yaw.astype(str)
    neutral = ok & (meta.expression == "NE")
    ref = key.map(meta.index.to_series()[neutral].groupby(key[neutral]).first())
    rows = []
    for expr, (t, sign) in EXPECTED.items():
        sel = ok & (meta.expression == expr) & ref.notna()
        idx, refs = np.flatnonzero(sel), ref[sel].to_numpy(dtype=int)
        p = np.clip(model.predict(features(pose.landmarks[idx]))[:, t], -1, 1)
        p0 = np.clip(model.predict(features(pose.landmarks[refs]))[:, t], -1, 1)
        delta = sign * (p - p0)
        rows.append(
            {**tags, "expression": expr, "target": TARGETS[t], "n": len(idx),
             "sign_acc": float((delta > 0).mean()), "mean_contrast": float(delta.mean())}
        )  # fmt: skip
    return rows


def _save(rows, name, only):
    path = OUT / name
    new = pd.DataFrame(rows)
    if only and path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old.model.isin(only)], new], ignore_index=True)
    new.to_csv(path, index=False)


def main(only: list[str] | None = None):
    ds = load_morphset()
    pose = load_pose_dataset()
    X = features(ds.landmarks)
    y, persons = ds.targets, ds.persons
    choices = pd.read_csv(OUT / "06_choices.csv")
    pls_k = int(choices[choices.model == "pls"].one_se.median())
    ridge_alpha = float(choices[choices.model == "ridge"].one_se.median())
    print(f"PLS k={pls_k}, ridge alpha={ridge_alpha:g}", flush=True)

    folds = repeated_group_kfold(persons, n_splits=5, n_repeats=1, seed=0)
    f0 = folds[0]
    candidates = candidate_models(pls_k, ridge_alpha, len(f0.train))
    if only:
        candidates = {k: v for k, v in candidates.items() if k in only}
    models, tuning = tune(
        candidates,
        X[f0.train],
        y[f0.train],
        persons[f0.train],
    )
    _save(tuning, "08_tuning.csv", only)

    schemes = [("cv", f.fold, f.train, f.test) for f in folds]
    datasets = ds.meta.dataset.astype(str).to_numpy()
    schemes += [
        ("lodo", d, np.flatnonzero(datasets != d), np.flatnonzero(datasets == d))
        for d in ("R", "K", "W", "F")
    ]

    scores, jitter, pose_inv, external = [], [], [], []
    for scheme, fold, tr, te in schemes:
        for name, est in models.items():
            t0 = time.time()
            if name == "ridge":  # alpha is per sample
                est = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha * len(tr)))
            model = Float64Output(est).fit(X[tr], y[tr])
            pred = model.predict(X[te])
            tags = {"scheme": scheme, "fold": fold, "repeat": 0, "model": name}
            scores += score_rows(y[te], pred, TARGETS, **tags, fit_seconds=time.time() - t0)
            if scheme == "cv":
                sub = te[RNG.choice(len(te), size=min(5000, len(te)), replace=False)]
                jitter.append(jitter_rows(model, ds.landmarks[sub], model.predict(X[sub]), tags))
                if fold == 0:
                    pose_inv += pose_rows(model, pose, tags)
                    external += external_rows(model, pose, tags)
            print(
                f"{scheme} {fold} {name:<13} "
                f"ccc A={ccc(y[te, 0], pred[:, 0]):.3f} V={ccc(y[te, 1], pred[:, 1]):.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    _save(scores, "08_scores.csv", only)
    _save(jitter, "08_jitter.csv", only)
    _save(pose_inv, "08_pose.csv", only)
    _save(external, "08_external.csv", only)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated subset of models (default: all)")
    args = ap.parse_args()
    main(args.models.split(",") if args.models else None)
