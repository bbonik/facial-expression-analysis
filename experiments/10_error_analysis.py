"""Error analysis of an emotion model from out-of-fold predictions.

5-fold CV grouped by person; every face is predicted by a model that never saw
its person. Breakdowns by expression class, source database, intensity level
and circumplex angle; calibration (slope of prediction on target, < 1 means
shrinkage towards the mean); person-level bootstrap CIs for the overall CCC.

Usage: python experiments/10_error_analysis.py --model kernel:0.5,1   (gamma*n_features, alpha)
       (or pls:31, ridge:2.5e-5)
Outputs: outputs/10_oof_<model>.parquet, outputs/10_breakdown_<model>.csv,
         reports/figures/error_analysis_<model>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.evaluation import (
    bootstrap_ci,
    pls_path,
    repeated_group_kfold,
    ridge_path,
)
from facial_expression_analysis.metrics import ccc
from facial_expression_analysis.training import compute_features

OUT, FIG = Path("outputs"), Path("reports/figures")
TARGETS = ("arousal", "valence", "intensity")


def fit_predict(kind, param, Xtr, ytr, Xte):
    if kind == "kernel":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        g, a = (float(v) for v in param.split(","))
        model = make_pipeline(
            StandardScaler(),
            Nystroem(gamma=g / Xtr.shape[1], n_components=2000, random_state=0),
            Ridge(alpha=a),
        ).fit(Xtr.astype(np.float64), ytr)
        return model.predict(Xte.astype(np.float64))
    if kind == "pls":
        return pls_path(Xtr, ytr, int(param)).predict(Xte)[-1]
    return ridge_path(Xtr, ytr, [float(param)]).predict(Xte)[0]


def main(spec: str) -> None:
    kind, param = spec.split(":")
    tag = spec.replace(":", "").replace(",", "_")
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ds = load_morphset()
    X, y = compute_features(ds.landmarks), ds.targets
    oof = np.zeros_like(y)
    for f in repeated_group_kfold(ds.persons, n_splits=5, seed=0):
        oof[f.test] = fit_predict(kind, param, X[f.train], y[f.train], X[f.test])
    oof[:, :2] = np.clip(oof[:, :2], -1, 1)
    oof[:, 2] = np.clip(oof[:, 2], 0, 1)

    df = ds.meta[["dataset", "label", "person", "mirrored"]].copy()
    for k, t in enumerate(TARGETS):
        df[t], df[f"{t}_pred"] = y[:, k], oof[:, k]
    df["angle"] = np.degrees(np.arctan2(y[:, 0], y[:, 1])) % 360
    df.to_parquet(OUT / f"10_oof_{tag}.parquet")

    print(f"=== {spec}: overall CCC with person-bootstrap 95% CI")
    for k, t in enumerate(TARGETS):
        est, lo, hi = bootstrap_ci(y[:, k], oof[:, k], ds.persons, n_boot=500)
        slope = np.polyfit(y[:, k], oof[:, k], 1)[0]
        print(f"  {t:<10} CCC={est:.4f} [{lo:.4f}, {hi:.4f}]  calibration slope={slope:.3f}")

    rows = []

    def add(group_name, groups):
        for g, sub in df.groupby(groups, observed=True):
            row = {"by": group_name, "group": str(g), "n": len(sub)}
            for t in TARGETS:
                err = sub[f"{t}_pred"] - sub[t]
                row[f"{t}_mae"] = err.abs().mean()
                row[f"{t}_bias"] = err.mean()
            rows.append(row)

    add("label", "label")
    add("dataset", "dataset")
    df["intensity_bin"] = pd.cut(df.intensity, [-0.01, 0.1, 0.3, 0.5, 0.7, 1.0])
    add("intensity", "intensity_bin")
    df["angle_bin"] = pd.cut(df.angle.where(df.intensity > 0.1), np.arange(0, 241, 30))
    add("angle", "angle_bin")
    add("mirrored", "mirrored")
    br = pd.DataFrame(rows)
    br.to_csv(OUT / f"10_breakdown_{tag}.csv", index=False)
    with pd.option_context("display.width", 200):
        for by, g in br.groupby("by", sort=False):
            print(f"\n--- by {by}")
            print(g.drop(columns="by").set_index("group").round(3).to_string())

    # Edge of the label space: share of predictions outside the trained angle range
    pa = np.degrees(np.arctan2(oof[:, 0], oof[:, 1])) % 360
    strong = np.hypot(oof[:, 0], oof[:, 1]) > 0.1
    outside = strong & ((pa > 215) & (pa < 355))
    print(
        f"\npredictions with angle in the unseen 215-355 deg range: {outside.mean():.2%} "
        f"(of faces whose true label is sad-like (SA*): "
        f"{outside[df.label.astype(str).str.contains('SA').to_numpy()].mean():.2%})"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (k, t) in zip(axes, enumerate(TARGETS), strict=True):
        h = ax.hexbin(y[:, k], oof[:, k], gridsize=45, bins="log", cmap="viridis", mincnt=1)
        lim = (-1, 1) if t != "intensity" else (0, 1)
        ax.plot(lim, lim, "w--", lw=1)
        ax.set(
            xlim=lim,
            ylim=lim,
            xlabel=f"true {t}",
            ylabel=f"predicted {t}",
            title=f"{t}: CCC={ccc(y[:, k], oof[:, k]):.3f}",
        )
        fig.colorbar(h, ax=ax, shrink=0.8)
    fig.suptitle(f"Out-of-fold predictions ({spec}), 5-fold CV grouped by person")
    fig.tight_layout()
    fig.savefig(FIG / f"error_analysis_{tag}.png", dpi=130)
    print(f"\nsaved {FIG / f'error_analysis_{tag}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="kernel:<g>,<a>, pls:<k> or ridge:<alpha>")
    main(ap.parse_args().model)
