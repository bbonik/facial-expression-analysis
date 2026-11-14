"""What does the PLS model look at? And do its components resemble action units?

Fits PLS on all of MorphSet and reports:
* the strongest standardised distance weights for arousal / valence / intensity,
  drawn on the mean frontal face;
* the share of |weight| per facial-region pair;
* the facial movement that accompanies an increase of each prediction, and the
  share of the output's sensitivity to landmark noise per region;
* the facial movement encoded by each PLS component (minimum-norm landmark
  displacement reproducing the component's loading pattern), with the share of
  movement per region and a concentration score (1 = a single region, like an
  action unit; 0 = uniform), compared against random feature directions.

Usage: python experiments/11_interpretability.py --components 31
Outputs: reports/figures/pls_weights.png, reports/figures/pls_target_movements.png,
         reports/figures/pls_components.png,
         outputs/11_region_shares.csv, outputs/11_component_profile.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.frontalization import Frontalizer
from facial_expression_analysis.interpret import (
    distance_jacobian,
    equivalent_movement,
    movement_region_shares,
    plot_movement,
    plot_pair_weights,
    pls_vip,
    region_shares,
    standardized_coefficients,
)

OUT, FIG = Path("outputs"), Path("reports/figures")
TARGETS = ("arousal", "valence", "intensity")


def main(k: int) -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ds = load_morphset()
    frontal = Frontalizer()(ds.landmarks)
    feats = PairwiseDistanceFeatures()
    X = feats(frontal)
    pairs = feats.pairs
    pls = PLSRegression(k).fit(X, ds.targets)
    coef = np.asarray(pls.coef_).T if pls.coef_.shape[0] == 3 else np.asarray(pls.coef_)
    std_coef = standardized_coefficients(coef, X)
    mean_face = frontal[~ds.meta.mirrored.to_numpy() & (ds.meta.label == "NE").to_numpy()].mean(0)

    # --- weights on the face
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    shares = {}
    for ax, (t, name) in zip(axes, enumerate(TARGETS), strict=True):
        plot_pair_weights(ax, mean_face, pairs, std_coef[:, t], top=60, title=name)
        shares[name] = region_shares(std_coef[:, t], pairs)
    fig.suptitle(f"PLS({k}): 60 strongest distance weights (red: larger distance -> higher value)")
    fig.tight_layout()
    fig.savefig(FIG / "pls_weights.png", dpi=130)
    sh = pd.DataFrame(shares).fillna(0).sort_values("valence", ascending=False)
    sh.to_csv(OUT / "11_region_shares.csv")
    print("share of |standardised weight| by region pair:\n", sh.round(3).head(10).to_string())

    vip = pls_vip(pls)
    top = np.argsort(-vip)[:10]
    print(
        "\ntop-10 VIP distances:",
        [f"{a}-{b} ({vip[i]:.2f})" for i, (a, b) in zip(top, pairs[top], strict=True)],
    )

    # --- facial movements: what each target responds to, and what each component encodes
    x_std = X.std(axis=0, dtype=np.float64)
    J = distance_jacobian(mean_face, pairs)
    # Movement that accompanies an increase of each *prediction*: regress the frontal
    # landmarks on the model output (shape change per unit of predicted value).
    pred = pls.predict(X)
    real = ~ds.meta.mirrored.to_numpy()
    L = frontal[real].reshape(real.sum(), -1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (t, name) in zip(axes, enumerate(TARGETS), strict=True):
        yt = pred[real, t] - pred[real, t].mean()
        beta = (L - L.mean(0)).T @ yt / (yt @ yt)
        move = beta.reshape(68, 2)
        sh_ = movement_region_shares(move)
        plot_movement(
            ax,
            mean_face,
            move,
            title=f"{name}: "
            + ", ".join(
                f"{r} {v:.0%}" for r, v in sh_.sort_values(ascending=False).head(2).items()
            ),
        )
    fig.suptitle("Facial movement accompanying an increase of each prediction")
    fig.tight_layout()
    fig.savefig(FIG / "pls_target_movements.png", dpi=130)
    # Sensitivity: norm of the output gradient per landmark (where landmark noise hurts most)
    sens = pd.DataFrame(
        {
            name: movement_region_shares(
                np.stack([(J.T @ coef[:, t])[:68], (J.T @ coef[:, t])[68:]], axis=1)
            )
            for t, name in enumerate(TARGETS)
        }
    )
    print(
        "\nshare of output sensitivity to landmark displacement, by region:\n",
        sens.round(3).to_string(),
    )

    regions = ["brows", "eyes", "nose", "mouth"]

    def concentration(move):
        p = movement_region_shares(move)[regions].to_numpy()
        p = p / p.sum()
        h = -(p[p > 0] * np.log(p[p > 0])).sum()
        return 1 - h / np.log(len(regions))

    loadings = pls.x_loadings_ * x_std[:, None]  # component patterns in raw distance units
    moves = [equivalent_movement(loadings[:, c], mean_face, pairs) for c in range(k)]
    rows = []
    for c, mv in enumerate(moves):
        sh_ = movement_region_shares(mv)
        rows.append(
            {
                "component": c + 1,
                **sh_[regions].round(3).to_dict(),
                "dominant": sh_[regions].idxmax(),
                "concentration": concentration(mv),
            }
        )
    prof = pd.DataFrame(rows).set_index("component")
    rng = np.random.default_rng(0)
    null = np.array(
        [
            concentration(
                equivalent_movement(rng.normal(size=len(pairs)) * x_std, mean_face, pairs)
            )
            for _ in range(200)
        ]
    )
    q95 = np.quantile(null, 0.95)
    prof.to_csv(OUT / "11_component_profile.csv")
    print(
        f"\ncomponent movement concentration: median {prof.concentration.median():.3f}; "
        f"random-direction null median {np.median(null):.3f}, 95th pct {q95:.3f}; "
        f"components above the null 95th pct: {(prof.concentration > q95).sum()}/{k}"
    )
    print(prof.head(12).round(3).to_string())

    n_show = 8
    fig, axes = plt.subplots(2, n_show // 2, figsize=(16, 8.5))
    for c, ax in enumerate(axes.ravel()):
        plot_movement(
            ax,
            mean_face,
            moves[c],
            title=f"component {c + 1}: {prof.dominant.iloc[c]} "
            f"(conc. {prof.concentration.iloc[c]:.2f})",
        )
    fig.suptitle(f"Facial movement encoded by the first {n_show} PLS components")
    fig.tight_layout()
    fig.savefig(FIG / "pls_components.png", dpi=130)
    print(f"\nsaved figures to {FIG}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", type=int, default=31)
    main(ap.parse_args().components)
