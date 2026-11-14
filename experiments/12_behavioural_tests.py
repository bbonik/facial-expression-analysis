"""Behavioural tests: does a model behave sensibly on faces of unseen people?

Offline accuracy (CCC on MorphSet) rewards fitting synthetic morph labels. In the live demo
other properties matter, and they favoured PLS(31) over the more accurate kernel model
(observation of the author, confirmed here). On Multi-PIE and CAS-PEAL persons (not in
MorphSet; KDEF excluded because its persons are in MorphSet):

A. Neutral calibration: frontal neutral faces should map to (A, V) = (0, 0).
B. Decomposition: a pure lip-corner pull (synthetic smile: mouth corners up and out, nothing
   else moves) applied to each neutral face should raise valence only. Reported: change
   per axis and the share of faces whose trajectory deviates > 20 deg from the valence axis.
C. Pose stability: std of the prediction across yaw (0 to 45 deg) for the same subject and
   expression.

Usage: python experiments/12_behavioural_tests.py
Outputs: outputs/12_behavioural.csv, reports/figures/behavioural_smile.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from facial_expression_analysis import Frontalizer
from facial_expression_analysis.frontalization_training import load_pose_dataset
from facial_expression_analysis.model import KernelEmotionModel, LinearEmotionModel

OUT, FIG = Path("outputs"), Path("reports/figures")
# Lip-corner pull: weight of each mouth landmark's displacement and its side (-1 left, +1 right)
SMILE = {48: (1.0, -1), 54: (1.0, 1), 60: (0.8, -1), 64: (0.8, 1),
         49: (0.4, -1), 59: (0.4, -1), 53: (0.4, 1), 55: (0.4, 1)}  # fmt: skip
OUT_FRAC, UP_FRAC = 0.07, 0.08  # maximum displacement, fraction of the inter-ocular distance


def synthetic_smile(landmarks: np.ndarray, strength: float) -> np.ndarray:
    out = landmarks.copy()
    iod = np.linalg.norm(landmarks[:, 36:42].mean(1) - landmarks[:, 42:48].mean(1), axis=1)
    for i, (w, side) in SMILE.items():
        out[:, i, 0] += side * w * strength * OUT_FRAC * iod
        out[:, i, 1] -= w * strength * UP_FRAC * iod  # image y points down
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pose = load_pose_dataset()
    meta = pose.meta.reset_index(drop=True)
    unseen = (meta.family != "kdef") & (meta.pitch == 0) & (meta.yaw.abs() <= 45)
    neutral0 = np.flatnonzero((unseen & (meta.expression == "NE") & (meta.yaw == 0)).to_numpy())
    front = Frontalizer()
    models = {"PLS(31)": LinearEmotionModel.load(), "kernel ridge": KernelEmotionModel.load()}

    rows, trajectories = [], {}
    steps = np.linspace(0, 1, 7)
    idx = np.flatnonzero(unseen.to_numpy())
    for name, model in models.items():
        # A. neutral calibration
        p0 = np.clip(model.predict(front(pose.landmarks[neutral0]))[:, :2], -1, 1)
        # B. synthetic smile trajectories
        traj = np.stack(
            [model.predict(front(synthetic_smile(pose.landmarks[neutral0], s)))[:, :2]
             for s in steps]
        )  # fmt: skip
        trajectories[name] = traj
        d = traj[-1] - traj[0]
        angle = np.degrees(np.arctan2(d[:, 0], d[:, 1]))
        # C. pose stability
        pred = np.clip(model.predict(front(pose.landmarks[idx]))[:, :2], -1, 1)
        sub = meta.iloc[idx].assign(A=pred[:, 0], V=pred[:, 1])
        spread = sub.groupby(["subject", "expression"])[["A", "V"]].std().dropna()
        rows.append(
            {
                "model": name,
                "neutral_A": p0[:, 0].mean(),
                "neutral_V": p0[:, 1].mean(),
                "neutral_distance": np.hypot(p0[:, 0], p0[:, 1]).mean(),
                "smile_dV": d[:, 1].mean(),
                "smile_dA": d[:, 0].mean(),
                "smile_off_axis_share": (np.abs(angle) > 20).mean(),
                "pose_std_A": spread.A.mean(),
                "pose_std_V": spread.V.mean(),
            }
        )
    res = pd.DataFrame(rows).set_index("model")
    res.to_csv(OUT / "12_behavioural.csv")
    print(f"{len(neutral0)} neutral frontal faces of unseen persons\n")
    print(res.round(3).T.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    show = rng.choice(len(neutral0), size=40, replace=False)
    for ax, (name, traj) in zip(axes, trajectories.items(), strict=True):
        for f in show:
            ax.plot(traj[:, f, 1], traj[:, f, 0], "-", lw=0.8, alpha=0.6)
            ax.plot(traj[0, f, 1], traj[0, f, 0], "k.", ms=3)
        ax.axhline(0, c="k", lw=0.5)
        ax.axvline(0, c="k", lw=0.5)
        r = res.loc[name]
        ax.set(
            xlim=(-1, 1), ylim=(-1, 1), xlabel="valence", aspect="equal",
            title=f"{name}\nneutral at distance {r.neutral_distance:.2f}; "
            f"{r.smile_off_axis_share:.0%} of paths > 20° off the valence axis",
        )  # fmt: skip
    axes[0].set_ylabel("arousal")
    fig.suptitle("Synthetic smile (mouth corners only) on neutral faces of 40 unseen people")
    fig.tight_layout()
    fig.savefig(FIG / "behavioural_smile.png", dpi=130)
    print(f"\nsaved {FIG / 'behavioural_smile.png'}")


if __name__ == "__main__":
    main()
