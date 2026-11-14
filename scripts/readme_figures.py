"""Draw the pipeline figure of the README (docs/figures/pipeline.png).

Panels: (a) dlib landmarks on a photo, (b) frontalization of a face turned by 45 degrees,
(c) the pairwise-distance features with the strongest valence weights of the default model,
(d) the estimate on Russell's circumplex.

Usage: uv run python scripts/readme_figures.py   (needs matplotlib and the multi-pose dataset)
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from facial_expression_analysis import (
    DlibLandmarkDetector,
    EmotionEstimator,
    Frontalizer,
    load_image,
    load_model,
    procrustes_normalize,
)
from facial_expression_analysis.labels import _ANGLE_BOUNDS, _EMOTION_NAMES
from facial_expression_analysis.training.frontalization import load_pose_dataset
from facial_expression_analysis.viz import plot_landmarks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "pipeline.png"
VIOLET, TEAL, ROSE, GREY = "#7c3aed", "#0d9488", "#e11d48", "#9ca3af"


def panel_photo(ax: plt.Axes) -> tuple[np.ndarray, float, float]:
    img = load_image(ROOT / "examples" / "happy.jpg")[..., :3]
    (face,) = DlibLandmarkDetector().detect(img)
    b = face.box
    pad = int(0.35 * b.width)
    y0, x0 = max(b.top - pad, 0), max(b.left - pad, 0)
    crop = img[y0 : b.bottom + pad, x0 : b.right + pad]
    lm = face.landmarks - [x0, y0]
    ax.imshow(crop)
    ax.add_patch(plt.Rectangle((b.left - x0, b.top - y0), b.width, b.height,
                               fill=False, ec=TEAL, lw=1.5))  # fmt: skip
    ax.scatter(lm[:, 0], lm[:, 1], s=14, c=VIOLET, edgecolors="white", linewidths=0.3)
    ax.set_title("(a) 68 dlib landmarks", fontsize=10)
    ax.axis("off")
    e = EmotionEstimator().estimate(face.landmarks)
    return face.landmarks, e.arousal, e.valence


def panel_frontalization(ax: plt.Axes) -> None:
    data = load_pose_dataset()
    m = data.meta
    frontalizer = Frontalizer()
    # a Multi-PIE-like subject with a 45 degree view and a frontal view of the same smile
    for subject in m.loc[(m.family == "multipie") & (m.yaw == 45), "subject"].unique():
        sel = (m.subject == subject) & (m.expression == "HA") & (m.pitch == 0)
        side = np.flatnonzero(sel & (m.yaw == 45))
        front = np.flatnonzero(sel & (m.yaw == 0))
        if len(side) and len(front):
            break
    posed = procrustes_normalize(data.landmarks[side[0]])
    frontal = procrustes_normalize(data.landmarks[front[0]])
    pred = frontalizer(data.landmarks[side[0]])
    plot_landmarks(posed, ax, color=GREY)
    plot_landmarks(frontal, ax, color=TEAL)
    plot_landmarks(pred, ax, color=VIOLET)
    ax.set_title("(b) frontalization", fontsize=10)
    ax.legend(
        handles=[plt.Line2D([], [], color=c, lw=2) for c in (GREY, VIOLET, TEAL)],
        labels=["input, 45° yaw", "frontalized", "true frontal"],
        loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=7, frameon=False, ncol=3,
    )  # fmt: skip


def panel_features(ax: plt.Axes, landmarks: np.ndarray) -> None:
    model = load_model()
    frontal = Frontalizer()(landmarks)
    pairs = model.features.pairs
    data = load_pose_dataset()
    frontal_all = Frontalizer()(data.landmarks[data.meta.yaw.to_numpy() == 0])
    std = model.features.transform(frontal_all).std(axis=0)
    w = model.coef[:, 1] * std  # standardised valence weights
    segs = frontal[pairs]
    ax.add_collection(LineCollection(segs, colors="#d1d5db", linewidths=0.15, alpha=0.6))
    top = np.argsort(-np.abs(w))[:24]
    colors = [TEAL if w[k] > 0 else ROSE for k in top]
    ax.add_collection(LineCollection(segs[top], colors=colors, linewidths=1.6))
    plot_landmarks(frontal, ax, color="#374151")
    ax.scatter(frontal[17:, 0], frontal[17:, 1], s=4, c="#374151")
    ax.set_title("(c) 1,275 distances, top valence weights", fontsize=10)
    ax.legend(
        handles=[plt.Line2D([], [], color=c, lw=2) for c in (TEAL, ROSE)],
        labels=["longer → more positive", "longer → more negative"],
        loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=7, frameon=False, ncol=2,
    )  # fmt: skip


def panel_circumplex(ax: plt.Axes, arousal: float, valence: float) -> None:
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GREY, lw=1)
    ax.axhline(0, color="#e5e7eb", lw=0.8, zorder=0)
    ax.axvline(0, color="#e5e7eb", lw=0.8, zorder=0)
    bounds = (0, *_ANGLE_BOUNDS)
    for k, (a0, a1) in enumerate(pairwise(bounds)):
        mid = np.radians((a0 + a1) / 2)
        c, s_ = np.cos(np.radians(a1)), np.sin(np.radians(a1))
        ax.plot([0.97 * c, c], [0.97 * s_, s_], color=GREY, lw=0.6)
        deg = np.degrees(mid)
        ax.text(1.17 * np.cos(mid), 1.17 * np.sin(mid), _EMOTION_NAMES[k], fontsize=5.5,
                ha="center", va="center", rotation=deg - 180 if 90 < deg < 270 else deg,
                rotation_mode="anchor", color="#4b5563")  # fmt: skip
    ax.scatter([valence], [arousal], s=60, c=VIOLET, edgecolors="white", zorder=5)
    ax.text(1.42, 0, "valence", fontsize=7, ha="left", va="center", color="#6b7280")
    ax.text(0, 1.42, "arousal", fontsize=7, ha="center", va="bottom", color="#6b7280")
    ax.set_xlim(-1.55, 1.75)
    ax.set_ylim(-1.55, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"(d) A {arousal:+.2f}, V {valence:+.2f} on the circumplex", fontsize=10)


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    landmarks, a, v = panel_photo(axes[0])
    panel_frontalization(axes[1])
    panel_features(axes[2], landmarks)
    panel_circumplex(axes[3], a, v)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
