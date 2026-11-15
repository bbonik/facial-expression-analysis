"""How many dimensions does facial expression have, and why does PLS(31) behave best live?

Analyses behind ``paper/pls_action_units.md`` (original MorphSet images, no mirrored copies):

N. Landmark noise. MorphSet holds a horizontally mirrored copy of every image whose landmarks
   dlib detected independently. Undoing the mirror and subtracting gives the difference of two
   detections of the same face: their error covariance (in raw, frontalized and feature space)
   and a bank of realistic, structured noise vectors used everywhere below.
A. Intrinsic dimension of expression geometry: the spectrum of within-person deformations
   (expression minus the same person's neutral face, 51 non-jaw landmarks) whitened by the
   measured noise covariance (signal-to-noise ratio per direction), with and without
   frontalization; MorphSet (morphed) and Multi-PIE / KDEF / CAS-PEAL (real expressions).
B. PLS components 1-60: facial movement of each component, its best-matching action-unit
   template (and a random-direction null), localisation, identity share of its score variance,
   replicability across disjoint halves of the persons, and how much of 16 action-unit
   templates the span of the first k components captures (vs random directions).
C. Signal-to-noise ratio along each component, from the same noise measurement.
D. Accuracy vs live behaviour as complexity grows (PLS k = 1-60, ridge penalty path, kernel
   ridge): 5-fold person-grouped CV CCC, within-person ("delta") CCC, and on unseen people
   (Multi-PIE / CAS-PEAL) identity leakage, action-unit response heterogeneity, jitter under
   the measured landmark noise, and pose sensitivity; ridge effective degrees of freedom.
E. Accuracy at higher landmark noise (webcam-like): CV CCC when the test faces get extra
   measured noise, and the number of components that maximises it.

Usage: python experiments/14_pls_dimensionality.py        (about 12 minutes)
Outputs: outputs/14_*.csv, outputs/14_log.txt, paper/figures/*.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from facial_expression_analysis import Frontalizer
from facial_expression_analysis.constants import MIRROR_PERMUTATION
from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.geometry import procrustes_normalize
from facial_expression_analysis.training.action_units import ACTION_UNITS, action_unit_field
from facial_expression_analysis.training.data import load_morphset
from facial_expression_analysis.training.evaluation import (
    pls_path,
    repeated_group_kfold,
    ridge_path,
)
from facial_expression_analysis.training.frontalization import load_pose_dataset
from facial_expression_analysis.training.interpret import (
    REGIONS,
    equivalent_movement,
    movement_region_shares,
    plot_movement,
)
from facial_expression_analysis.training.metrics import ccc
from facial_expression_analysis.viz import plot_landmarks

OUT, FIG = Path("outputs"), Path("paper/figures")
K = 60  # PLS components examined
ALPHAS = np.logspace(-7, -1, 13)  # ridge penalties (per sample)
AU_STEP = 0.06  # size of a synthetic action unit, normalised units (~9 px at a 150 px IOD)
NOISE_LEVELS = (
    1.0,
    2.0,
    3.0,
)  # extra measured noise added to test faces in part E (x one detection)
NJ = np.arange(17, 68)  # the 51 non-jaw landmarks used by the features
LOG: list[str] = []

# ---------------------------------------------------------------------------- figure style
# ggplot-like theme; categorical colours validated for colour-vision deficiency
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6b7280"
COLORS = {"PLS": BLUE, "ridge": AQUA, "kernel ridge": ORANGE, "noise": GREY}
BAND = {"color": "#9ca3af", "alpha": 0.18, "lw": 0}  # components 20-35
SEQ = LinearSegmentedColormap.from_list("blues", ["#b7d3f6", "#2a78d6", "#0d366b"])
plt.style.use("ggplot")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f2f2f2", "axes.edgecolor": "none",
    "grid.color": "white", "grid.linewidth": 1.0, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "axes.labelcolor": "#333333", "xtick.color": "#555555",
    "ytick.color": "#555555", "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 1.8, "lines.markersize": 4, "savefig.dpi": 160,
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, AQUA, YELLOW]),
})  # fmt: skip


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def kernel_ridge(n_features: int):
    return make_pipeline(
        StandardScaler(),
        Nystroem(gamma=0.5 / n_features, n_components=2000, random_state=0),
        Ridge(alpha=1.0),
    )


def eye_distance(landmarks: np.ndarray) -> np.ndarray:
    """Distance between the eye centres, ``(n,)``."""
    return np.linalg.norm(landmarks[:, 36:42].mean(1) - landmarks[:, 42:48].mean(1), axis=1)


def add_noise(
    landmarks: np.ndarray, bank: np.ndarray, rng: np.random.Generator, scale: float = 1.0
) -> np.ndarray:
    """Add measured dlib noise: random rows of the twin noise bank, scaled to each face."""
    idx = rng.integers(len(bank), size=len(landmarks))
    return landmarks + scale * bank[idx] * eye_distance(landmarks)[:, None, None]


def whitened_snr(D: np.ndarray, noise_cov: np.ndarray, n_detections: int) -> np.ndarray:
    """SNR spectrum of a set of shapes or deformations against single-detection noise.

    The observed covariance is signal + ``n_detections`` x noise. In noise-whitened
    coordinates the generalised eigenvalues mu give SNR = mu - n_detections per direction.
    The spectrum is invariant to any invertible linear map of the coordinates.
    """
    w, V = np.linalg.eigh(noise_cov)
    keep = w > w.max() * 1e-6
    wh = V[:, keep] / np.sqrt(w[keep])
    C = np.cov(D - D.mean(0), rowvar=False)
    return np.linalg.eigvalsh(wh.T @ C @ wh)[::-1] - n_detections


# ============================================================================ data
def load() -> dict:
    data = load_morphset()
    m = data.meta
    orig = ~m.mirrored.to_numpy()
    meta = m[orig].reset_index(drop=True)
    front = Frontalizer()
    frontal = front(data.landmarks[orig])
    feat = PairwiseDistanceFeatures()
    X = feat(frontal).astype(np.float64)
    neutral = (meta.label == "NE").to_numpy()
    ne_rows = pd.Series(np.flatnonzero(neutral), index=meta.subject.to_numpy()[neutral])
    neutral_row = ne_rows[~ne_rows.index.duplicated()].reindex(meta.subject.to_numpy()).to_numpy()
    expr = ~neutral & ~np.isnan(neutral_row)
    return {
        "data": data, "meta": meta, "orig": orig, "front": front, "frontal": frontal,
        "feat": feat, "X": X, "Y": data.targets[orig], "persons": data.persons[orig],
        "neutral": neutral, "neutral_row": neutral_row, "expr": expr,
        "nrow": np.nan_to_num(neutral_row).astype(int),
        "apex": expr & (meta.filename.str.split("_").str[-2] == "10").to_numpy(),
        "mean_face": frontal[neutral].mean(0), "raw": data.landmarks[orig].astype(np.float64),
    }  # fmt: skip


# ============================================================================ N
def part_n(d: dict) -> dict:
    log("N. Landmark noise from independently re-detected mirror images")
    data, m = d["data"], d["data"].meta
    stem = m.filename.str.replace(r"_[12]\.jpg$", "", regex=True)
    a = pd.Series(m.index[~m.mirrored], index=stem[~m.mirrored])
    b = pd.Series(m.index[m.mirrored], index=stem[m.mirrored])
    pairs = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().astype(int)
    A = data.landmarks[pairs.a.to_numpy()].astype(np.float64)
    B = data.landmarks[pairs.b.to_numpy()].astype(np.float64)
    # undo the mirror: flip x and swap left/right landmark labels (the x offset is removed below)
    B = np.stack([-B[..., 0], B[..., 1]], -1)[:, np.array(MIRROR_PERMUTATION)]
    D = B - A
    D -= D.mean(1, keepdims=True)
    iod = eye_distance(A)
    rel = D / iod[:, None, None]
    bias = rel.mean(0)  # systematic left/right asymmetry of dlib, not noise
    bank = (rel - bias) / np.sqrt(2)  # error of ONE detection, in eye distances
    outer = np.linalg.norm(A[:, 36] - A[:, 45], axis=1)
    twin_jitter = 100 * float((np.linalg.norm(D, axis=-1).mean(1) / outer).mean())
    sd_px = bank.std(0) * 150
    log(f"  {len(A):,} twin pairs; eye distance in MorphSet images: median {np.median(iod):.1f} px")
    log(f"  one detection: noise sd {bank[:, NJ].std() * 150:.2f} px per coordinate (51 non-jaw "
        f"landmarks), {bank.std() * 150:.2f} px (all 68); systematic mirror bias rms "
        f"{np.sqrt((bias**2).mean()) * 150:.2f} px")  # fmt: skip
    regions = {"jaw": range(0, 17), "brows": range(17, 27), "nose": range(27, 36),
               "eyes": range(36, 48), "mouth": range(48, 68)}  # fmt: skip
    for name, idx in regions.items():
        log(f"    {name:6s} sd {sd_px[list(idx)].mean():.2f} px")
    log(f"  difference of two detections (experiment-13 'landmark jitter' units): {twin_jitter:.2f}%"
        " of the outer eye distance")  # fmt: skip

    front, feat = d["front"], d["feat"]
    spaces = {}
    for name, T in [("frontalized", lambda Z: front(Z)[:, NJ]),
                    ("procrustes", lambda Z: procrustes_normalize(Z)[:, NJ])]:  # fmt: skip
        dn = (T(B) - T(A)).reshape(len(A), -1)
        dn -= dn.mean(0)
        spaces[name] = dn.T @ dn / (len(dn) - 1) / 2  # single-detection noise covariance
    cov = spaces["frontalized"]
    ev = np.linalg.eigvalsh(cov)[::-1]
    mf = d["mean_face"]
    px_per_unit = 150 / float(np.linalg.norm(mf[36:42].mean(0) - mf[42:48].mean(0)))
    log(f"  frontalized: noise sd {np.sqrt(np.trace(cov) / len(cov)):.4f} normalised units "
        f"(1 unit = {px_per_unit:.0f} px); the top 10 of 102 directions hold "
        f"{ev[:10].sum() / ev.sum():.0%} of the noise variance, the top 20 {ev[:20].sum() / ev.sum():.0%}")  # fmt: skip
    sub = np.random.default_rng(0).choice(len(A), 12000, replace=False)
    dX = feat(front(B[sub])) - feat(front(A[sub]))
    return {"bank": bank, "cov": spaces, "dX": dX - dX.mean(0), "twin_jitter": twin_jitter,
            "sd_px": sd_px, "px_per_unit": px_per_unit, "noise_eig": ev}  # fmt: skip


# ============================================================================ A
def part_a(d: dict, nz: dict) -> None:
    log("\nA. Intrinsic dimension of expression geometry (51 non-jaw landmarks)")
    front, raw = d["front"], d["raw"]
    pose = load_pose_dataset()
    pm = pose.meta.reset_index(drop=True)
    frontal_view = (pm.yaw == 0) & (pm.pitch == 0)
    key = pm.subject.astype(str)
    is_ne = frontal_view & (pm.expression == "NE")
    ne = pm.index.to_series()[is_ne].groupby(key[is_ne]).first()
    sel = np.flatnonzero(frontal_view & (pm.expression != "NE") & key.isin(ne.index))
    sel_ne = ne.reindex(key[sel]).to_numpy()

    rows, spectra = [], {}
    for space, T in [("frontalized", lambda Z: front(Z)[:, NJ]),
                     ("procrustes", lambda Z: procrustes_normalize(Z)[:, NJ])]:  # fmt: skip
        noise = nz["cov"][space]
        G, P = T(raw), T(pose.landmarks.astype(np.float64))
        flat = lambda Z: Z.reshape(len(Z), -1)
        sets = {
            "MorphSet expressions (apex)": (flat(G - G[d["nrow"]])[d["apex"]], 2),
            "real expressions (frontal)": (flat(P[sel] - P[sel_ne]), 2),
            "identity (neutral faces)": (flat(G[d["neutral"]]), 1),
        }
        for name, (D, n_det) in sets.items():
            snr = whitened_snr(D, noise, n_det)
            iso = np.linalg.eigvalsh(np.cov(D - D.mean(0), rowvar=False))[::-1]
            p, n = len(snr), len(D)
            detect = n_det * ((1 + np.sqrt(p / n)) ** 2 - 1)  # Marchenko-Pastur edge, SNR units
            rows.append({"space": space, "set": name, "n": n,
                         "dims_95pct_var": int(np.searchsorted(np.cumsum(iso) / iso.sum(), 0.95) + 1),
                         "isotropic_floor": int((iso > n_det * np.trace(noise) / len(noise)).sum()),
                         "detectable": int((snr > detect).sum()), "snr_gt_1": int((snr > 1).sum()),
                         "snr_gt_2": int((snr > 2).sum()), "snr_gt_4": int((snr > 4).sum())})  # fmt: skip
            spectra[(space, name)] = snr
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "14_dimensions.csv", index=False)
    log(df.to_string(index=False))
    pd.DataFrame({f"{s}: {n}": pd.Series(v[:80]) for (s, n), v in spectra.items()}).to_csv(
        OUT / "14_spectra.csv", index_label="dim"
    )

    # Figure 1: the noise and the SNR spectra
    fig = plt.figure(figsize=(13, 4.3))
    ax0 = fig.add_subplot(1, 3, 1)
    mf = d["mean_face"]
    plot_landmarks(mf, ax=ax0, color="#9ca3af")
    bank = nz["bank"]
    to_units = 8 * 150 / nz["px_per_unit"]  # eye distances -> normalised units, x8 for visibility
    for i in range(68):
        w, V = np.linalg.eigh(np.cov(bank[:, i].T))
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        ax0.add_patch(Ellipse(mf[i], 2 * to_units * np.sqrt(w[1]), 2 * to_units * np.sqrt(w[0]),
                              angle=ang, color=ORANGE if i < 17 else BLUE, alpha=0.55, lw=0))  # fmt: skip
    ax0.set_facecolor("white")
    ax0.grid(False)
    ax0.set(
        xticks=[],
        yticks=[],
        title="dlib re-detection noise (1 sd ellipses, \N{MULTIPLICATION SIGN}8)",
    )
    ax1 = fig.add_subplot(1, 3, 2)
    k = np.arange(1, 61)
    deform = (d["frontal"] - d["frontal"][d["nrow"]])[:, NJ].reshape(len(d["raw"]), -1)[d["apex"]]
    expr_ev = np.linalg.eigvalsh(np.cov(deform, rowvar=False))[::-1]
    ax1.semilogy(k, expr_ev[:60], "o-", color=BLUE, label="expression deformations")
    ax1.semilogy(k, 2 * nz["noise_eig"][:60], "s-", color=GREY, label="noise of a deformation")
    ax1.axhline(2 * nz["noise_eig"].mean(), color=GREY, ls=":", lw=1,
                label="isotropic noise (same total)")  # fmt: skip
    ax1.set(xlabel="principal direction (own ordering)", ylabel="variance (normalised units²)",
            title="Noise is concentrated in few directions")  # fmt: skip
    ax1.legend(loc="upper right")
    ax2 = fig.add_subplot(1, 3, 3)
    for name, c, mk in [("MorphSet expressions (apex)", BLUE, "o"),
                        ("identity (neutral faces)", YELLOW, "^")]:  # fmt: skip
        s = spectra[("frontalized", name)][:60]
        ax2.semilogy(k[s > 0], s[s > 0], marker=mk, color=c, label=name)
    ax2.axhline(1, color="#333333", ls="--", lw=1)
    ax2.axhline(2, color="#333333", ls=":", lw=1)
    ax2.text(59, 1.08, "SNR 1", ha="right", fontsize=7.5)
    ax2.text(59, 2.16, "SNR 2", ha="right", fontsize=7.5)
    ax2.axvspan(20, 35, **BAND)
    ax2.axvline(31, color="k", lw=0.8, ls=":")
    ax2.set(xlabel="noise-whitened direction", ylabel="signal variance / noise variance",
            title="Signal-to-noise ratio of each direction")  # fmt: skip
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_spectra.png")
    plt.close(fig)


# ============================================================================ B, C
def part_bc(d: dict, nz: dict) -> pd.DataFrame:
    log("\nB. PLS components and action units;  C. signal-to-noise per component")
    X, Y, persons = d["X"], d["Y"], d["persons"]
    x_mean, x_std = X.mean(0), X.std(0, ddof=1)
    Xs = (X - x_mean) / x_std
    Ys = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    pls = PLSRegression(K, scale=False).fit(Xs, Ys)
    T, W = pls.x_scores_, pls.x_rotations_
    neutral, nrow, expr = d["neutral"], d["nrow"], d["expr"]
    var_id, var_ex = T[neutral].var(0), (T - T[nrow])[expr].var(0)
    q = pls.y_loadings_
    y_share = (q**2).sum(0) * (T**2).sum(0) / (Ys**2).sum()

    # facial movement of each component; localisation and best action unit, with random nulls
    mean_face, pairs = d["mean_face"], d["feat"].pairs
    loadings = pls.x_loadings_ * x_std[:, None]
    moves = np.array([equivalent_movement(loadings[:, c], mean_face, pairs) for c in range(K)])
    rng = np.random.default_rng(0)
    rand_moves = np.array(
        [
            equivalent_movement(rng.normal(size=len(pairs)) * x_std, mean_face, pairs)
            for _ in range(300)
        ]
    )
    regions = [r for r in REGIONS if r != "jaw"]

    def concentration(mv: np.ndarray) -> float:
        p = movement_region_shares(mv)[regions].to_numpy()
        p = p / p.sum()
        return float(1 - -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(regions)))

    templates = np.array([action_unit_field(n, mean_face)[NJ].ravel() for n in ACTION_UNITS])
    unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
    M, R = unit(moves[:, NJ].reshape(K, -1)), unit(rand_moves[:, NJ].reshape(len(rand_moves), -1))
    cos, cos_null = np.abs(M @ templates.T), np.abs(R @ templates.T).max(1)
    conc, conc_null = (
        np.array([concentration(mv) for mv in moves]),
        np.array([concentration(mv) for mv in rand_moves]),
    )

    # replicability: best match of each component in a model fitted on the other half of the persons
    uniq = np.unique(persons)
    repl, subspace = [], []
    for s in range(2):
        half = np.isin(persons, np.random.default_rng(s).permutation(uniq)[: len(uniq) // 2])
        WA = PLSRegression(K, scale=False).fit(Xs[half], Ys[half]).x_rotations_
        WB = PLSRegression(K, scale=False).fit(Xs[~half], Ys[~half]).x_rotations_
        repl.append(np.abs(unit(WA.T) @ unit(WB.T).T).max(1))
        subspace.append([
            (np.linalg.svd(np.linalg.qr(WA[:, :k])[0].T @ np.linalg.qr(WB[:, :k])[0], compute_uv=False) ** 2).mean()
            for k in range(1, K + 1)
        ])  # fmt: skip

    # action-unit coverage of the span of the first k component movements
    def coverage(Mk: np.ndarray) -> np.ndarray:
        Q = np.linalg.qr(Mk.T)[0]
        return ((templates @ Q) ** 2).sum(1)

    cov = np.array([coverage(M[:k]) for k in range(1, K + 1)])
    cov_rand = np.array([coverage(R[:k]) for k in range(1, K + 1)])

    # C. signal-to-noise along each component (score direction W, standardised features):
    # observed deformation variance = signal + 2 x noise (two detections); noise from the twins
    noise_k = ((nz["dX"] / x_std) @ W).var(0) / 2
    observed_k = ((Xs - Xs[nrow])[expr] @ W).var(0)
    snr = (observed_k - 2 * noise_k) / noise_k

    comp = pd.DataFrame({
        "component": np.arange(1, K + 1), "y_variance_share": y_share,
        "identity_share": var_id / (var_id + var_ex), "best_au": [ACTION_UNITS[i] for i in cos.argmax(1)],
        "best_au_cos": cos.max(1), "localisation": conc, "replicability": np.mean(repl, 0),
        "subspace_stability_k": np.mean(subspace, 0), "deformation_var": observed_k,
        "noise_var": noise_k, "snr": snr,
        "au_coverage_k": cov.mean(1), "au_coverage_random_k": cov_rand.mean(1),
    })  # fmt: skip
    comp["localised"] = comp.localisation > np.quantile(conc_null, 0.95)
    comp["au_match"] = comp.best_au_cos > np.quantile(cos_null, 0.95)
    comp.to_csv(OUT / "14_components.csv", index=False)
    pd.DataFrame(cov, columns=ACTION_UNITS, index=np.arange(1, K + 1)).to_csv(
        OUT / "14_au_coverage.csv", index_label="k"
    )
    log(f"  nulls (random directions, 95th pct): best-AU cosine {np.quantile(cos_null, 0.95):.2f}, "
        f"localisation {np.quantile(conc_null, 0.95):.2f}")  # fmt: skip
    show = comp.set_index("component").loc[[1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 31, 35, 40, 50, 60]]
    log(show[["best_au", "best_au_cos", "localised", "identity_share", "replicability", "snr",
              "au_coverage_k", "au_coverage_random_k"]].round(3).to_string())  # fmt: skip
    for lo, hi in [(1, 10), (11, 20), (21, 31), (32, 45), (46, 60)]:
        blk = comp[(comp.component >= lo) & (comp.component <= hi)]
        log(f"  components {lo:2d}-{hi:2d}: AU-matched {blk.au_match.sum():2d}/{len(blk)}, localised "
            f"{blk.localised.sum():2d}/{len(blk)}, replicability {blk.replicability.mean():.2f}, SNR {blk.snr.median():.2f}")  # fmt: skip
    below = lambda t: int(np.argmax(snr < t)) + 1
    log(f"  first component with SNR < 4: {below(4)}, < 2: {below(2)}, < 1: {below(1)}; "
        f"components with SNR >= 1: {(snr >= 1).sum()}, >= 2: {(snr >= 2).sum()}")  # fmt: skip
    log("  per-AU coverage at k = 10 / 20 / 31:")
    for n, c10, c20, c31 in zip(ACTION_UNITS, cov[9], cov[19], cov[30], strict=True):
        log(f"    {n:28s} {c10:.2f}  {c20:.2f}  {c31:.2f}")

    # figures
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    for c, ax in enumerate(axes.ravel()):
        plot_movement(
            ax,
            mean_face,
            moves[c],
            title=f"component {c + 1}\n≈ {comp.best_au[c]} (cos {comp.best_au_cos[c]:.2f})",
        )
        ax.set_facecolor("white")
        ax.grid(False)
        ax.set(xticks=[], yticks=[])
        ax.title.set_fontsize(9.5)
    fig.suptitle(
        "Facial movement encoded by the first 8 PLS components and the best-matching action unit"
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig2_components.png")
    plt.close(fig)

    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(k, comp.au_coverage_k, color=BLUE, label="PLS components")
    axes[0].plot(k, comp.au_coverage_random_k, color=GREY, ls="--", label="random directions")
    axes[0].plot(k, comp.au_coverage_k - comp.au_coverage_random_k, color=ORANGE,
                 label="excess over random")  # fmt: skip
    axes[0].set(xlabel="number of components k", ylabel="share of action-unit energy in span",
                title="Action units captured by the first k components")  # fmt: skip
    im = axes[1].imshow(cov.T, aspect="auto", cmap=SEQ, vmin=0, vmax=1,
                        extent=(0.5, K + 0.5, len(ACTION_UNITS) - 0.5, -0.5))  # fmt: skip
    axes[1].set_yticks(
        range(len(ACTION_UNITS)), [n.split(" ", 1)[0] for n in ACTION_UNITS], fontsize=7
    )
    axes[1].grid(False)
    axes[1].set(xlabel="number of components k", title="Coverage per action unit")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    pos = snr > 0
    axes[2].semilogy(k[pos], snr[pos], "o-", color=BLUE, label="signal / noise")
    axes[2].axhline(1, color="#333333", ls="--", lw=1)
    axes[2].axhline(2, color="#333333", ls=":", lw=1)
    axes[2].set(xlabel="PLS component", ylabel="expression variance / noise variance",
                title="Signal-to-noise ratio along each component")  # fmt: skip
    for ax in (axes[0], axes[2]):
        ax.axvspan(20, 35, **BAND)
        ax.axvline(31, color="k", lw=0.8, ls=":")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig3_coverage_snr.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(k, comp.replicability, color=BLUE,
            label="component replicability (other half of the people)")  # fmt: skip
    ax.plot(k, comp.identity_share, color=ORANGE, label="identity share of score variance")
    ax.scatter(k[comp.localised], np.full(comp.localised.sum(), 0.08), marker="|", s=80,
               color="#333333", label="localised like an action unit")  # fmt: skip
    ax.axvspan(20, 35, **BAND)
    ax.set(xlabel="PLS component", ylim=(0, 1.05), title="Properties of each PLS component")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_component_properties.png")
    plt.close(fig)
    return comp


# ============================================================================ D, E
def part_de(d: dict, nz: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("\nD. Accuracy vs live behaviour as complexity grows")
    X, Y, persons, feat, front = d["X"], d["Y"], d["persons"], d["feat"], d["front"]
    nrow, expr, bank = d["nrow"], d["expr"], nz["bank"]
    keys = (
        [("PLS", k) for k in range(1, K + 1)]
        + [("ridge", a) for a in ALPHAS]
        + [("kernel ridge", np.nan)]
    )
    # E: a random subset of test faces with extra measured noise at several levels
    sub = np.sort(np.random.default_rng(3).choice(len(X), 15000, replace=False))
    where = np.full(len(X), -1)
    where[sub] = np.arange(len(sub))
    XE = {
        lv: feat(front(add_noise(d["raw"][sub], bank, np.random.default_rng(20 + i), lv)))
        for i, lv in enumerate(NOISE_LEVELS)
    }
    oof = {key: np.zeros_like(Y) for key in keys}
    oofE = {(lv, key): np.zeros((len(sub), 3)) for lv in NOISE_LEVELS for key in keys}
    for f in repeated_group_kfold(persons, n_splits=5, seed=0):
        pp, rp = pls_path(X[f.train], Y[f.train], K), ridge_path(X[f.train], Y[f.train], ALPHAS)
        km = kernel_ridge(X.shape[1]).fit(X[f.train], Y[f.train])
        te = f.test[where[f.test] >= 0]
        for Z, rows, store in [(X[f.test], f.test, None)] + [
            (XE[lv][where[te]], where[te], lv) for lv in NOISE_LEVELS
        ]:
            preds = [*pp.predict(Z), *rp.predict(Z), km.predict(Z)]
            for key, p in zip(keys, preds, strict=True):
                (oof[key] if store is None else oofE[(store, key)])[rows] = p
        log(f"  CV fold {f.fold} done")

    pose = load_pose_dataset()
    pm = pose.meta.reset_index(drop=True)
    unseen = ((pm.family != "kdef") & (pm.pitch == 0) & (pm.yaw.abs() <= 45)).to_numpy()
    n0 = np.flatnonzero(unseen & (pm.expression == "NE").to_numpy() & (pm.yaw == 0).to_numpy())
    L0 = pose.landmarks[n0].astype(np.float64)
    F0 = front(L0)
    inputs = {"neutral": feat(F0), "pose": feat(front(pose.landmarks[np.flatnonzero(unseen)]))}
    inputs |= {
        f"noisy{i}": feat(front(add_noise(L0, bank, np.random.default_rng(10 + i))))
        for i in range(3)
    }
    inputs |= {
        n: feat(np.stack([f + AU_STEP * action_unit_field(n, f) for f in F0])) for n in ACTION_UNITS
    }
    um = pm.iloc[np.flatnonzero(unseen)].reset_index(drop=True)
    log(f"  unseen people: {len(n0)} neutral frontal faces, {unseen.sum()} faces across poses")

    def behaviour(predict) -> dict:
        P = {name: np.clip(predict(Z), -1, 1)[:, :2] for name, Z in inputs.items()}
        num = den = 0.0
        for n in ACTION_UNITS:
            r = P[n] - P["neutral"]
            num, den = num + ((r - r.mean(0)) ** 2).sum(), den + (r**2).sum()
        spread = (
            um.assign(A=P["pose"][:, 0], V=P["pose"][:, 1])
            .groupby(["subject", "expression"])[["A", "V"]]
            .std()
        )
        return {
            "identity_leakage": float(P["neutral"].std(0).mean()),
            "au_heterogeneity": num / den,
            "jitter": float(
                np.mean([np.abs(P[f"noisy{i}"] - P["neutral"]).mean() for i in range(3)])
            ),
            "pose_sensitivity": float(spread.mean().mean()),
        }

    path, rpath = pls_path(X, Y, K), ridge_path(X, Y, ALPHAS)
    km = kernel_ridge(X.shape[1]).fit(X, Y)
    xs = (X - X.mean(0)) / X.std(0, ddof=1)
    ev = np.linalg.eigvalsh(xs.T @ xs)
    dof = {a: float((ev / (ev + a * len(X))).sum()) for a in ALPHAS}
    rows, rowsE = [], []
    for model, param in keys:
        if model == "PLS":
            predict = lambda Z, k=param: Z @ path.coef[k - 1] + path.intercept[k - 1]
        elif model == "ridge":
            j = int(np.flatnonzero(param == ALPHAS)[0])
            predict = lambda Z, j=j: Z @ rpath.coef[j] + rpath.intercept[j]
        else:
            predict = km.predict
        p = oof[(model, param)]
        delta = p - p[nrow]
        row = {"model": model, "param": param, **behaviour(predict)}
        for t, name in enumerate("AVI"):
            row[f"ccc_{name}"] = ccc(Y[:, t], p[:, t])
            row[f"delta_ccc_{name}"] = ccc(Y[expr, t], delta[expr, t])
        if model == "ridge":
            row["ridge_dof"] = dof[param]
        rows.append(row)
        for lv in (0.0, *NOISE_LEVELS):
            pe = p[sub] if lv == 0 else oofE[(lv, (model, param))]
            rowsE.append({"model": model, "param": param, "extra_noise": lv,
                          "total_noise": float(np.sqrt(1 + lv**2)),
                          "ridge_dof": dof.get(param, np.nan),
                          "ccc": float(np.mean([ccc(Y[sub, t], pe[:, t]) for t in range(3)]))})  # fmt: skip
    df = pd.DataFrame(rows)
    df["ccc"] = df[["ccc_A", "ccc_V", "ccc_I"]].mean(axis=1)
    df["delta_ccc"] = df[["delta_ccc_A", "delta_ccc_V", "delta_ccc_I"]].mean(axis=1)
    df.to_csv(OUT / "14_tradeoff.csv", index=False)
    cols = ["model", "param", "ccc", "delta_ccc", "identity_leakage", "au_heterogeneity", "jitter",
            "pose_sensitivity", "ridge_dof"]  # fmt: skip
    show = df[(df.model != "PLS") | df.param.isin([5, 10, 15, 20, 25, 31, 40, 50, 60])]
    log(show[cols].round(3).to_string(index=False))

    dfE = pd.DataFrame(rowsE)
    dfE.to_csv(OUT / "14_noise_accuracy.csv", index=False)
    log("\nE. Accuracy when test faces carry more landmark noise (15,000 CV test faces)")
    for _lv, g in dfE.groupby("extra_noise"):
        gp, gr, gk = g[g.model == "PLS"], g[g.model == "ridge"], g[g.model == "kernel ridge"]
        best, bestr = gp.loc[gp.ccc.idxmax()], gr.loc[gr.ccc.idxmax()]
        within = gp[gp.ccc >= best.ccc - 0.005].param.min()
        log(f"  noise x{g.total_noise.iloc[0]:.2f}: best PLS k={int(best.param)} (CCC {best.ccc:.3f}; "
            f"within 0.005 from k={int(within)}), PLS(31) {gp[gp.param == 31].ccc.iloc[0]:.3f}, "
            f"best ridge {bestr.ridge_dof:.0f} df (CCC {bestr.ccc:.3f}), kernel {gk.ccc.iloc[0]:.3f}")  # fmt: skip

    # figures
    pls_df, ridge_df, kr = (df[df.model == "PLS"], df[df.model == "ridge"],
                            df[df.model == "kernel ridge"].iloc[0])  # fmt: skip
    metrics = [("ccc", "CCC (5-fold CV by person)", "higher is better"),
               ("jitter", "jitter |ΔA|,|ΔV| under measured noise", "lower is better"),
               ("pose_sensitivity", "spread across head yaw", "lower is better"),
               ("au_heterogeneity", "action-unit response heterogeneity", "lower is better")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, (col, title, note) in zip(axes, metrics, strict=True):
        ax.plot(pls_df.param, pls_df[col], color=BLUE, label="PLS, k components")
        ax.axhline(kr[col], color=ORANGE, ls="--", label="kernel ridge")
        ax.axvspan(20, 35, **BAND)
        ax.axvline(31, color="k", lw=0.8, ls=":")
        ax.set(xlabel="PLS components k", title=f"{title}\n({note})")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig6_tradeoff.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sc = ax.scatter(pls_df.jitter, pls_df.ccc, c=pls_df.param, cmap=SEQ, s=22,
                    label="PLS (colour: k)", zorder=3)  # fmt: skip
    ax.plot(ridge_df.jitter, ridge_df.ccc, "s-", ms=5, color=AQUA, label="ridge (penalty path)")
    ax.scatter([kr.jitter], [kr.ccc], marker="*", s=200, color=ORANGE, label="kernel ridge",
               zorder=5, edgecolor="white")  # fmt: skip
    p31 = pls_df[pls_df.param == 31].iloc[0]
    ax.annotate("PLS(31)", (p31.jitter, p31.ccc), xytext=(8, -12), textcoords="offset points",
                fontsize=8)  # fmt: skip
    fig.colorbar(sc, ax=ax, label="PLS components k")
    ax.set(xlabel="jitter under measured noise (lower is better)", ylabel="CCC (higher is better)",
           title="Accuracy vs jitter: linear models trade one for the other")  # fmt: skip
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_pareto.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    shades = SEQ(np.linspace(0.05, 1, len(NOISE_LEVELS) + 1))
    for (_lv, g), c in zip(dfE.groupby("extra_noise"), shades, strict=True):
        lab = f"noise \N{MULTIPLICATION SIGN} {g.total_noise.iloc[0]:.1f}"
        gp, gr = g[g.model == "PLS"], g[g.model == "ridge"]
        axes[0].plot(gp.param, gp.ccc, color=c, label=lab)
        b = gp.loc[gp.ccc.idxmax()]
        axes[0].plot([b.param], [b.ccc], "o", color=c, ms=7, mec="white")
        axes[1].semilogx(gr.ridge_dof, gr.ccc, "s-", color=c, label=lab)
        b = gr.loc[gr.ccc.idxmax()]
        axes[1].plot([b.ridge_dof], [b.ccc], "o", color=c, ms=8, mec="white")
    axes[0].axvspan(20, 35, **BAND)
    axes[0].axvline(31, color="k", lw=0.8, ls=":")
    axes[0].set(xlabel="PLS components k", ylabel="CCC (5-fold CV by person)",
                title="PLS: the best k falls as landmark noise grows")  # fmt: skip
    axes[1].set(xlabel="ridge effective degrees of freedom",
                title="Ridge: the best penalty rises with noise")  # fmt: skip
    axes[0].legend(
        title="test-face noise (\N{MULTIPLICATION SIGN} one detection)", title_fontsize=8
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig5_noise_accuracy.png")
    plt.close(fig)
    return df, dfE


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = load()
    log(f"MorphSet: {len(d['Y']):,} original images of {len(np.unique(d['persons']))} persons\n")
    nz = part_n(d)
    part_a(d, nz)
    part_bc(d, nz)
    part_de(d, nz)
    log(f"\n({time.perf_counter() - t0:.0f} s)")
    (OUT / "14_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
