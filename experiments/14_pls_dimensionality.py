"""How many dimensions does facial expression have, and why does PLS(31) behave best live?

Analyses behind ``paper/pls_action_units.md`` (original MorphSet images, no mirrored copies):

A. Intrinsic dimension of expression geometry: eigen-spectrum of within-person deformations
   (expression minus the same person's neutral face, 51 non-jaw frontalized landmarks) against
   the dlib noise floor measured from MorphSet's independently re-detected mirror images;
   MorphSet (morphed) and Multi-PIE / KDEF / CAS-PEAL (real expressions).
B. PLS components 1-60: facial movement of each component, its best-matching action-unit
   template (and a random-direction null), localisation, identity share of its score variance,
   replicability across disjoint halves of the persons, and how much of 16 action-unit
   templates the span of the first k components captures (vs random directions).
C. Signal-to-noise ratio along each component: expression variance vs dlib-noise variance.
D. Accuracy vs live behaviour as complexity grows (PLS k = 1-60, ridge penalty path, kernel
   ridge): 5-fold person-grouped CV CCC, within-person ("delta") CCC, and on unseen people
   (Multi-PIE / CAS-PEAL) identity leakage, action-unit response heterogeneity, jitter under
   dlib-like landmark noise, and pose sensitivity; ridge effective degrees of freedom.

Usage: python experiments/14_pls_dimensionality.py        (about 15 minutes)
Outputs: outputs/14_*.csv, outputs/14_log.txt, paper/figures/*.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from facial_expression_analysis import Frontalizer
from facial_expression_analysis.constants import MIRROR_PERMUTATION
from facial_expression_analysis.features import PairwiseDistanceFeatures
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

OUT, FIG = Path("outputs"), Path("paper/figures")
K = 60  # PLS components examined
ALPHAS = np.logspace(-7, -1, 13)  # ridge penalties (per sample)
AU_STEP = 0.06  # size of a synthetic action unit, normalised units (~9 px, 6% of a 150 px IOD)
NJ = np.arange(17, 68)  # the 51 non-jaw landmarks used by the features
LOG: list[str] = []
COLORS = {"PLS": "#7c3aed", "ridge": "#0d9488", "kernel ridge": "#e11d48", "noise": "#6b7280"}


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def kernel_ridge(n_features: int):
    return make_pipeline(
        StandardScaler(),
        Nystroem(gamma=0.5 / n_features, n_components=2000, random_state=0),
        Ridge(alpha=1.0),
    )


def dlib_noise(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add dlib-like landmark noise: 1 px per coordinate at a 150 px eye distance."""
    iod = np.linalg.norm(landmarks[:, 36:42].mean(1) - landmarks[:, 42:48].mean(1), axis=1)
    return landmarks + rng.normal(0, 1.0, landmarks.shape) * (iod / 150.0)[:, None, None]


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
    return {
        "data": data, "meta": meta, "orig": orig, "front": front, "frontal": frontal,
        "feat": feat, "X": X, "Y": data.targets[orig], "persons": data.persons[orig],
        "neutral": neutral, "neutral_row": neutral_row,
        "mean_face": frontal[neutral].mean(0), "raw": data.landmarks[orig],
    }  # fmt: skip


# ============================================================================ A
def part_a(d: dict) -> float:
    log("A. Intrinsic dimension of expression geometry (51 non-jaw landmarks, frontalized)")
    data, m = d["data"], d["data"].meta
    F = d["front"](data.landmarks)
    stem = m.filename.str.replace(r"_[12]\.jpg$", "", regex=True)
    a = pd.Series(m.index[~m.mirrored], index=stem[~m.mirrored])
    b = pd.Series(m.index[m.mirrored], index=stem[m.mirrored])
    pairs = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().astype(int)
    perm = np.array(MIRROR_PERMUTATION)
    fa, fb = F[pairs.a.to_numpy()], F[pairs.b.to_numpy()]
    diff = (fb - np.stack([-fa[:, perm, 0], fa[:, perm, 1]], -1))[:, NJ].reshape(len(fa), -1)
    noise_var = (diff - diff.mean(0)).var(0).mean() / 2  # one detection, per coordinate
    log(f"  dlib noise (mirror twins): sd {np.sqrt(noise_var):.4f} normalised units per coordinate")

    def spectrum(D: np.ndarray, floor: float, label: str) -> np.ndarray:
        D = D - D.mean(0)
        ev = np.linalg.eigvalsh(np.cov(D, rowvar=False))[::-1]
        cum = np.cumsum(ev) / ev.sum()
        log(f"  {label:44s} n={len(D):5d}  95% var: {np.searchsorted(cum, 0.95) + 1:2d} dims  "
            f"99%: {np.searchsorted(cum, 0.99) + 1:2d}  above noise floor: {(ev > floor).sum():2d}")  # fmt: skip
        return ev

    meta, fr = d["meta"], d["frontal"]
    expr = ~d["neutral"] & ~np.isnan(d["neutral_row"])
    apex = expr & (meta.filename.str.split("_").str[-2] == "10").to_numpy()
    deform = (fr - fr[np.nan_to_num(d["neutral_row"]).astype(int)])[:, NJ].reshape(len(fr), -1)
    rows = []
    # a deformation is the difference of two detections: noise floor 2 x noise_var
    rows.append(
        (
            "MorphSet (morphed), apex",
            spectrum(deform[apex], 2 * noise_var, "MorphSet expressions (apex)"),
        )
    )
    pose = load_pose_dataset()
    pm = pose.meta.reset_index(drop=True)
    frontal_view = (pm.yaw == 0) & (pm.pitch == 0)
    key = pm.subject.astype(str)
    is_ne = frontal_view & (pm.expression == "NE")
    ne = pm.index.to_series()[is_ne].groupby(key[is_ne]).first()
    sel = frontal_view & (pm.expression != "NE") & key.isin(ne.index)
    P = d["front"](pose.landmarks)
    real = (P[np.flatnonzero(sel)] - P[ne.reindex(key[sel]).to_numpy()])[:, NJ].reshape(
        sel.sum(), -1
    )
    rows.append(
        (
            "real expressions (Multi-PIE, KDEF, CAS-PEAL)",
            spectrum(real, 2 * noise_var, "real expressions (frontal)"),
        )
    )
    ident = fr[d["neutral"]][:, NJ].reshape(d["neutral"].sum(), -1)
    rows.append(
        (
            "identity (neutral faces of different people)",
            spectrum(ident, noise_var, "identity (neutral faces)"),
        )
    )
    pd.DataFrame({name: pd.Series(ev[:60]) for name, ev in rows}).to_csv(
        OUT / "14_spectra.csv", index_label="dim"
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for (name, ev), c in zip(rows, ["#7c3aed", "#0d9488", "#d97706"], strict=True):
        ax.semilogy(np.arange(1, 41), ev[:40], "o-", ms=3, lw=1.4, color=c, label=name)
    ax.axhline(
        2 * noise_var, color=COLORS["noise"], ls="--", lw=1, label="noise floor (deformations)"
    )
    ax.axhline(noise_var, color=COLORS["noise"], ls=":", lw=1, label="noise floor (single shapes)")
    ax.set(xlabel="dimension (principal component)", ylabel="variance (normalised units²)",
           title="Expression geometry: about 20 dimensions above the landmark noise floor")  # fmt: skip
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_spectra.png", dpi=150)
    return noise_var


# ============================================================================ B, C
def part_bc(d: dict) -> pd.DataFrame:
    log("\nB. PLS components and action units;  C. signal-to-noise per component")
    X, Y, persons = d["X"], d["Y"], d["persons"]
    x_mean, x_std = X.mean(0), X.std(0, ddof=1)
    Xs = (X - x_mean) / x_std
    Ys = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    pls = PLSRegression(K, scale=False).fit(Xs, Ys)
    T, W = pls.x_scores_, pls.x_rotations_
    neutral, nrow = d["neutral"], np.nan_to_num(d["neutral_row"]).astype(int)
    expr = ~neutral & ~np.isnan(d["neutral_row"])
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

    # C. signal-to-noise: expression deformation variance vs dlib-noise variance along each component
    sub = np.random.default_rng(1).choice(len(X), 6000, replace=False)
    noisy = (
        d["feat"](d["front"](dlib_noise(d["raw"][sub], np.random.default_rng(2)))) - x_mean
    ) / x_std
    snr = (Xs - Xs[nrow])[expr] @ W
    snr = snr.var(0) / ((noisy - Xs[sub]) @ W).var(0)

    comp = pd.DataFrame({
        "component": np.arange(1, K + 1), "y_variance_share": y_share,
        "identity_share": var_id / (var_id + var_ex), "best_au": [ACTION_UNITS[i] for i in cos.argmax(1)],
        "best_au_cos": cos.max(1), "localisation": conc, "replicability": np.mean(repl, 0),
        "subspace_stability_k": np.mean(subspace, 0), "snr": snr,
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
    show = comp.set_index("component").loc[[1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 31, 40, 50, 60]]
    log(show[["best_au", "best_au_cos", "localised", "identity_share", "replicability", "snr",
              "au_coverage_k", "au_coverage_random_k"]].round(3).to_string())  # fmt: skip
    for lo, hi in [(1, 10), (11, 20), (21, 31), (32, 45), (46, 60)]:
        blk = comp[(comp.component >= lo) & (comp.component <= hi)]
        log(f"  components {lo:2d}-{hi:2d}: AU-matched {blk.au_match.sum():2d}/{len(blk)}, localised "
            f"{blk.localised.sum():2d}/{len(blk)}, replicability {blk.replicability.mean():.2f}, SNR {blk.snr.median():.2f}")  # fmt: skip
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
        ax.title.set_fontsize(9.5)
    fig.suptitle(
        "Facial movement encoded by the first 8 PLS components and the best-matching action unit"
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig2_components.png", dpi=150)

    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(k, comp.au_coverage_k, color=COLORS["PLS"], label="PLS components")
    axes[0].plot(
        k, comp.au_coverage_random_k, color=COLORS["noise"], ls="--", label="random directions"
    )
    axes[0].plot(
        k,
        comp.au_coverage_k - comp.au_coverage_random_k,
        color="#d97706",
        label="excess over random",
    )
    axes[0].set(xlabel="number of components k", ylabel="share of action-unit energy in span",
                title="Action units captured by the first k components")  # fmt: skip
    im = axes[1].imshow(
        cov.T,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=(0.5, K + 0.5, len(ACTION_UNITS) - 0.5, -0.5),
    )
    axes[1].set_yticks(
        range(len(ACTION_UNITS)), [n.split(" ", 1)[0] for n in ACTION_UNITS], fontsize=7
    )
    axes[1].set(xlabel="number of components k", title="Coverage per action unit")
    fig.colorbar(im, ax=axes[1], shrink=0.8)
    axes[2].semilogy(k, snr, "o-", ms=3, color=COLORS["PLS"])
    axes[2].axhline(1, color=COLORS["noise"], ls="--", lw=1)
    axes[2].set(xlabel="PLS component", ylabel="expression variance / noise variance",
                title="Signal-to-noise ratio along each component")  # fmt: skip
    for ax in (axes[0], axes[2]):
        ax.axvspan(20, 35, color="#fde68a", alpha=0.35, lw=0)
        ax.axvline(31, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_coverage_snr.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(
        k,
        comp.replicability,
        color=COLORS["PLS"],
        label="component replicability (other half of the people)",
    )
    ax.plot(k, comp.identity_share, color="#d97706", label="identity share of score variance")
    ax.scatter(
        k[comp.localised],
        np.full(comp.localised.sum(), 0.08),
        marker="|",
        s=80,
        color="k",
        label="localised like an action unit",
    )
    ax.axvspan(20, 35, color="#fde68a", alpha=0.35, lw=0)
    ax.set(xlabel="PLS component", ylim=(0, 1.05), title="Properties of each PLS component")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_component_properties.png", dpi=150)
    return comp


# ============================================================================ D
def part_d(d: dict) -> pd.DataFrame:
    log("\nD. Accuracy vs live behaviour as complexity grows")
    X, Y, persons, feat, front = d["X"], d["Y"], d["persons"], d["feat"], d["front"]
    nrow = np.nan_to_num(d["neutral_row"]).astype(int)
    expr = ~d["neutral"] & ~np.isnan(d["neutral_row"])
    keys = (
        [("PLS", k) for k in range(1, K + 1)]
        + [("ridge", a) for a in ALPHAS]
        + [("kernel ridge", np.nan)]
    )
    oof = {key: np.zeros_like(Y) for key in keys}
    for f in repeated_group_kfold(persons, n_splits=5, seed=0):
        for k, p in enumerate(pls_path(X[f.train], Y[f.train], K).predict(X[f.test]), start=1):
            oof[("PLS", k)][f.test] = p
        for a, p in zip(
            ALPHAS, ridge_path(X[f.train], Y[f.train], ALPHAS).predict(X[f.test]), strict=True
        ):
            oof[("ridge", a)][f.test] = p
        oof[("kernel ridge", np.nan)][f.test] = (
            kernel_ridge(X.shape[1]).fit(X[f.train], Y[f.train]).predict(X[f.test])
        )
        log(f"  CV fold {f.fold} done")

    pose = load_pose_dataset()
    pm = pose.meta.reset_index(drop=True)
    unseen = ((pm.family != "kdef") & (pm.pitch == 0) & (pm.yaw.abs() <= 45)).to_numpy()
    n0 = np.flatnonzero(unseen & (pm.expression == "NE").to_numpy() & (pm.yaw == 0).to_numpy())
    F0 = front(pose.landmarks[n0])
    inputs = {"neutral": feat(F0), "pose": feat(front(pose.landmarks[np.flatnonzero(unseen)]))}
    inputs |= {
        f"noisy{i}": feat(front(dlib_noise(pose.landmarks[n0], np.random.default_rng(10 + i))))
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
    rows = []
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
            row["ridge_dof"] = float((ev / (ev + param * len(X))).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["ccc"] = df[["ccc_A", "ccc_V", "ccc_I"]].mean(axis=1)
    df["delta_ccc"] = df[["delta_ccc_A", "delta_ccc_V", "delta_ccc_I"]].mean(axis=1)
    df.to_csv(OUT / "14_tradeoff.csv", index=False)
    cols = [
        "model",
        "param",
        "ccc",
        "delta_ccc",
        "identity_leakage",
        "au_heterogeneity",
        "jitter",
        "pose_sensitivity",
        "ridge_dof",
    ]
    show = df[(df.model != "PLS") | df.param.isin([5, 10, 15, 20, 25, 31, 40, 50, 60])]
    log(show[cols].round(3).to_string(index=False))

    pls_df, ridge_df, kr = (
        df[df.model == "PLS"],
        df[df.model == "ridge"],
        df[df.model == "kernel ridge"].iloc[0],
    )
    metrics = [("ccc", "CCC (5-fold CV by person)", "higher is better"),
               ("jitter", "jitter |ΔA|,|ΔV| under dlib noise", "lower is better"),
               ("pose_sensitivity", "spread across head yaw", "lower is better"),
               ("au_heterogeneity", "action-unit response heterogeneity", "lower is better")]  # fmt: skip
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    for ax, (col, title, note) in zip(axes, metrics, strict=True):
        ax.plot(pls_df.param, pls_df[col], color=COLORS["PLS"], label="PLS, k components")
        ax.axhline(kr[col], color=COLORS["kernel ridge"], ls="--", label="kernel ridge")
        ax.axvspan(20, 35, color="#fde68a", alpha=0.35, lw=0)
        ax.axvline(31, color="k", lw=0.8, ls=":")
        ax.set(xlabel="PLS components k", title=f"{title}\n({note})")
        ax.title.set_fontsize(9.5)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_tradeoff.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sc = ax.scatter(
        pls_df.jitter, pls_df.ccc, c=pls_df.param, cmap="viridis", s=18, label="PLS (colour: k)"
    )
    ax.plot(
        ridge_df.jitter,
        ridge_df.ccc,
        "s-",
        ms=4,
        color=COLORS["ridge"],
        label="ridge (penalty path)",
    )
    ax.scatter(
        [kr.jitter],
        [kr.ccc],
        marker="*",
        s=180,
        color=COLORS["kernel ridge"],
        label="kernel ridge",
        zorder=5,
    )
    p31 = pls_df[pls_df.param == 31].iloc[0]
    ax.annotate(
        "PLS(31)", (p31.jitter, p31.ccc), xytext=(8, -12), textcoords="offset points", fontsize=8
    )
    fig.colorbar(sc, ax=ax, label="PLS components k")
    ax.set(xlabel="jitter under dlib noise (lower is better)", ylabel="CCC (higher is better)",
           title="Accuracy vs jitter: linear models trade one for the other")  # fmt: skip
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_pareto.png", dpi=150)
    return df


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    d = load()
    log(f"MorphSet: {len(d['Y']):,} original images of {len(np.unique(d['persons']))} persons\n")
    part_a(d)
    part_bc(d)
    part_d(d)
    log(f"\n({time.perf_counter() - t0:.0f} s)")
    (OUT / "14_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
