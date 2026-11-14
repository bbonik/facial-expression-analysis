"""Frontalization: reproduce, evaluate by head pose, improve.

5-fold CV grouped by person on the multi-pose dataset (the mean-face template
and the frontalization weights are fit on training persons only).

Part 1 - landmark error of the frontalized shape vs the (identity-normalised)
frontal target, by |yaw|, for mirroring modes and ridge penalties. Baselines:
  none      - Procrustes-normalised input used as is
  identity  - input with facial parts moved to the template (no pose correction)
Also: CV grouped by subject id instead of person, to measure identity leakage.

Part 2 - pose invariance of the emotion estimate on held-out persons: for every
(posed, frontal) pair of the same subject and expression, |delta arousal| and
|delta valence| between the two predictions. Each pipeline uses its own PLS(31)
emotion model trained on MorphSet with that pipeline's frontalization:
  no_frontalization, frontalization_2021 (naive mirror, lstsq), improved.

Usage:
  python experiments/07_frontalization.py --part 1
  python experiments/07_frontalization.py --part 2 --improved permute:1e-4

Outputs: outputs/07_landmark_error.csv, outputs/07_pose_invariance.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.frontalization import Frontalizer
from facial_expression_analysis.frontalization_training import (
    MirrorMode,
    align_parts_to_template,
    assign_pose_persons,
    build_training_pairs,
    fit_frontalization,
    frontalization_error,
    load_pose_dataset,
    mean_shape,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
RIDGES = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
YAW_BINS = [0, 15, 22, 30, 45, 67]


def evaluate_landmarks(ds, pairs, pred, tags):
    """Landmark error on real (non-mirrored) test pairs, per |yaw|."""
    real = ~pairs.mirrored
    meta = ds.meta.iloc[pairs.row[real]]
    err = frontalization_error(pred[real], pairs.targets[real])
    df = pd.DataFrame(
        {"abs_yaw": meta["yaw"].abs().to_numpy(), "family": meta["family"].to_numpy(), "err": err}
    )
    df = df[df.abs_yaw.isin(YAW_BINS)]
    return [
        {**tags, "abs_yaw": y, "err": g.err.mean(), "n": len(g)} for y, g in df.groupby("abs_yaw")
    ]


def main(part: int, improved: tuple[MirrorMode, float] | None):
    ds = load_pose_dataset()
    persons = assign_pose_persons(ds)
    rows_err, rows_inv = [], []
    morph = load_morphset() if part == 2 else None

    groupings = [("person", persons)]
    if part == 1:
        groupings.append(("subject", ds.meta["subject"].to_numpy()))
    for grouping, groups in groupings:
        for fold, (tr, te) in enumerate(
            GroupKFold(5, shuffle=True, random_state=0).split(ds.landmarks, groups=groups)
        ):
            t0 = time.time()
            train, test = (
                ds.subset(np.isin(np.arange(len(ds)), tr)),
                ds.subset(np.isin(np.arange(len(ds)), te)),
            )
            template = mean_shape(train.landmarks)
            test_pairs = build_training_pairs(test, template, MirrorMode.NONE)
            base = {"grouping": grouping, "fold": fold}
            rows_err += evaluate_landmarks(
                test, test_pairs, test_pairs.inputs, {**base, "model": "none"}
            )
            rows_err += evaluate_landmarks(
                test,
                test_pairs,
                align_parts_to_template(test_pairs.inputs, template),
                {**base, "model": "identity"},
            )
            weights = {}
            for mode in MirrorMode:
                pairs = build_training_pairs(train, template, mode)
                for lam in RIDGES if grouping == "person" else [0.0]:
                    W = fit_frontalization(pairs, ridge=lam)
                    weights[(mode, lam)] = W
                    pred = Frontalizer(W)(test_pairs.inputs)  # inputs are already normalised
                    rows_err += evaluate_landmarks(
                        test, test_pairs, pred, {**base, "model": f"{mode.value}/{lam:g}"}
                    )
            print(f"[landmarks] {grouping} fold {fold} done in {time.time() - t0:.0f}s", flush=True)

            if part != 2:
                continue
            # ---- Part 2: pose invariance of the emotion estimate on held-out persons
            pipelines = {
                "no_frontalization": None,
                "frontalization_2021": weights[(MirrorMode.NAIVE, 0.0)],
                "improved": weights[improved] if improved else None,
            }
            feats = PairwiseDistanceFeatures()
            meta = test.meta.reset_index(drop=True)
            frontal = (meta.yaw == 0) & (meta.pitch == 0)
            key = meta.subject.astype(str) + "|" + meta.expression
            ref_row = key.map(meta.index.to_series()[frontal].groupby(key[frontal]).first())
            ok = ref_row.notna() & ~frontal & meta.yaw.abs().isin(YAW_BINS)
            posed_idx, ref_idx = np.flatnonzero(ok), ref_row[ok].to_numpy(dtype=int)
            for name, W in pipelines.items():
                if name == "improved" and W is None:
                    continue
                front = Frontalizer(W) if W is not None else np.asarray
                Xm = feats(front(morph.landmarks))
                pls = PLSRegression(31).fit(Xm, morph.targets)
                pred_posed = pls.predict(feats(front(test.landmarks[posed_idx])))
                pred_ref = pls.predict(feats(front(test.landmarks[ref_idx])))
                d = np.abs(np.clip(pred_posed[:, :2], -1, 1) - np.clip(pred_ref[:, :2], -1, 1))
                df = pd.DataFrame(
                    {
                        "abs_yaw": meta.yaw.abs().to_numpy()[posed_idx],
                        "family": meta.family.to_numpy()[posed_idx],
                        "dA": d[:, 0],
                        "dV": d[:, 1],
                    }
                )
                for (fam, y), g in df.groupby(["family", "abs_yaw"]):
                    rows_inv.append(
                        {
                            "fold": fold,
                            "pipeline": name,
                            "family": fam,
                            "abs_yaw": y,
                            "dA": g.dA.mean(),
                            "dV": g.dV.mean(),
                            "n": len(g),
                        }
                    )
            print(f"[invariance] fold {fold} done in {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows_inv).to_csv(OUT / "07_pose_invariance.csv", index=False)
        if part == 1:
            pd.DataFrame(rows_err).to_csv(OUT / "07_landmark_error.csv", index=False)

    if part == 1:
        err = pd.DataFrame(rows_err)
        table = err.groupby(["grouping", "model", "abs_yaw"]).err.mean().unstack()
        print(table.round(4).to_string())
    else:
        inv = pd.DataFrame(rows_inv)
        print(
            inv.groupby(["pipeline", "abs_yaw"])[["dA", "dV"]].mean().unstack().round(4).to_string()
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2], required=True)
    ap.add_argument("--improved", help="mirror_mode:ridge, e.g. permute:1e-4")
    args = ap.parse_args()
    imp = None
    if args.improved:
        mode, lam = args.improved.split(":")
        imp = (MirrorMode(mode), float(lam))
    main(args.part, imp)
