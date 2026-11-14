"""Landmark tracking and low-light robustness of the dlib landmark pipeline.

Live, the pipeline re-ran dlib's HOG face detector on every frame. Its boxes move in
steps (8-pixel grid, coarse scale pyramid) and in low light it misses faces. A tracker
can instead derive the next frame's box from the current landmarks, but the landmark
fitter is sensitive to how the box sits on the face: a tight rectangle around the
landmarks gives poor fits. Questions:

A. Can the detector's box be predicted from the 68 landmarks well enough that the fitter
   converges as if the detector had been run? (Tight rectangle vs fixed offset vs
   linear regression; IoU with the detector box and landmark difference.)
B. Noise floor: how much do the detector-based landmarks themselves change when the face
   moves by 1-4 px? Does tracking drift when it is iterated on the same image?
C. Simulated low-light video (LFW faces, slow random head motion, darkening, sensor
   noise, with and without camera auto-gain): landmark error and jitter, frames without a
   face, and the resulting arousal/valence error and jitter of the default model, for
   (1) the old per-frame detector, (2) the robust per-frame detector and (3) the tracker.

Landmark error is measured against the detector-based fit on the clean, unmoved photo
(transformed with the simulated motion), in % of the outer inter-ocular distance. Jitter
is the frame-to-frame change of that error (what is seen as shaking).

Data: Labeled Faces in the Wild (13,233 photos, 5,749 people, downloaded by scikit-learn,
~200 MB on first run). The box regressor trained on all of LFW is saved to
``src/facial_expression_analysis/assets/box_regressor.npz``.

Usage: python experiments/13_tracking.py [--n-seq 250]
Outputs: outputs/13_box.csv, outputs/13_lowlight.csv, outputs/13_log.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from facial_expression_analysis import EmotionEstimator
from facial_expression_analysis.assets import cache_dir
from facial_expression_analysis.landmarks import BoxRegressor, DlibLandmarkDetector
from facial_expression_analysis.landmarks.tracker import ASSET, LandmarkTracker

OUT = Path("outputs")
# (gain, noise sigma in grey levels, camera auto-gain)
CONDITIONS = {
    "clean": (1.0, 0.0, False),
    "dim": (0.3, 2.0, False),
    "dark": (0.1, 1.5, False),
    "night, auto-gain": (0.1, 1.5, True),
    "very dark, auto-gain": (0.05, 1.5, True),
}
LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(msg)


def load_lfw() -> dict[str, np.ndarray]:
    """LFW photos with the dlib detector box and landmarks of the central face (cached)."""
    path = cache_dir() / "lfw_dlib.npz"
    if path.is_file():
        return dict(np.load(path))
    from sklearn.datasets import fetch_lfw_people

    lfw = fetch_lfw_people(min_faces_per_person=0, resize=1.0, slice_=None, color=True)
    gray = ((lfw.images * 255).round().astype(np.uint8) @ [0.299, 0.587, 0.114]).round()
    gray = gray.astype(np.uint8)
    det = DlibLandmarkDetector(upsample=0)
    keep, boxes, lms = [], [], []
    for i, g in enumerate(gray):
        # the target face of an LFW photo is centred at (125, 125)
        central = [b for b in det.detect_boxes(g)
                   if abs((b.left + b.right) / 2 - 125) < 25 and abs((b.top + b.bottom) / 2 - 125) < 25]  # fmt: skip
        if central:
            b = central[-1]
            keep.append(i)
            boxes.append([b.left, b.top, b.right, b.bottom])
            lms.append(det.fit(g, b).landmarks)
    data = {"gray": gray[keep], "boxes": np.array(boxes, float), "landmarks": np.array(lms),
            "person": lfw.target[keep]}  # fmt: skip
    np.savez_compressed(path, **data)
    return data


def nme(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    iod = np.linalg.norm(ref[..., 36, :] - ref[..., 45, :], axis=-1)
    return np.linalg.norm(a - ref, axis=-1).mean(-1) / iod


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    h = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = w * h
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a) + area(b) - inter)


def warp(g: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        g, M, g.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def apply(M: np.ndarray, lm: np.ndarray) -> np.ndarray:
    return lm @ M[:, :2].T + M[:, 2]


def random_motion(r: np.random.Generator, shift: float, rot: float, scale: float) -> np.ndarray:
    M = cv2.getRotationMatrix2D((125, 125), r.uniform(-rot, rot), 1 + r.uniform(-scale, scale))
    M[:, 2] += r.uniform(-shift, shift, 2)
    return M


def degrade(
    g: np.ndarray, r: np.random.Generator, gain: float, sigma: float, auto: bool
) -> np.ndarray:
    x = np.clip(np.round(g * gain + r.normal(0, sigma, g.shape)), 0, 255)  # sensor: 8-bit output
    if auto:  # the camera amplifies the dark signal back, noise and quantisation included
        x = np.clip(np.round(x / gain), 0, 255)
    return x.astype(np.uint8)


def part_a(d: dict, det: DlibLandmarkDetector, test: np.ndarray, reg: BoxRegressor) -> None:
    log("A. Box from landmarks (2,000 test photos of people not used to train the regressor)")
    L, B = d["landmarks"], d["boxes"]
    tight = np.c_[L[..., 0].min(1), L[..., 1].min(1), L[..., 0].max(1), L[..., 1].max(1)]
    # tight rectangle + the average offset/scale between it and the detector box
    c_t, s_t = (tight[:, :2] + tight[:, 2:]) / 2, tight[:, 2:] - tight[:, :2]
    c_b, s_b = (B[:, :2] + B[:, 2:]) / 2, B[:, 2:] - B[:, :2]
    off = ((c_b - c_t) / s_t)[~test].mean(0)
    ratio = (s_b / s_t)[~test].mean(0)
    c, s = c_t + off * s_t, s_t * ratio
    variants = {"tight landmark rectangle": tight, "tight + mean offset/scale": np.c_[c - s / 2, c + s / 2],
                "regression (BoxRegressor)": reg.predict(L)}  # fmt: skip
    idx = np.flatnonzero(test)[:2000]
    rows = []
    for name, boxes in variants.items():
        lm = np.array([det.fit(d["gray"][i], boxes[i]).landmarks for i in idx])
        e = nme(lm, L[idx])
        rows.append({"box": name, "iou_with_detector": iou(boxes[idx], B[idx]).mean(),
                     "nme_mean": 100 * e.mean(), "nme_p95": 100 * np.quantile(e, 0.95),
                     "pct_over_5": 100 * (e > 0.05).mean()})  # fmt: skip
        log(
            f"  {name:28s} IoU {rows[-1]['iou_with_detector']:.3f}  landmark diff mean "
            f"{rows[-1]['nme_mean']:.2f}%  p95 {rows[-1]['nme_p95']:.2f}%  >5%: {rows[-1]['pct_over_5']:.1f}%"
        )
    pd.DataFrame(rows).to_csv(OUT / "13_box.csv", index=False)


def part_b(d: dict, det: DlibLandmarkDetector, test: np.ndarray, reg: BoxRegressor) -> None:
    idx = np.flatnonzero(test)[:400]
    log("\nB1. Noise floor: photo shifted by (dx, dx/2) px; ref = shifted clean landmarks")
    for dx in (1, 2, 3, 4):
        M = np.float32([[1, 0, dx], [0, 1, dx / 2]])
        e_det, e_trk = [], []
        for i in idx:
            g, ref = warp(d["gray"][i], M), apply(M, d["landmarks"][i])
            boxes = det.detect_boxes(g)
            if boxes:
                b = min(boxes, key=lambda b: abs(b.left + b.right - 2 * ref[:, 0].mean()))
                e_det.append(nme(det.fit(g, b).landmarks, ref))
            e_trk.append(nme(det.fit(g, reg.predict(ref)).landmarks, ref))
        log(
            f"  {dx} px: detector re-run {100 * np.mean(e_det):.2f}%   box from landmarks {100 * np.mean(e_trk):.2f}%"
        )
    log("\nB2. Drift: box -> fit -> box ... repeated on the same photo")
    err = np.zeros((len(idx), 30))
    for j, i in enumerate(idx):
        lm = d["landmarks"][i]
        for t in range(30):
            lm = det.fit(d["gray"][i], reg.predict(lm)).landmarks
            err[j, t] = nme(lm, d["landmarks"][i])
    for t in (0, 4, 29):
        log(
            f"  iteration {t + 1:2d}: mean {100 * err[:, t].mean():.2f}%  p95 {100 * np.quantile(err[:, t], 0.95):.2f}%"
        )


def run_sequences(d, idx, method, cond, estimator, reg, n_frames=15):
    gain, sigma, auto = CONDITIONS[cond]
    r = np.random.default_rng(7)  # same motion and noise for every method
    plain = DlibLandmarkDetector(upsample=0)
    robust = DlibLandmarkDetector(upsample=0, robust=True)
    tracker = LandmarkTracker(robust, reg)  # regressor not trained on these people
    err, jit, av_err, av_jit, lost, n = [], [], [], [], 0, 0
    for i in idx:
        tracker.reset()
        M = np.float32([[1, 0, 0], [0, 1, 0]])
        av_ref = estimator.predict(d["landmarks"][i])[0, :2]
        prev = None
        for _ in range(n_frames):
            M = random_motion(r, 2, 1, 0.01) @ np.r_[M, [[0, 0, 1]]]
            g = degrade(warp(d["gray"][i], M), r, gain, sigma, auto)
            ref = apply(M, d["landmarks"][i])
            n += 1
            faces = (
                tracker if method == "tracker" else plain if method == "detector" else robust
            ).detect(g)
            faces = [f for f in faces if np.linalg.norm(f.landmarks.mean(0) - ref.mean(0)) < 30]
            if not faces:
                lost += 1
                prev = None
                continue
            lm = faces[0].landmarks
            av = estimator.predict(lm)[0, :2]
            err.append(nme(lm, ref))
            av_err.append(np.linalg.norm(av - av_ref))
            if prev is not None:
                jit.append(nme((lm - ref) - prev[0] + ref, ref))
                av_jit.append(np.linalg.norm(av - prev[1]))
            prev = (lm - ref, av)
    return {"condition": cond, "method": method, "landmark_error": 100 * np.mean(err),
            "landmark_jitter": 100 * np.mean(jit), "av_error": np.mean(av_err),
            "av_jitter": np.mean(av_jit), "pct_no_face": 100 * lost / n}  # fmt: skip


def part_c(d: dict, test: np.ndarray, reg: BoxRegressor, n_seq: int) -> None:
    log(f"\nC. Simulated video: {n_seq} test faces x 15 frames, slow head motion")
    estimator = EmotionEstimator()
    idx = np.flatnonzero(test)[:n_seq]
    rows = []
    log(
        f"  {'condition':22s} {'method':16s} {'lm error':>8s} {'lm jitter':>9s} {'AV error':>8s} {'AV jitter':>9s} {'no face':>8s}"
    )
    for cond in CONDITIONS:
        for method in ("detector", "robust detector", "tracker"):
            row = run_sequences(d, idx, method, cond, estimator, reg)
            rows.append(row)
            log(
                f"  {cond:22s} {method:16s} {row['landmark_error']:7.2f}% {row['landmark_jitter']:8.2f}% "
                f"{row['av_error']:8.3f} {row['av_jitter']:9.3f} {row['pct_no_face']:7.1f}%"
            )
    pd.DataFrame(rows).to_csv(OUT / "13_lowlight.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seq", type=int, default=250)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    d = load_lfw()
    log(
        f"LFW: {len(d['gray'])} photos with a detected central face, {np.unique(d['person']).size} people"
    )
    rng = np.random.default_rng(0)
    people = np.unique(d["person"])
    rng.shuffle(people)
    test = np.isin(d["person"], people[: len(people) // 2])
    reg = BoxRegressor.fit(d["landmarks"][~test], d["boxes"][~test])

    full = BoxRegressor.fit(d["landmarks"], d["boxes"], metadata={
        "trained_on": "LFW, dlib HOG detector (upsample 0) boxes and dlib 68-point fits",
        "n_faces": len(d["gray"]), "script": "experiments/13_tracking.py"})  # fmt: skip
    full.save(ASSET)
    log(f"Saved {ASSET.name} (trained on all {len(d['gray'])} faces)\n")

    det = DlibLandmarkDetector(upsample=0)
    part_a(d, det, test, reg)
    part_b(d, det, test, reg)
    part_c(d, test, reg, args.n_seq)
    log(f"\n({time.perf_counter() - t0:.0f} s)")
    (OUT / "13_log.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
