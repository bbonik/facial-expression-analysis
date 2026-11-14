"""Does facial expression bias the scale estimate? And which feature set was intended?"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from facial_expression_analysis import Frontalizer, PairwiseDistanceFeatures
from facial_expression_analysis.data import load_morphset, split_by_group
from facial_expression_analysis.geometry import shape_scale
from facial_expression_analysis.metrics import ccc

ds = load_morphset()
L = ds.landmarks
eye_c = lambda X: np.linalg.norm(X[:, 36:42].mean(1) - X[:, 42:48].mean(1), axis=1)
SCALES = {
    "all 68 (Procrustes)": lambda X: shape_scale(X),
    "17-67 (v1 feature scale)": lambda X: shape_scale(X[:, 17:]),
    "0-50 (v1 pairs)": lambda X: shape_scale(X[:, :51]),
    "stable: eye corners + nose": lambda X: shape_scale(
        X[:, [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45]]
    ),
    "interocular (eye centres)": eye_c,
}
print("1) Within-subject variation of the scale estimate in raw (eye-aligned) images")
print("   ratio = scale(face) / scale(subject's neutral face); ideal = 1.000 for every face\n")
df = ds.meta[["subject", "label"]].copy()
rows = []
for name, fn in SCALES.items():
    s = fn(L)
    df["s"] = s
    neutral = df[df.label == "NE"].groupby("subject").s.mean()
    r = df.s / df.subject.map(neutral)
    by = r.groupby(df.label, observed=True).mean()
    rows.append(
        {
            "scale": name,
            "CV within subject %": 100
            * (df.groupby("subject").s.std() / df.groupby("subject").s.mean()).mean(),
            "SU (open mouth)": by["SU"],
            "HA": by["HA"],
            "AF": by["AF"],
            "SA": by["SA"],
            "max |ratio-1| %": 100 * (r - 1).abs().max(),
        }
    )
print(pd.DataFrame(rows).set_index("scale").round(3).to_string())

print("\n2) Same split, PLS(31): feature set x feature-normalisation scale (test CCC)")
tr, va, te = split_by_group(ds.persons)
F = Frontalizer()(L)


def run(pairs_idx, scale_fn, label):
    X = PairwiseDistanceFeatures(landmark_indices=pairs_idx)(F)
    X *= (shape_scale(F[:, pairs_idx]) / scale_fn(F))[:, None].astype(np.float32)  # re-normalise
    p = PLSRegression(31).fit(X[tr], ds.targets[tr]).predict(X[te])
    c = [ccc(ds.targets[te][:, k], p[:, k]) for k in range(3)]
    print(f"   {label:<58} A={c[0]:.4f} V={c[1]:.4f} I={c[2]:.4f}")


no_jaw, v1 = np.arange(17, 68), np.arange(51)
intraface49 = np.array([i for i in range(17, 68) if i not in (60, 64)])
run(v1, SCALES["17-67 (v1 feature scale)"], "v1 as shipped: pairs 0-50, scale 17-67")
run(no_jaw, SCALES["17-67 (v1 feature scale)"], "pairs 17-67 (docstring intent), scale 17-67")
run(
    intraface49,
    lambda X: shape_scale(X[:, intraface49]),
    "49-pt IntraFace set (2021 paper), own scale",
)
run(no_jaw, SCALES["stable: eye corners + nose"], "pairs 17-67, scale = stable eye corners + nose")
run(no_jaw, SCALES["interocular (eye centres)"], "pairs 17-67, scale = interocular distance")
run(v1, SCALES["stable: eye corners + nose"], "pairs 0-50, scale = stable eye corners + nose")
