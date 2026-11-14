"""Does the v2 frontalizer change accuracy on (frontal) MorphSet faces?

Paired 5-fold x 2 CV grouped by person, PLS(31), 2021 vs v2 frontalization.
Output: outputs/09_frontalizer_scores.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.evaluation import (
    paired_difference,
    repeated_group_kfold,
    score_rows,
)
from facial_expression_analysis.features import PairwiseDistanceFeatures
from facial_expression_analysis.frontalization import Frontalizer

OUT = Path("outputs")
ds = load_morphset()
feats = PairwiseDistanceFeatures()
X = {"2021": feats(Frontalizer.from_2021()(ds.landmarks)), "v2": feats(Frontalizer()(ds.landmarks))}
rows = []
for f in repeated_group_kfold(ds.persons, n_splits=5, n_repeats=2, seed=0):
    for name, Xn in X.items():
        pred = PLSRegression(31).fit(Xn[f.train], ds.targets[f.train]).predict(Xn[f.test])
        rows += score_rows(
            ds.targets[f.test], pred, ds.target_names, model=name, repeat=f.repeat, fold=f.fold
        )
    print(f"repeat {f.repeat} fold {f.fold} done", flush=True)
scores = pd.DataFrame(rows)
scores.to_csv(OUT / "09_frontalizer_scores.csv", index=False)
print(scores.groupby(["model", "target"]).ccc.agg(["mean", "std"]).unstack().round(4).to_string())
print(paired_difference(scores, "v2", "2021").round(4).to_string())
