"""Grouped 5-fold CV (by person): landmark set x scale reference, PLS(31)."""

import json
import time
from pathlib import Path

import numpy as np

from facial_expression_analysis import Frontalizer, PairwiseDistanceFeatures
from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.training import cross_validate_pls

ds = load_morphset()
F = Frontalizer()(ds.landmarks)
configs = {
    "v1 shipped (pairs 0-50, scale 17-67)": PairwiseDistanceFeatures(
        landmark_indices=np.arange(51), scale=np.arange(17, 68)
    ),
    "no_jaw-51 / subset scale": PairwiseDistanceFeatures("no_jaw", scale="subset"),
    "no_jaw-51 / stable scale": PairwiseDistanceFeatures("no_jaw", scale="stable"),
    "intraface-49 / subset scale": PairwiseDistanceFeatures("intraface49", scale="subset"),
    "intraface-49 / stable scale": PairwiseDistanceFeatures("intraface49", scale="stable"),
    "all-68 / subset scale": PairwiseDistanceFeatures("all", scale="subset"),
    "all-68 / stable scale": PairwiseDistanceFeatures("all", scale="stable"),
}
results = {}
for name, feat in configs.items():
    t = time.time()
    X = feat(F)
    s = cross_validate_pls(X, ds.targets, ds.persons)
    results[name] = s.tolist()
    m, sd = s.mean(0), s.std(0, ddof=1)
    print(
        f"{name:<40} A={m[0]:.4f}±{sd[0]:.4f}  V={m[1]:.4f}±{sd[1]:.4f}  I={m[2]:.4f}±{sd[2]:.4f}  ({time.time() - t:.0f}s)",
        flush=True,
    )
# leakage: same config grouped by CSV subject id instead of person
X = configs["no_jaw-51 / stable scale"](F)
s = cross_validate_pls(X, ds.targets, ds.subjects)
results["no_jaw-51 / stable, grouped by SUBJECT (leaky)"] = s.tolist()
m, sd = s.mean(0), s.std(0, ddof=1)
print(
    f"{'no_jaw-51 / stable, grouped by SUBJECT':<40} A={m[0]:.4f}±{sd[0]:.4f}  V={m[1]:.4f}±{sd[1]:.4f}  I={m[2]:.4f}±{sd[2]:.4f}",
    flush=True,
)
out = Path("outputs/cv_results.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=1))
