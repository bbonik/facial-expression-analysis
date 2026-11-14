"""Are MorphSet subject IDs unique people? Compare neutral-face geometry between IDs."""

from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.geometry import procrustes_normalize

ds = load_morphset()
m = ds.meta
print("NE rows per subject:", Counter(m[m.label == "NE"].groupby("subject").size()).most_common(3))
print("NE filenames example:", m[m.label == "NE"].filename.head(4).tolist())
ne = m[(m.label == "NE") & m.filename.str.endswith("_1.jpg")].drop_duplicates("subject")
shapes = procrustes_normalize(ds.landmarks[ne.index.to_numpy()]).reshape(len(ne), -1)
for d in ["R", "K", "W", "F"]:
    mask = (ne.dataset == d).to_numpy()
    ids, X = ne.subject.to_numpy()[mask], shapes[mask]
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    nnd = D.min(1)
    # distance scale: median distance between random different IDs
    typical = np.median(D[np.isfinite(D)])
    offs = Counter(ids[nn] - ids).most_common(6)
    print(
        f"\n[{d}] {mask.sum()} IDs  ids {ids.min()}-{ids.max()}  NN dist median={np.median(nnd):.4f}  "
        f"typical pair dist={typical:.4f}  ratio={np.median(nnd) / typical:.3f}"
    )
    print(f"     most common ID offset to nearest neighbour: {offs}")
    print(f"     NN dist quantiles 10/50/90%: {np.quantile(nnd, [0.1, 0.5, 0.9]).round(4)}")

print(
    "\n=== Validate grouping hypothesis: R triplets from 1, K pairs from 202, F pairs from 371, W singletons"
)
START = {"R": (1, 3), "K": (202, 2), "F": (371, 2), "W": (342, 1)}
all_ids = set(m.subject.unique())
for d, (s, k) in START.items():
    mask = (ne.dataset == d).to_numpy()
    ids, X = ne.subject.to_numpy()[mask], shapes[mask]
    grp = (ids - s) // k
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    if k > 1:
        same = (grp[nn] == grp).mean()
        within = np.array(
            [
                D[i, j]
                for i in range(len(ids))
                for j in range(len(ids))
                if i < j and grp[i] == grp[j]
            ]
        )
        between_nn = np.array([D[i][grp != grp[i]].min() for i in range(len(ids))])
        print(
            f"[{d}] groups={len(set(grp))}  NN in same group: {same:.1%}  within-group dist median={np.median(within):.3f} "
            f"max={within.max():.3f} | nearest other-group median={np.median(between_nn):.3f} min={between_nn.min():.3f}"
        )
    missing = sorted(set(range(s, ids.max() + 1)) - all_ids)
    sizes = Counter(Counter(grp).values())
    print(f"     missing IDs in range: {missing}  group sizes: {dict(sizes)}")
