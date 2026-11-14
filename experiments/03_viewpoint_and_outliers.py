import numpy as np
import pandas as pd

from facial_expression_analysis.data import load_morphset
from facial_expression_analysis.geometry import shape_scale

ds = load_morphset()
m = ds.meta.copy()
L = ds.landmarks
le, re_ = L[:, 36:42].mean(1), L[:, 42:48].mean(1)
iod = np.linalg.norm(re_ - le, axis=1)
mid = (le + re_) / 2
m["yaw_proxy"] = (L[:, 30, 0] - mid[:, 0]) / iod  # nose-tip offset from eye midline
m["jaw_asym"] = (
    np.linalg.norm(L[:, 0] - L[:, 30], axis=1) - np.linalg.norm(L[:, 16] - L[:, 30], axis=1)
) / iod
m["iod_px"] = iod
print("Head-pose proxies per dataset (|yaw|>0.25 ~ clearly non-frontal):")
g = m.groupby("dataset", observed=True)
print(
    pd.DataFrame(
        {
            "yaw p1": g.yaw_proxy.quantile(0.01),
            "yaw p50": g.yaw_proxy.median(),
            "yaw p99": g.yaw_proxy.quantile(0.99),
            "% |yaw|>0.25": g.yaw_proxy.apply(lambda s: 100 * (s.abs() > 0.25).mean()),
            "IOD px median": g.iod_px.median(),
        }
    )
    .round(3)
    .to_string()
)
# Scale outliers (yesterday: ~55% deviations) -> inspect
s = shape_scale(L[:, [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45]])
m["s"] = s
ne = m[m.label == "NE"].groupby("subject").s.mean()
m["ratio"] = m.s / m.subject.map(ne)
out = m[(m.ratio - 1).abs() > 0.15]
print(
    f"\nfaces with stable-scale ratio off by >15%: {len(out)} ({100 * len(out) / len(m):.2f}%) from {out.subject.nunique()} subjects"
)
print(out.groupby(["dataset"], observed=True).size().to_string())
print(
    out.sort_values("ratio")
    .head(6)[["filename", "ratio", "yaw_proxy", "iod_px"]]
    .round(3)
    .to_string()
)
# Are they whole subjects (bad neutral) or isolated faces?
per_subj = out.groupby("subject").size().sort_values(ascending=False)
print("\noutlier faces per affected subject (top):", per_subj.head(8).to_dict())
print("faces per subject:", m.groupby("subject").size().median())
