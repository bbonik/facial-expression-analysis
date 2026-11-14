import numpy as np
import pandas as pd

from facial_expression_analysis.data import load_morphset

ds = load_morphset()
m = ds.meta
stem = m.filename.str.replace(r"_[12]\.jpg$", "", regex=True)
k = m.filename.str.extract(r"_([12])\.jpg$")[0]
i1 = pd.Series(m.index[k == "1"], index=stem[k == "1"])
i2 = pd.Series(m.index[k == "2"], index=stem[k == "2"])
pairs = pd.concat([i1, i2], axis=1, keys=["a", "b"]).dropna().astype(int)
print(
    "pairs:",
    len(pairs),
    " same targets in pair:",
    np.allclose(ds.targets[pairs.a], ds.targets[pairs.b]),
    " same subject:",
    (m.subject.values[pairs.a] == m.subject.values[pairs.b]).all(),
)
A, B = ds.landmarks[pairs.a.values], ds.landmarks[pairs.b.values]
PERM = np.array(
    list(range(16, -1, -1))
    + list(range(26, 16, -1))
    + [27, 28, 29, 30]
    + [35, 34, 33, 32, 31]
    + [45, 44, 43, 42, 47, 46]
    + [39, 38, 37, 36, 41, 40]
    + [54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55]
    + [64, 63, 62, 61, 60, 67, 66, 65]
)
assert sorted(PERM) == list(range(68))
assert (PERM[PERM] == np.arange(68)).all()


def fit_err(B_pred_fn):
    # best image width W per pair: x_b = W - x_a  -> W = mean(x_b + x_a')
    P = B_pred_fn(A)  # A with optional permutation, x negated
    W = (B[:, :, 0] - P[:, :, 0]).mean(1, keepdims=True)
    dx = B[:, :, 0] - (P[:, :, 0] + W)
    dy = B[:, :, 1] - P[:, :, 1]
    return np.sqrt(dx**2 + dy**2).mean(1), W[:, 0]


err_same = np.sqrt(((B - A) ** 2).sum(-1)).mean(1)
err_flip, W1 = fit_err(lambda X: np.stack([-X[:, :, 0], X[:, :, 1]], -1))
err_flip_perm, W2 = fit_err(lambda X: np.stack([-X[:, PERM, 0], X[:, PERM, 1]], -1))
for name, e in [
    ("identical copy", err_same),
    ("mirror, x negated, NO index swap", err_flip),
    ("mirror WITH left/right index swap", err_flip_perm),
]:
    print(
        f"{name:<36} mean landmark err px: median={np.median(e):.2f}  90%={np.quantile(e, 0.9):.2f}"
    )
print(
    "implied image width (perm hypothesis): median",
    np.median(W2).round(1),
    " IQR",
    np.quantile(W2, [0.25, 0.75]).round(1),
)
best = np.argmin(np.stack([err_same, err_flip, err_flip_perm]), 0)
print("best hypothesis per pair:", np.bincount(best, minlength=3), "(same, flip-noswap, flip-swap)")
# Which copy is the original? Check left-eye centre x < right-eye centre x (dlib: 36-41 is image-left)
for tag, X in [("_1", A), ("_2", B)]:
    ok = (X[:, 36:42, 0].mean(1) < X[:, 42:48, 0].mean(1)).mean()
    print(f"copy {tag}: eye 36-41 is left of eye 42-47 in {ok:.1%} of faces")
# residual for perm hypothesis: is it ~0 (pure numeric flip+swap) or detector noise (re-detected)?
print(
    "perm-hypothesis residual quantiles px:",
    np.quantile(err_flip_perm, [0.1, 0.5, 0.9, 0.99]).round(2),
)
