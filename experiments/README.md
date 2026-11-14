# Experiments

Standalone analyses behind the claims in the main README and CHANGELOG. Run them from the
repository root, e.g. `uv run python experiments/05_feature_config_cv.py`.

| Script | Question | Finding |
|---|---|---|
| `01_person_identity.py` | Are MorphSet subject ids unique people? | No: RaFD 3, KDEF 2, FACES 2 ids per person → 337 people |
| `02_mirrored_copies.py` | What are the `_1` / `_2` file pairs? | `_2` is a horizontal mirror, landmarks re-detected by dlib (1.8 px residual) |
| `03_viewpoint_and_outliers.py` | Head pose and landmark failures | All frontal, eye-aligned (IOD ≈ 150 px); 12 detection failures (0.01%) |
| `04_scale_bias.py` | Do expressions bias the scale estimate? | Yes for mouth/brow-inclusive scales (surprise +5.6%); stable points +0.4% |
| `05_feature_config_cv.py` | Landmark set × scale reference (5-fold CV by person) | Template fix +0.02 CCC; jaw hurts intensity; 49 ≈ 51 points |

These will be consolidated into the Phase 3 analysis notebooks.
