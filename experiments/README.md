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
| `06_linear_model_selection.py` | PLS components / ridge penalty (nested CV); intensity from A, V? | One-SE PLS k ≈ 50, ridge better; regressing intensity beats `hypot(A, V)` by 0.058 |
| `07_frontalization.py` | Reproduce and improve the frontalization model | 2021 weights reproduced (3e-13); no mirroring −9% error; frontalization cuts the 45° arousal shift 0.54 → 0.11 |
| `08_model_comparison.py` | Seven regressors on five axes (CV, leave-one-database-out, jitter, pose, external) | Kernel ridge best on accuracy, generalisation, valence jitter and scream; PLS(31) most pose-robust; PLS + residual kernel hybrid not better |
| `09_frontalizer_on_morphset.py` | v2 vs 2021 frontalizer on MorphSet | Arousal +0.011 CCC, 10/10 folds |
| `10_error_analysis.py` | Where does the model fail? (`--model kernel:0.5,1`) | Shrinkage (slope 0.61 A); sad too aroused, anger/fear not aroused enough; FACES hardest |
| `11_interpretability.py` | What does PLS look at? Are components action units? | Inner mouth corners (48–60, 54–64) top VIP; 17/31 components localised; arousal ≈ mouth opening |

Experiments 06–08 take 10–60 minutes; the others take under 2 minutes.
