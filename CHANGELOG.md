# Changelog

## [Unreleased] - v2.0.0

### Fixed
- **Feature template indexing.** The 2021 Python code was ported from a MATLAB system built
  on the 49-point IntraFace landmarks, whose arrays contain no jaw points. Its feature template
  kept IntraFace-style *positions* (0–50) while indexing the 68-point dlib array. As a result
  the "no jaw" features used the jaw, dropped 17 of the 20 mouth landmarks, and were normalised
  by the scale of a different landmark set. Fixing it improves CCC by +0.021 (arousal),
  +0.009 (valence) and +0.019 (intensity) in 5-fold person-grouped CV, better in every fold.
  Excluding the jaw remains the right choice: adding it costs 0.011 intensity CCC.
- **Identity leakage in splits.** MorphSet's `Subject` column identifies photo series, not
  people: RaFD gaze directions (3 ids per person), KDEF sessions (2) and FACES picture sets (2)
  each got their own id, so the 711 "subjects" are 337 people. This was established from
  neutral-face geometry and matches the source databases. All splits are now made by person.
- The split dropped the highest subject id (`range(1, max)`).
- Procrustes rotation raised `NameError` when the eye line was vertical (`dx == 0`); it now
  uses `atan2`.
- An angle of exactly 354° produced the label "Unknown".
- The neutral label had a leading space (`" Neutral"`).
- Model and weight loading failed silently and crashed later; errors are now raised
  immediately.
- The model file name (`pls=30`) did not match the training configuration (31 components).
- Features were stored as float16, losing precision.
- The webcam demo could not be closed with `q`.

### Changed
- **New default model: RBF kernel ridge** (2000-face Nyström basis, 2.1 MB, 0.33 ms per face).
  Chosen from seven regressors evaluated on five axes (`experiments/08_model_comparison.py`).
  Against the 2021-style PLS(31) it gains +0.086 / +0.043 / +0.053 CCC (A/V/I) in 5-fold
  person-grouped CV, winning every fold, and +0.057 / +0.029 / +0.055 leaving one database
  out. It has 22% less valence jitter under landmark noise. On real scream faces of unseen
  people, arousal rises in 95% of cases, against 74% for PLS(31). PLS(31) is still bundled
  (`emotion_model_linear.npz`, `LinearEmotionModel.load()`), is the most pose-robust, and is
  the model to use for interpretation. See [MODEL_CARD.md](MODEL_CARD.md).
- Intensity is predicted directly by the regressor. It is no longer derived as
  `hypot(arousal, valence)`, which cost 0.058 CCC. (MorphSet labels satisfy that identity
  exactly, but the predictions are shrunk towards the mean.)
- **Frontalization retrained.** The 2021 training mirrored shapes without swapping left/right
  landmark ids. Without mirroring, landmark error drops 9% at every yaw, and arousal CCC on
  MorphSet rises by 0.011 (better in 10/10 folds). The 2021 weights remain available as
  `Frontalizer.from_2021()`.
- Model file format v3 records the model kind (linear or kernel); `load_model()` reads both,
  and v1/v2 files still load.
- Features are normalised by a face scale estimated from expression-stable landmarks (nose
  and eye corners) by default. The scale used in 2021 (all non-jaw landmarks) is inflated by
  expressions (surprise: +5.6%, against +0.4% for the stable set), which mixed mouth and
  eyebrow movements into every feature. It is still available via `--scale subset`.
- New `intraface49` landmark subset reproduces the 49-point scheme of the original MATLAB
  system. It performs on par with the 51 non-jaw landmarks.
- The data loader exposes `person` ids and a `mirrored` flag (every image has a horizontally
  mirrored copy, with landmarks re-detected by dlib).
- Model files store the scale landmarks (format v2); v1 files still load.
- Rewritten as an installable, typed package (`src/facial_expression_analysis`) with a `fea`
  CLI. Python ≥ 3.11 and current numpy / scikit-learn / OpenCV; managed with `uv`.
- Emotion models are stored as portable linear weights (`.npz` + JSON metadata, 31 KB). The
  56 MB scikit-learn 0.23 pickle is gone, and inference no longer needs scikit-learn.
- Feature extraction and frontalization are vectorised: 166k faces take 3 s, versus 24 s for
  the original per-face loop.
- Large files are no longer tracked: the dlib model downloads on first use (hash-verified).
- Detection is behind a `LandmarkDetector` interface, ready for more back-ends.

### Added
- Test suite (108 tests), including equivalence tests that reproduce the v1 outputs to
  float32 precision.
- GitHub Actions CI (lint, mypy, tests on Linux and macOS for Python 3.11–3.13) and
  pre-commit hooks.
- `fea benchmark` for latency per pipeline stage.
- `cross_validate_pls` for grouped k-fold evaluation.
- `evaluation.py`: repeated grouped k-fold, leave-one-group-out, Nadeau–Bengio corrected
  paired tests, person-level bootstrap CIs, and one-fit regularisation paths for PLS and ridge.
- `fea train --estimator kernel_ridge|pls` and `fea train-frontalization`. The multi-pose
  frontalization data is now a pinned, hash-verified download; retraining reproduces the 2021
  weights to 3e-13.
- `KernelEmotionModel.basis_distance`: distance to the nearest training face, a reliability
  signal for unusual faces or bad landmark fits. Validation quantiles are stored in the model
  metadata.
- `interpret.py`: VIP scores, region shares, and the facial movement encoded by each PLS
  component. 17 of 31 components are localised movements resembling action units.
- `experiments/01–11`: reproducible analyses behind every claim, with figures in
  `reports/figures/`.

## [1.0.0] - 2021
Original release.
