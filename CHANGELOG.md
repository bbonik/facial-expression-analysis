# Changelog

## [Unreleased] - v2.0.0

### Fixed
- **Feature template indexing:** with `full_features=False` the 2021 code built distance pairs
  from landmark *positions* 0–50 rather than landmark *ids* 17–67. It kept the jaw, dropped
  17 of the 20 mouth landmarks, and normalised by the scale of a different landmark set.
  Retraining with the corrected features improves test CCC by +0.017 (arousal), +0.015
  (valence) and +0.014 (intensity) on an identical subject split.
- The subject split dropped the highest subject id (`range(1, max)`).
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
- Rewritten as an installable, typed package (`src/facial_expression_analysis`) with a `fea`
  CLI. Python ≥ 3.11 and current numpy / scikit-learn / OpenCV; managed with `uv`.
- Emotion models are stored as portable linear weights (`.npz` + JSON metadata, 31 KB). The
  56 MB scikit-learn 0.23 pickle is gone, and inference no longer needs scikit-learn.
- Feature extraction and frontalization are vectorised: 166k faces take 3 s, versus 24 s for
  the original per-face loop.
- Large files are no longer tracked: the dlib model downloads on first use (hash-verified).
- Detection is behind a `LandmarkDetector` interface, ready for more back-ends.

### Added
- Test suite (69 tests), including equivalence tests that reproduce the v1 outputs to
  float32 precision.
- GitHub Actions CI (lint, mypy, tests on Linux and macOS for Python 3.11–3.13) and
  pre-commit hooks.
- `fea benchmark` for latency per pipeline stage.

## [1.0.0] - 2021
Original release.
