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
- Test suite (78 tests), including equivalence tests that reproduce the v1 outputs to
  float32 precision.
- GitHub Actions CI (lint, mypy, tests on Linux and macOS for Python 3.11–3.13) and
  pre-commit hooks.
- `fea benchmark` for latency per pipeline stage.
- `cross_validate_pls` for grouped k-fold evaluation.

## [1.0.0] - 2021
Original release.
