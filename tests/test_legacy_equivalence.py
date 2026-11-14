"""The refactored pipeline must reproduce the original 2021 implementation.

The 2021 code was ported from a MATLAB system built on the 49-point IntraFace
landmarks, whose arrays contain no jaw points. Its feature template kept
IntraFace-style *positions* (0..50) while indexing the full 68-point dlib array,
and normalised by the scale of dlib landmarks 17..67. Configured that way, the
new code reproduces the old outputs to float32 precision, which validates the
refactor independently of the fix.
"""

from __future__ import annotations

import numpy as np

from facial_expression_analysis import (
    EmotionEstimator,
    Frontalizer,
    LinearEmotionModel,
    PairwiseDistanceFeatures,
    describe_emotion,
)

LEGACY_PAIRS = np.arange(51)
LEGACY_SCALE = np.arange(17, 68)


def test_frontalization_matches_legacy(legacy_reference: dict[str, np.ndarray]) -> None:
    frontal = Frontalizer()(legacy_reference["landmarks"])
    np.testing.assert_allclose(frontal, legacy_reference["frontal"], atol=1e-12)


def test_legacy_model_reproduced_exactly(
    legacy_reference: dict[str, np.ndarray], legacy_weights: dict[str, np.ndarray]
) -> None:
    feats = PairwiseDistanceFeatures(landmark_indices=LEGACY_PAIRS, scale=LEGACY_SCALE)
    model = LinearEmotionModel(
        legacy_weights["coef"], legacy_weights["intercept"], LEGACY_PAIRS, LEGACY_SCALE
    )
    assert model.features.n_features == feats.n_features == 1275

    frontal = Frontalizer()(legacy_reference["landmarks"])
    np.testing.assert_allclose(
        model.predict(frontal), legacy_reference["raw_predictions"], atol=1e-5
    )

    # Labels: the legacy code rounded arousal/valence to 3 decimals before
    # labelling, which matters for faces right at an intensity boundary.
    avi = EmotionEstimator(model).predict(legacy_reference["landmarks"]).round(3)
    labels = [str(describe_emotion(a, v)) for a, v, _ in avi]
    assert labels == [str(s).strip() for s in legacy_reference["labels"]]
