"""The refactored pipeline must reproduce the original 2021 implementation.

The legacy model used a buggy feature template (pairs of landmarks 0..50,
normalised by the scale of landmarks 17..67). Replicating that quirk here, the
new vectorised code reproduces the old outputs to float32 precision, which
validates the refactor independently of the bug fix.
"""

from __future__ import annotations

import numpy as np

from facial_expression_analysis import Frontalizer, PairwiseDistanceFeatures, describe_emotion
from facial_expression_analysis.geometry import shape_scale


def legacy_features(frontal: np.ndarray) -> np.ndarray:
    feats = PairwiseDistanceFeatures(landmark_indices=np.arange(51))(frontal)
    ratio = shape_scale(frontal[:, :51]) / shape_scale(frontal[:, 17:])
    return feats * ratio[:, None]


def test_frontalization_matches_legacy(legacy_reference: dict[str, np.ndarray]) -> None:
    frontal = Frontalizer()(legacy_reference["landmarks"])
    np.testing.assert_allclose(frontal, legacy_reference["frontal"], atol=1e-12)


def test_predictions_and_labels_match_legacy(
    legacy_reference: dict[str, np.ndarray], legacy_weights: dict[str, np.ndarray]
) -> None:
    frontal = Frontalizer()(legacy_reference["landmarks"])
    raw = legacy_features(frontal) @ legacy_weights["coef"] + legacy_weights["intercept"]
    np.testing.assert_allclose(raw, legacy_reference["raw_predictions"], atol=1e-5)

    arousal = np.clip(raw[:, 0], -1, 1).round(3)
    valence = np.clip(raw[:, 1], -1, 1).round(3)
    labels = [str(describe_emotion(a, v)) for a, v in zip(arousal, valence, strict=True)]
    assert labels == [str(s).strip() for s in legacy_reference["labels"]]
