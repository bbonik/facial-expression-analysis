from __future__ import annotations

import math

import pytest

from facial_expression_analysis.labels import (
    EMOTION_WORDS,
    circumplex_angle,
    describe_emotion,
    emotion_word,
    intensity_qualifier,
)


def polar(angle_deg: float, radius: float) -> tuple[float, float]:
    """Return (arousal, valence) for a point in the circumplex."""
    rad = math.radians(angle_deg)
    return radius * math.sin(rad), radius * math.cos(rad)


def test_24_distinct_words() -> None:
    assert len(EMOTION_WORDS) == 24


@pytest.mark.parametrize(
    ("angle", "word"),
    [
        (0, "pleased"), (15.9, "pleased"), (16, "happy"), (45, "delighted"),
        (90, "astonished"), (100, "aroused"), (135, "afraid"), (180, "miserable"),
        (181, "sad"), (225, "bored"), (270, "sleepy"), (290, "calm"), (345, "satisfied"),
        (353.9, "satisfied"), (354, "pleased"), (359.9, "pleased"),
    ],
)  # fmt: skip
def test_sector_boundaries(angle: float, word: str) -> None:
    # 354 degrees used to return "Unknown" in the 2021 implementation.
    assert emotion_word(angle) == word


@pytest.mark.parametrize(
    ("r", "qualifier"),
    [(0.0, None), (0.099, None), (0.1, "Slightly"), (0.4, "Moderately"), (0.6, "Very"),
     (0.775, "Extremely"), (1.0, "Extremely")],
)  # fmt: skip
def test_intensity_qualifiers(r: float, qualifier: str | None) -> None:
    assert intensity_qualifier(r) == qualifier


def test_angle_convention() -> None:
    assert circumplex_angle(arousal=1, valence=0) == pytest.approx(90)
    assert circumplex_angle(arousal=-1, valence=0) == pytest.approx(270)
    assert circumplex_angle(arousal=0, valence=-1) == pytest.approx(180)


def test_describe_emotion() -> None:
    label = describe_emotion(*polar(45, 0.6))
    assert str(label) == "Very delighted"
    assert label.emotion == "delighted"
    assert not label.is_neutral


def test_neutral_has_no_leading_space() -> None:
    label = describe_emotion(0.01, -0.02)
    assert str(label) == "Neutral"
    assert label.is_neutral


def test_explicit_intensity_overrides_radius() -> None:
    assert describe_emotion(*polar(45, 0.9), intensity=0.2).qualifier == "Slightly"
