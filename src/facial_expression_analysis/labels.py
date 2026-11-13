"""Map arousal-valence coordinates to verbal emotion labels.

Labels follow Russell's circumplex model of affect: the angle in the
valence-arousal plane selects one of 24 emotion words and the radius selects an
intensity qualifier.

Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and
Social Psychology*, 39(6), 1161-1178.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Final

NEUTRAL_THRESHOLD: Final = 0.1

# Upper bounds (exclusive) of each intensity qualifier.
_INTENSITY_BOUNDS: Final = (0.325, 0.55, 0.775)
_INTENSITY_NAMES: Final = ("Slightly", "Moderately", "Very", "Extremely")

# Upper bounds (exclusive, degrees counter-clockwise from +valence) of each sector.
_ANGLE_BOUNDS: Final = (
    16, 34, 62.5, 78.5, 93, 104, 115, 126, 137, 148, 159, 170,
    181, 192, 203, 215, 230, 245, 260, 280, 300, 320, 340, 354,
)  # fmt: skip
_EMOTION_NAMES: Final = (
    # quadrant I: positive valence, high arousal
    "pleased", "happy", "delighted", "excited", "astonished",
    # quadrant II: negative valence, high arousal
    "aroused", "tensed", "alarmed", "afraid", "annoyed", "distressed", "frustrated",
    "miserable",
    # quadrant III: negative valence, low arousal
    "sad", "gloomy", "depressed", "bored", "droopy", "tired", "sleepy",
    # quadrant IV: positive valence, low arousal
    "calm", "serene", "content", "satisfied",
    "pleased",  # wrap-around: [354, 360)
)  # fmt: skip

EMOTION_WORDS: Final = tuple(dict.fromkeys(_EMOTION_NAMES))
"""The 24 distinct emotion words, in counter-clockwise order."""


@dataclass(frozen=True, slots=True)
class EmotionLabel:
    """Verbal description of a point in the arousal-valence plane."""

    emotion: str
    """Emotion word, e.g. ``"happy"``, or ``"neutral"``."""
    qualifier: str | None
    """Intensity qualifier, e.g. ``"Moderately"``; ``None`` for neutral."""
    angle: float
    """Angle in degrees in ``[0, 360)``, counter-clockwise from +valence."""

    @property
    def is_neutral(self) -> bool:
        return self.qualifier is None

    def __str__(self) -> str:
        return "Neutral" if self.qualifier is None else f"{self.qualifier} {self.emotion}"


def circumplex_angle(arousal: float, valence: float) -> float:
    """Angle of ``(valence, arousal)`` in degrees in ``[0, 360)``."""
    return math.degrees(math.atan2(arousal, valence)) % 360.0


def intensity_qualifier(intensity: float) -> str | None:
    """Return the qualifier for an intensity in ``[0, 1]``; ``None`` if neutral."""
    if intensity < NEUTRAL_THRESHOLD:
        return None
    return _INTENSITY_NAMES[bisect_right(_INTENSITY_BOUNDS, intensity)]


def emotion_word(angle: float) -> str:
    """Return the circumplex emotion word for an angle in degrees."""
    return _EMOTION_NAMES[bisect_right(_ANGLE_BOUNDS, angle % 360.0)]


def describe_emotion(
    arousal: float, valence: float, intensity: float | None = None
) -> EmotionLabel:
    """Describe an arousal-valence point with a verbal label.

    Parameters
    ----------
    arousal, valence
        Coordinates in ``[-1, 1]``.
    intensity
        Expression intensity in ``[0, 1]``. Defaults to the distance from the
        origin, ``sqrt(arousal**2 + valence**2)``.
    """
    if intensity is None:
        intensity = math.hypot(arousal, valence)
    angle = circumplex_angle(arousal, valence)
    qualifier = intensity_qualifier(intensity)
    word = "neutral" if qualifier is None else emotion_word(angle)
    return EmotionLabel(emotion=word, qualifier=qualifier, angle=angle)
