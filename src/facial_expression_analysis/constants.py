"""Constants describing the 68-point iBUG-300W / dlib facial landmark scheme."""

from __future__ import annotations

from typing import Final

N_LANDMARKS: Final = 68

# Index ranges (half-open) of each facial part in the 68-point scheme.
JAW: Final = range(0, 17)
LEFT_BROW: Final = range(17, 22)
RIGHT_BROW: Final = range(22, 27)
NOSE_BRIDGE: Final = range(27, 31)
NOSE_BASE: Final = range(31, 36)
LEFT_EYE: Final = range(36, 42)
RIGHT_EYE: Final = range(42, 48)
OUTER_MOUTH: Final = range(48, 60)
INNER_MOUTH: Final = range(60, 68)

# Poly-lines used to draw a face; ``True`` marks closed contours.
FACE_CONTOURS: Final[tuple[tuple[range, bool], ...]] = (
    (JAW, False),
    (LEFT_BROW, False),
    (RIGHT_BROW, False),
    (NOSE_BRIDGE, False),
    (NOSE_BASE, False),
    (LEFT_EYE, True),
    (RIGHT_EYE, True),
    (OUTER_MOUTH, True),
    (INNER_MOUTH, True),
)

# The 49-point IntraFace scheme (used by the original MATLAB system) expressed
# in dlib ids: every non-jaw landmark except the inner-mouth corners 60 and 64,
# which nearly coincide with the outer-mouth corners 48 and 54.
INTRAFACE_49: Final = tuple(i for i in range(JAW.stop, N_LANDMARKS) if i not in (60, 64))

# Landmarks that barely move with facial expressions: the nose and the eye
# corners. Used as the reference for scale normalisation, so that an open mouth
# or raised eyebrows do not inflate the scale estimate.
STABLE_POINTS: Final = (27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 42, 45)

# Index permutation that maps a horizontally mirrored face back to the dlib
# convention (left and right parts swap roles): mirrored[i] = flipped[PERM[i]].
MIRROR_PERMUTATION: Final = (
    *range(16, -1, -1),  # jaw
    *range(26, 16, -1),  # eyebrows
    27, 28, 29, 30,  # nose bridge
    35, 34, 33, 32, 31,  # nose base
    45, 44, 43, 42, 47, 46,  # left eye <- right eye
    39, 38, 37, 36, 41, 40,  # right eye <- left eye
    54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,  # outer mouth
    64, 63, 62, 61, 60, 67, 66, 65,  # inner mouth
)  # fmt: skip
