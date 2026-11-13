"""Dimensional facial expression analysis from 68-point facial landmarks.

Quick start::

    from facial_expression_analysis import EmotionEstimator, DlibLandmarkDetector, load_image

    estimator = EmotionEstimator()
    faces = estimator.analyze_image(load_image("face.jpg"), DlibLandmarkDetector())
    for face in faces:
        print(face.emotion.label, face.emotion.arousal, face.emotion.valence)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("facial-expression-analysis")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0.0.0"

from .features import LandmarkSubset, PairwiseDistanceFeatures
from .frontalization import Frontalizer
from .geometry import procrustes_normalize
from .io import load_image
from .labels import EmotionLabel, describe_emotion
from .landmarks import BoundingBox, DetectedFace, DlibLandmarkDetector, LandmarkDetector
from .model import LinearEmotionModel
from .pipeline import EmotionEstimate, EmotionEstimator, FaceAnalysis

__all__ = [
    "BoundingBox",
    "DetectedFace",
    "DlibLandmarkDetector",
    "EmotionEstimate",
    "EmotionEstimator",
    "EmotionLabel",
    "FaceAnalysis",
    "Frontalizer",
    "LandmarkDetector",
    "LandmarkSubset",
    "LinearEmotionModel",
    "PairwiseDistanceFeatures",
    "__version__",
    "describe_emotion",
    "load_image",
    "procrustes_normalize",
]
