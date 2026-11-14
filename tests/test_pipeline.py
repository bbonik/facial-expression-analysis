from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from facial_expression_analysis import EmotionEstimator, FaceAnalysis
from facial_expression_analysis.landmarks import BoundingBox, DetectedFace
from facial_expression_analysis.pipeline import largest_face

from .conftest import IMAGES, random_similarity, requires_dlib_model


@pytest.fixture(scope="module")
def estimator() -> EmotionEstimator:
    return EmotionEstimator()


def test_outputs_are_in_range(estimator: EmotionEstimator, landmarks: np.ndarray) -> None:
    avi = estimator.predict(landmarks)
    assert avi.shape == (len(landmarks), 3)
    assert np.all(np.abs(avi[:, :2]) <= 1)
    assert np.all((avi[:, 2] >= 0) & (avi[:, 2] <= 1))


def test_estimate_matches_batch(estimator: EmotionEstimator, landmarks: np.ndarray) -> None:
    single = estimator.estimate(landmarks[0])
    batch = estimator.predict(landmarks[:1])[0]
    assert (single.arousal, single.valence, single.intensity) == pytest.approx(tuple(batch))
    assert single.frontal_landmarks.shape == (68, 2)


def test_estimate_rejects_batches(estimator: EmotionEstimator, landmarks: np.ndarray) -> None:
    with pytest.raises(ValueError, match="single"):
        estimator.estimate(landmarks[:2])


def test_invariant_to_head_position_and_scale(
    estimator: EmotionEstimator, landmarks: np.ndarray, rng: np.random.Generator
) -> None:
    moved = random_similarity(landmarks[0], rng)
    np.testing.assert_allclose(estimator.predict(moved), estimator.predict(landmarks[0]), atol=1e-6)


def test_intensity_defaults_to_regressor_output(landmarks: np.ndarray) -> None:
    est = EmotionEstimator()
    raw = est.model.predict(est.frontalizer(landmarks))
    np.testing.assert_allclose(est.predict(landmarks)[:, 2], np.clip(raw[:, 2], 0, 1))


def test_intensity_from_av_option(landmarks: np.ndarray) -> None:
    avi = EmotionEstimator(intensity_from_av=True).predict(landmarks)
    np.testing.assert_allclose(avi[:, 2], np.minimum(np.hypot(avi[:, 0], avi[:, 1]), 1))


def test_analyze_image_with_stub_detector(
    estimator: EmotionEstimator, landmarks: np.ndarray
) -> None:
    class Stub:
        def detect(self, image: np.ndarray) -> list[DetectedFace]:
            return [
                DetectedFace(BoundingBox(0, 0, 10, 10), landmarks[0]),
                DetectedFace(BoundingBox(0, 0, 50, 40), landmarks[1]),
            ]

    faces = estimator.analyze_image(np.zeros((64, 64, 3), np.uint8), Stub())
    assert len(faces) == 2
    biggest = largest_face(faces)
    assert biggest is not None
    assert biggest.box.area == 2000
    payload = json.dumps([f.to_dict() for f in faces])
    assert "label" in payload


def test_largest_face_of_empty_list() -> None:
    faces: list[FaceAnalysis] = []
    assert largest_face(faces) is None


@pytest.mark.slow
@requires_dlib_model
def test_end_to_end_on_sample_image(estimator: EmotionEstimator) -> None:
    from facial_expression_analysis import DlibLandmarkDetector, load_image

    # Faces with clean landmark fits only: on hard faces (hands on cheeks, a wide-open
    # roaring mouth) dlib misplaces the lip points and errors say little about the model.
    detector = DlibLandmarkDetector()
    assert len(estimator.analyze_image(load_image(IMAGES / "kids.jpg"), detector)) == 2
    (happy,) = estimator.analyze_image(load_image(IMAGES / "happy.jpg"), detector)
    (angry,) = estimator.analyze_image(load_image(IMAGES / "angry.jpg"), detector)
    assert happy.emotion.valence > 0.3
    assert angry.emotion.valence < 0
    assert angry.emotion.arousal > 0.2


@pytest.mark.slow
@requires_dlib_model
def test_cli_predict_json(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from facial_expression_analysis.cli import app

    result = CliRunner().invoke(
        app, ["predict", str(IMAGES / "happy.jpg"), "--json", "-o", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    (faces,) = payload.values()
    assert len(faces) == 1
    assert (tmp_path / "happy_annotated.png").is_file()
