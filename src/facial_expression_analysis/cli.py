"""Command-line interface: ``fea --help``."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .features import LandmarkSubset

app = typer.Typer(
    help="Dimensional facial expression analysis (arousal, valence, intensity).",
    no_args_is_help=True,
    add_completion=False,
)

ModelOpt = Annotated[
    Path | None, typer.Option("--model", "-m", help="Model .npz (default: bundled model).")
]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def download(
    dataset: Annotated[bool, typer.Option(help="Also fetch the MorphSet dataset.")] = False,
) -> None:
    """Download and cache the dlib landmark model (and optionally the dataset)."""
    from .assets import fetch_dlib_predictor, fetch_morphset

    typer.echo(f"dlib predictor: {fetch_dlib_predictor()}")
    if dataset:
        typer.echo(f"MorphSet:       {fetch_morphset()}")


@app.command()
def predict(
    images: Annotated[list[Path], typer.Argument(exists=True, dir_okay=False)],
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", "-o", help="Save annotated images here.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON to stdout.")] = False,
    upsample: Annotated[int, typer.Option(help="dlib upsampling (finds small faces).")] = 1,
    model: ModelOpt = None,
) -> None:
    """Estimate the emotions of every face in one or more images."""
    from .io import load_image, save_image
    from .landmarks import DlibLandmarkDetector
    from .model import LinearEmotionModel
    from .pipeline import EmotionEstimator
    from .viz import annotate_image

    estimator = EmotionEstimator(LinearEmotionModel.load(model))
    detector = DlibLandmarkDetector(upsample=upsample)
    results = {}
    for path in images:
        image = load_image(path)
        faces = estimator.analyze_image(image, detector)
        results[str(path)] = [f.to_dict() for f in faces]
        if not as_json:
            typer.secho(f"{path}: {len(faces)} face(s)", bold=True)
            for k, f in enumerate(faces, 1):
                e = f.emotion
                typer.echo(
                    f"  [{k}] {e.label!s:<24} arousal={e.arousal:+.3f} "
                    f"valence={e.valence:+.3f} intensity={e.intensity:.3f}"
                )
        if output_dir is not None:
            save_image(output_dir / f"{path.stem}_annotated.png", annotate_image(image, faces))
    if as_json:
        typer.echo(json.dumps(results, indent=2))


@app.command()
def train(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/emotion_model.npz"),
    components: Annotated[int, typer.Option(min=1)] = 31,
    subset: Annotated[LandmarkSubset, typer.Option()] = LandmarkSubset.NO_JAW,
    seed: Annotated[int, typer.Option()] = 1,
    data: Annotated[Path | None, typer.Option(help="Path to Morphset.csv.zip.")] = None,
) -> None:
    """Train a PLS emotion model on MorphSet with a subject-wise split."""
    from .data import load_morphset
    from .training import TrainConfig, train_pls

    dataset = load_morphset(data)
    result = train_pls(dataset, TrainConfig(n_components=components, subset=subset, seed=seed))
    result.model.save(output)
    output.with_suffix(".metrics.json").write_text(json.dumps(result.metrics, indent=2))
    _print_metrics(result.metrics)
    typer.echo(f"Timings: {json.dumps({k: round(v, 2) for k, v in result.timings.items()})}")
    typer.echo(f"Saved model to {output}")


@app.command()
def evaluate(
    model: ModelOpt = None,
    seed: Annotated[int, typer.Option(help="Seed of the subject split.")] = 1,
    data: Annotated[Path | None, typer.Option(help="Path to Morphset.csv.zip.")] = None,
) -> None:
    """Evaluate a model on the held-out test subjects of MorphSet."""
    from .data import load_morphset, split_subjects
    from .model import LinearEmotionModel
    from .training import evaluate_model

    dataset = load_morphset(data)
    _, val, test = split_subjects(dataset.subjects, seed=seed)
    m = LinearEmotionModel.load(model)
    _print_metrics(
        {
            "validation": evaluate_model(m, dataset.subset(val)),
            "test": evaluate_model(m, dataset.subset(test)),
        }
    )


@app.command()
def benchmark(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    repeats: Annotated[int, typer.Option(min=1)] = 50,
    upsample: Annotated[int, typer.Option()] = 0,
) -> None:
    """Time each pipeline stage on an image (median over repeats)."""
    from .features import PairwiseDistanceFeatures
    from .frontalization import Frontalizer
    from .io import load_image
    from .labels import describe_emotion
    from .landmarks import DlibLandmarkDetector
    from .model import LinearEmotionModel

    rgb = load_image(image)
    detector = DlibLandmarkDetector(upsample=upsample)
    model = LinearEmotionModel.load()
    frontalizer, features = Frontalizer(), model.features
    assert isinstance(features, PairwiseDistanceFeatures)

    faces = detector.detect(rgb)
    if not faces:
        raise typer.BadParameter("No face detected in the image")
    lm = faces[0].landmarks

    def timeit(fn: object) -> float:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()  # type: ignore[operator]
            times.append(time.perf_counter() - t0)
        return float(np.median(times)) * 1e3

    front = frontalizer(lm)
    feats = features(front)
    pred = model.predict_features(feats[None])[0]
    stages = {
        "detect + landmarks (dlib)": timeit(lambda: detector.detect(rgb)),
        "frontalization": timeit(lambda: frontalizer(lm)),
        "features": timeit(lambda: features(front)),
        "regression": timeit(lambda: model.predict_features(feats[None])),
        "label": timeit(lambda: describe_emotion(pred[0], pred[1])),
    }
    h, w = rgb.shape[:2]
    typer.echo(f"Image {w}x{h}, {len(faces)} face(s), median of {repeats} runs:")
    for name, ms in stages.items():
        typer.echo(f"  {name:<28}{ms:9.3f} ms")
    typer.echo(f"  {'total':<28}{sum(stages.values()):9.3f} ms")


@app.command()
def webcam(
    camera: Annotated[int, typer.Option(help="OpenCV camera index.")] = 0,
    smoothing: Annotated[float, typer.Option(min=0, max=0.99, help="EMA smoothing.")] = 0.3,
    model: ModelOpt = None,
) -> None:
    """Live emotion overlay from a webcam (press q or Esc to quit)."""
    from .live import run_webcam

    run_webcam(camera=camera, smoothing=smoothing, model_path=model)


def _print_metrics(metrics: dict[str, dict]) -> None:  # type: ignore[type-arg]
    for split, res in metrics.items():
        typer.secho(f"\n{split}: {res['n_samples']} faces, {res['n_subjects']} subjects", bold=True)
        for view in ("regressor", "pipeline"):
            typer.echo(f"  {view}:")
            for target, m in res[view].items():
                typer.echo(
                    f"    {target:<10} CCC={m['ccc']:.4f}  r={m['pearson']:.4f}  "
                    f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
                )


if __name__ == "__main__":
    app()
