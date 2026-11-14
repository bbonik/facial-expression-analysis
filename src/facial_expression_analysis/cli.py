"""Command-line interface: ``fea --help``."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .features import LandmarkSubset, ScaleReference
from .training import Estimator

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
    from .model import load_model
    from .pipeline import EmotionEstimator
    from .viz import annotate_image

    estimator = EmotionEstimator(load_model(model))
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
    estimator: Annotated[
        Estimator,
        typer.Option(help="pls (default, linear) or kernel_ridge (highest offline CCC)."),
    ] = Estimator.PLS,
    components: Annotated[int, typer.Option(min=1, help="PLS components.")] = 31,
    gamma: Annotated[float, typer.Option(help="Kernel width x n_features.")] = 0.5,
    alpha: Annotated[float, typer.Option(help="Kernel ridge penalty.")] = 1.0,
    n_basis: Annotated[int, typer.Option(min=10, help="Nystroem basis size.")] = 2000,
    subset: Annotated[LandmarkSubset, typer.Option()] = LandmarkSubset.NO_JAW,
    scale: Annotated[
        ScaleReference, typer.Option(help="Landmarks used to estimate the face scale.")
    ] = ScaleReference.STABLE,
    seed: Annotated[int, typer.Option()] = 1,
    data: Annotated[Path | None, typer.Option(help="Path to Morphset.csv.zip.")] = None,
) -> None:
    """Train an emotion model on MorphSet with a person-wise split."""
    from .data import load_morphset
    from .training import TrainConfig, train_model

    dataset = load_morphset(data)
    config = TrainConfig(
        estimator=estimator, n_components=components, gamma=gamma, alpha=alpha,
        n_basis=n_basis, subset=subset, scale=scale, seed=seed,
    )  # fmt: skip
    result = train_model(dataset, config)
    result.model.save(output)
    output.with_suffix(".metrics.json").write_text(json.dumps(result.metrics, indent=2))
    _print_metrics(result.metrics)
    typer.echo(f"Timings: {json.dumps({k: round(v, 2) for k, v in result.timings.items()})}")
    typer.echo(f"Saved model to {output}")


@app.command("train-frontalization")
def train_frontalization(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "models/frontalization_dlib68.npy"
    ),
    mirror: Annotated[str, typer.Option(help="Mirroring: none, naive (2021), permute.")] = "none",
    ridge: Annotated[float, typer.Option(min=0.0, help="Per-sample ridge penalty.")] = 0.0,
) -> None:
    """Train the landmark frontalization model on the multi-pose dataset."""
    from .frontalization_training import (
        MirrorMode,
        build_training_pairs,
        fit_frontalization,
        load_pose_dataset,
        mean_shape,
    )

    data = load_pose_dataset()
    template = mean_shape(data.landmarks)
    pairs = build_training_pairs(data, template, MirrorMode(mirror))
    weights = fit_frontalization(pairs, ridge=ridge)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, weights, allow_pickle=False)
    typer.echo(f"Trained on {len(pairs.inputs)} pairs from {len(data)} faces -> {output}")


@app.command()
def evaluate(
    model: ModelOpt = None,
    seed: Annotated[int, typer.Option(help="Seed of the person split.")] = 1,
    data: Annotated[Path | None, typer.Option(help="Path to Morphset.csv.zip.")] = None,
) -> None:
    """Evaluate a model on the held-out test persons of MorphSet."""
    from .data import load_morphset, split_by_group
    from .model import load_model
    from .training import evaluate_model

    dataset = load_morphset(data)
    _, val, test = split_by_group(dataset.persons, seed=seed)
    m = load_model(model)
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
    from .model import load_model

    rgb = load_image(image)
    detector = DlibLandmarkDetector(upsample=upsample)
    model = load_model()
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
    smoothing: Annotated[
        float, typer.Option(min=0, max=0.9, help="EMA smoothing (also -/+ keys).")
    ] = 0.3,
    upsample: Annotated[int, typer.Option(min=0, help="dlib upsampling (small faces).")] = 0,
    max_width: Annotated[int, typer.Option(min=160, help="Resize frames to this width.")] = 640,
    track: Annotated[
        bool, typer.Option(help="Track landmarks between frames (steadier, better in low light).")
    ] = True,
    model: ModelOpt = None,
) -> None:
    """Live emotion overlay from a webcam.

    Keys: m switch model (PLS / kernel), -/+ smoothing, l landmarks, t tracking,
    q or Esc quit.
    """
    from .live import CameraError, run_webcam

    try:
        run_webcam(
            camera=camera, smoothing=smoothing, model_path=model, upsample=upsample,
            max_width=max_width, tracking=track,
        )  # fmt: skip
    except CameraError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def _print_metrics(metrics: dict[str, dict]) -> None:  # type: ignore[type-arg]
    for split, res in metrics.items():
        typer.secho(f"\n{split}: {res['n_samples']} faces, {res['n_persons']} persons", bold=True)
        for view in ("regressor", "pipeline"):
            typer.echo(f"  {view}:")
            for target, m in res[view].items():
                typer.echo(
                    f"    {target:<10} CCC={m['ccc']:.4f}  r={m['pearson']:.4f}  "
                    f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
                )


if __name__ == "__main__":
    app()
