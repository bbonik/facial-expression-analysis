"""Download and cache large external assets (dlib model, MorphSet dataset).

Files are cached in ``$FEA_CACHE_DIR`` if set, otherwise in the platform cache
directory (e.g. ``~/Library/Caches/facial-expression-analysis`` on macOS).
"""

from __future__ import annotations

import bz2
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "facial-expression-analysis"


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    sha256: str
    """Hash of the downloaded file (before any decompression)."""


DLIB_PREDICTOR = RemoteFile(
    name="shape_predictor_68_face_landmarks.dat",
    url="http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    sha256="7d6637b8f34ddb0c1363e09a4628acb34314019ec3566fd66b80c04dda6980f5",
)

MORPHSET = RemoteFile(
    name="Morphset.csv.zip",
    url="https://github.com/bbonik/facial-expression-analysis/releases/download/data-v1/Morphset.csv.zip",
    sha256="8b59c090d0937b3c2da7d5a063b5439dfe89fe3082dbcf7bebe6bc5192c46d9e",
)

POSE_DATASET = RemoteFile(
    name="landmark_dataset.csv",
    url=(
        "https://raw.githubusercontent.com/bbonik/facial-landmark-frontalization/"
        "aea4c35a23f2a6d74c43680d6873a961fbfa7e20/data/landmark_dataset.csv"
    ),
    sha256="8b05214e6ac15090a423528ddbb3d446620d28050b9f625b2cc8838d426cc631",
)
"""Multi-pose landmark dataset used to train the frontalization model."""


def cache_dir() -> Path:
    """Return (and create) the asset cache directory."""
    env = os.environ.get("FEA_CACHE_DIR")
    if env:
        path = Path(env).expanduser()
    else:
        import pooch

        path = Path(pooch.os_cache(APP_NAME))
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256sum(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(remote: RemoteFile, dest_dir: Path) -> Path:
    import pooch

    fname = remote.url.rsplit("/", 1)[-1]
    logger.info("Downloading %s ...", remote.url)
    return Path(
        pooch.retrieve(
            remote.url,
            known_hash=f"sha256:{remote.sha256}",
            fname=fname,
            path=dest_dir,
            progressbar=False,
        )
    )


def fetch_dlib_predictor() -> Path:
    """Return the path of the dlib 68-landmark predictor, downloading it if needed."""
    target = cache_dir() / DLIB_PREDICTOR.name
    if target.is_file():
        return target
    archive = _download(DLIB_PREDICTOR, cache_dir())
    tmp = target.with_suffix(".tmp")
    with bz2.open(archive, "rb") as src, tmp.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(target)
    archive.unlink()
    return target


def _repo_root() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else None


def fetch_morphset(path: str | Path | None = None) -> Path:
    """Locate the zipped MorphSet CSV, downloading it if necessary.

    Search order: ``path`` argument, ``$FEA_MORPHSET``, ``data/`` of a source
    checkout, the cache directory, and finally the GitHub release download.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if env := os.environ.get("FEA_MORPHSET"):
        candidates.append(Path(env))
    if (root := _repo_root()) is not None:
        candidates.append(root / "data" / MORPHSET.name)
    candidates.append(cache_dir() / MORPHSET.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if path is not None:
        raise FileNotFoundError(f"MorphSet file not found: {path}")
    return _download(MORPHSET, cache_dir())


def fetch_pose_dataset() -> Path:
    """Return the path of the multi-pose landmark dataset, downloading it if needed."""
    target = cache_dir() / POSE_DATASET.name
    if target.is_file():
        return target
    return _download(POSE_DATASET, cache_dir())
