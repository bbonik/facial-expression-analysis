"""Build and deploy the web demo to a Hugging Face Space.

The Space is self-contained: the app, a vendored copy of the package (with its bundled
models), example photos and a requirements file. Nothing is fetched from GitHub at runtime;
the dlib landmark model downloads from dlib.net on the Space's first start.

Usage (via the wrapper):
  scripts/deploy_space.sh --dry-run   # build ./build/space only, inspect it
  scripts/deploy_space.sh             # deploy to <your-user>/facial-expression-analysis
  scripts/deploy_space.sh --space org/name --private

Requires a Hugging Face token with write access: run `hf auth login` once, or set HF_TOKEN.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "space"
APP_FILES = ("app.py", "core.py", "render.py", "style.css")
REQUIREMENTS = """\
numpy>=1.26
opencv-python-headless>=4.9
dlib-bin>=19.24
pooch>=1.8
"""
# Remote files this script manages; anything else in the Space (e.g. .gitattributes) is kept.
MANAGED = ["*.py", "*.css", "*.txt", "README.md", "examples/*", "assets/*",
           "facial_expression_analysis/**"]  # fmt: skip


def _gradio_version() -> str:
    try:
        import gradio

        return str(gradio.__version__)
    except ImportError:
        return "6.32.0"


def _git_ref() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def space_readme(ref: str) -> str:
    return f"""---
title: Facial Expression Analysis
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: {_gradio_version()}
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Real-time arousal and valence from facial landmarks
tags:
  - affective-computing
  - facial-expression
  - emotion-recognition
  - classical-ml
---

# Facial Expression Analysis

Estimate **arousal, valence and intensity** from 68 facial landmarks in real time with an
interpretable linear model (PLS, 31 components) trained on 83k faces of 337 people.

- Code, experiments and evaluation: https://github.com/bbonik/facial-expression-analysis
- Model card: https://github.com/bbonik/facial-expression-analysis/blob/v2/MODEL_CARD.md
- Papers: MorphSet (ICIP 2021), landmark frontalization (ICIP 2020)

Built from commit `{ref}`.
"""


def build(out: Path = BUILD) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in APP_FILES:
        shutil.copy2(ROOT / "app" / name, out / name)
    shutil.copytree(
        ROOT / "src" / "facial_expression_analysis",
        out / "facial_expression_analysis",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (out / "examples").mkdir()
    for img in sorted((ROOT / "data" / "images").iterdir()):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(img, out / "examples" / img.name)
    figure = ROOT / "reports" / "figures" / "behavioural_smile.png"
    if figure.is_file():
        (out / "assets").mkdir()
        shutil.copy2(figure, out / "assets" / figure.name)
    (out / "requirements.txt").write_text(REQUIREMENTS)
    (out / "README.md").write_text(space_readme(_git_ref()))
    return out


def deploy(folder: Path, space: str | None, private: bool) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is missing; run through scripts/deploy_space.sh")
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:  # not logged in / invalid token
        sys.exit(
            "Not logged in to Hugging Face. Run `uvx --from huggingface_hub hf auth login` "
            "(token with write access) or set HF_TOKEN, then run this script again."
        )
    repo_id = space or f"{user}/facial-expression-analysis"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", private=private,
                    exist_ok=True)  # fmt: skip
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Deploy {_git_ref()}",
        delete_patterns=MANAGED,
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--space", help="Space id, e.g. user/name (default: <you>/facial-expression-analysis)"
    )
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true", help="only build the Space folder")
    args = ap.parse_args()

    folder = build()
    files = sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())
    size = sum((folder / f).stat().st_size for f in files) / 1e6
    print(f"Built {folder} ({len(files)} files, {size:.1f} MB)")
    if args.dry_run:
        print("Dry run: nothing uploaded. Test it locally with:")
        print(f"  cd {folder} && uv run --project {ROOT} --extra demo python app.py")
        return
    url = deploy(folder, args.space, args.private)
    print(
        f"Deployed: {url}\n"
        "The first start takes a few minutes (installing dlib, fetching its model)."
    )


if __name__ == "__main__":
    main()
