#!/usr/bin/env bash
# Set up and launch the real-time webcam demo.
#
# Usage: scripts/demo.sh [fea webcam options]
#   scripts/demo.sh                     # default camera, kernel model
#   scripts/demo.sh --camera 1          # another camera
#   scripts/demo.sh --smoothing 0.6     # calmer output
#   scripts/demo.sh --upsample 1        # find smaller / more distant faces (slower)
#   scripts/demo.sh --model my.npz      # a model trained with `fea train`
#
# Keys in the demo: m switch model (kernel / PLS), -/+ smoothing, l landmarks, q or Esc quit.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "==> Installing dependencies (first run only takes a minute)"
uv sync --quiet

echo "==> Fetching the dlib landmark model (~100 MB, cached after the first run)"
uv run --quiet fea download

if [ "$(uname)" = "Darwin" ]; then
    echo "==> macOS: the first run asks for camera access for this terminal app. Click Allow"
    echo "    (or enable it in System Settings > Privacy & Security > Camera), then quit and"
    echo "    reopen the terminal and run this script again."
fi

echo "==> Starting the demo. Keys: m model | -/+ smoothing | l landmarks | q quit"
exec uv run --quiet fea webcam "$@"
