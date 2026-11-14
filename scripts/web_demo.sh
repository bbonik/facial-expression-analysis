#!/usr/bin/env bash
# Set up and launch the Gradio web demo in your browser.
#
# Usage: scripts/web_demo.sh [--port 7860] [--share]
#   --port N   serve on another port (default 7860)
#   --share    also create a temporary public gradio.live link (to try it on a phone)
#
# Press Ctrl+C in this terminal to stop the demo.
set -euo pipefail

cd "$(dirname "$0")/.."

port=7860
share=0
while [ $# -gt 0 ]; do
    case "$1" in
        --port) port="$2"; shift 2 ;;
        --share) share=1; shift ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it with:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "==> Installing dependencies (first run only takes a minute)"
uv sync --quiet --extra demo

echo "==> Fetching the dlib landmark model (~100 MB, cached after the first run)"
uv run --quiet fea download

echo "==> Starting the web demo at http://127.0.0.1:${port} (opens in your browser; Ctrl+C to stop)"
echo "    In the Live tab, click the camera and allow access when the browser asks."
export FEA_OPEN_BROWSER="${FEA_OPEN_BROWSER:-1}" GRADIO_SERVER_PORT="$port"
if [ "$share" = 1 ]; then export GRADIO_SHARE=True; fi
exec uv run --quiet --extra demo python app/app.py
