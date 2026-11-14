#!/usr/bin/env bash
# Deploy the Gradio web demo to a Hugging Face Space.
#
# Usage: scripts/deploy_space.sh [--dry-run] [--space user/name] [--private]
#   scripts/deploy_space.sh --dry-run   # build ./build/space and stop (inspect or test it)
#   scripts/deploy_space.sh             # deploy to <your-hf-user>/facial-expression-analysis
#
# One-time setup: a Hugging Face account and a token with write access:
#   uvx --from huggingface_hub hf auth login
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

exec uv run --quiet --extra demo --with huggingface_hub python scripts/deploy_space.py "$@"
