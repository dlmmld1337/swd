#!/bin/bash
set -e

source /code/auto_remaster/sandbox/DiffSynth-Studio/.venv_diff/bin/activate
cd "$(dirname "$0")"

python test_klein_single_image_inference.py \
    --sample-index 0 \
    --steps 4 30 \
    "$@"