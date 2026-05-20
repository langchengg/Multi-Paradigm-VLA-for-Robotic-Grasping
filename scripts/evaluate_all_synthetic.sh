#!/usr/bin/env bash
set -euo pipefail
python -m src.training.evaluate --dataset synthetic --all_decoders --num_episodes 50 --save_video "$@"

