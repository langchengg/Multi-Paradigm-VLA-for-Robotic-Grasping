#!/usr/bin/env bash
set -euo pipefail
python -m src.visualization.live_watch --dataset synthetic --decoder diffusion --checkpoint_dir checkpoints/synthetic/diffusion --viewer "$@"

