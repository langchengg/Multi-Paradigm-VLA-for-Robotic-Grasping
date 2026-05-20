#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train --dataset synthetic --decoder diffusion --num_epochs 20 --batch_size 32 --device auto

