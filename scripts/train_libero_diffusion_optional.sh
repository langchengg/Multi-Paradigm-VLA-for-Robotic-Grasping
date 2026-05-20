#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train --dataset libero --libero_suite libero_object --decoder diffusion --num_epochs 10 --batch_size 16 --device auto "$@"

