#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train --dataset synthetic --decoder flow_matching --num_epochs 20 --batch_size 32 --device auto

