#!/usr/bin/env bash
set -euo pipefail
python -m src.data.generate_synthetic_demos --output data/synthetic_demos.hdf5 --num_episodes 500 --image_size 128

