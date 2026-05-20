#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
PY

