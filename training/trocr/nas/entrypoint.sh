#!/usr/bin/env bash
set -euo pipefail

cmd="${1:-train}"
shift || true

case "$cmd" in
  detect|gpu)
    exec detect_gpu.sh "$@"
    ;;
  train|handsoff|v2)
    exec train_v2_handsoff.sh "$@"
    ;;
  train-v3|v3|handsoff-v3)
    exec train_v3_handsoff.sh "$@"
    ;;
  train-v4|v4|handsoff-v4)
    exec train_v4_handsoff.sh "$@"
    ;;
  smoke-v4|smoke_v4)
    exec smoke_v4_batch.sh "$@"
    ;;
  smoke)
    # Tiny sanity: XPU matmul + one Trainer step worth of imports.
    python - <<'PY'
import torch
from device_utils import log_device, recommend_batch_size
assert torch.xpu.is_available(), "XPU not visible inside container"
log_device()
x = torch.randn(256, 256, device="xpu")
y = (x @ x).sum()
torch.xpu.synchronize()
print("[smoke] matmul_ok", float(y), "batch≈", recommend_batch_size())
PY
    ;;
  bash|sh)
    exec bash "$@"
    ;;
  *)
    echo "Usage: entrypoint.sh [detect|train|train-v3|train-v4|smoke|smoke-v4|bash]"
    exec "$cmd" "$@"
    ;;
esac
