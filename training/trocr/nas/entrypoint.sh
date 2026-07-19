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
    echo "Usage: entrypoint.sh [detect|train|smoke|bash]"
    exec "$cmd" "$@"
    ;;
esac
