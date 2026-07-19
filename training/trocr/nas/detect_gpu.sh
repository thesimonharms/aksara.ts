#!/usr/bin/env bash
# Detect Intel Level Zero / DRM render nodes and pick a discrete Arc when possible.
set -euo pipefail

echo "[detect] /dev/dri:"
ls -la /dev/dri 2>/dev/null || echo "  (missing — host must pass --device /dev/dri)"

echo "[detect] render nodes:"
shopt -s nullglob
for n in /dev/dri/renderD*; do
  echo "  $n  gid=$(stat -c %g "$n" 2>/dev/null || echo '?')"
done

if command -v sycl-ls >/dev/null 2>&1; then
  echo "[detect] sycl-ls:"
  sycl-ls || true
fi

python - <<'PY'
import os, sys
try:
    import torch
except Exception as e:
    print(f"[detect] torch import failed: {e}")
    sys.exit(0)

print(f"[detect] torch={torch.__version__}")
xpu = getattr(torch, "xpu", None)
if not xpu or not torch.xpu.is_available():
    print("[detect] torch.xpu NOT available")
    sys.exit(0)

n = torch.xpu.device_count()
print(f"[detect] xpu_count={n}")
best = 0
best_mem = -1
for i in range(n):
    name = torch.xpu.get_device_name(i)
    props = torch.xpu.get_device_properties(i)
    mem = int(getattr(props, "total_memory", 0) or 0)
    integrated = bool(getattr(props, "is_integrated_gpu", False))
    print(f"[detect] xpu:{i} name={name!r} mem_gb={mem/1024**3:.1f} integrated={integrated}")
    # Prefer discrete + most VRAM (B70 / B60 over iGPU).
    score = mem + (0 if integrated else 10**12)
    if score > best_mem:
        best_mem = score
        best = i

print(f"[detect] selected_index={best}")
# Emit shell-friendly assignment for callers that `eval` this script's last lines.
print(f"ONEAPI_DEVICE_SELECTOR=level_zero:{best}")
print(f"ZE_AFFINITY_MASK={best}")
PY
