#!/usr/bin/env bash
# Find a stable TROCR_BATCH_SIZE for trocr-large-printed on this XPU.
# Tries 8 → 4 → 2 with a short max_steps smoke (freeze encoder).
set -euo pipefail

cd /workspace/trocr

mapfile -t DETECT_LINES < <(detect_gpu.sh | tee /dev/stderr)
for line in "${DETECT_LINES[@]}"; do
  case "$line" in
    ONEAPI_DEVICE_SELECTOR=*|ZE_AFFINITY_MASK=*)
      export "${line?}"
      ;;
  esac
done

: "${HF_TOKEN:?HF_TOKEN must be set}"

BASE_MODEL="${BASE_MODEL:-microsoft/trocr-large-printed}"
SMOKE_OUT="${SMOKE_OUT:-/workspace/output/trocr_v4_smoke}"
MAX_STEPS="${SMOKE_MAX_STEPS:-40}"
LOG="/workspace/logs/smoke_v4_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$SMOKE_OUT" /workspace/logs

try_batch() {
  local bs="$1"
  local out="$SMOKE_OUT/bs${bs}"
  rm -rf "$out"
  mkdir -p "$out"
  echo "[smoke] trying batch=$bs max_steps=$MAX_STEPS" | tee -a "$LOG"
  if python -u finetune_trocr.py \
    --base_model "$BASE_MODEL" \
    --dataset_name thesimonharms/javanese-dataset \
    --output_dir "$out" \
    --epochs 1 \
    --max_steps "$MAX_STEPS" \
    --max_train_samples 2000 \
    --batch_size "$bs" \
    --lr 3e-5 \
    --freeze_encoder \
    --warmup_ratio 0.05 \
    --eval_every_epochs 1 \
    --skip_final_cer \
    --pdf_labeled_dir none \
    --no_push \
    --gradient_checkpointing \
    --save_total_limit 1 \
    2>&1 | tee -a "$LOG"
  then
    echo "[smoke] OK batch=$bs" | tee -a "$LOG"
    echo "$bs" > /workspace/output/trocr_v4_smoke/best_batch.txt
    return 0
  fi
  echo "[smoke] FAIL batch=$bs" | tee -a "$LOG"
  return 1
}

for bs in 8 4 2; do
  if try_batch "$bs"; then
    echo "[smoke] selected TROCR_BATCH_SIZE=$bs" | tee -a "$LOG"
    exit 0
  fi
done

echo "[smoke] ERROR: no batch size worked (tried 8/4/2)" | tee -a "$LOG"
exit 1
