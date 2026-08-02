#!/usr/bin/env bash
# Find a stable TROCR_BATCH_SIZE (+ GC on/off) for trocr-large-printed on this XPU.
# Prefers transformers-xpu-helper vision recommendation, then probes neighbors.
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

# Hint size from xpu-helper (ignore TROCR_BATCH_SIZE during discovery).
HINT_BS="$(
  unset TROCR_BATCH_SIZE
  export GRADIENT_CHECKPOINTING=1
  python -c 'from device_utils import recommend_batch_size; print(recommend_batch_size())' 2>/dev/null || echo 8
)"
echo "[smoke] xpu-helper hint batch=$HINT_BS" | tee -a "$LOG"

try_batch() {
  local bs="$1"
  local gc="$2"  # 1 or 0
  local tag="bs${bs}_gc${gc}"
  local out="$SMOKE_OUT/$tag"
  rm -rf "$out"
  mkdir -p "$out"
  local gc_flag=(--gradient_checkpointing)
  if [[ "$gc" == "0" ]]; then
    gc_flag=(--no-gradient_checkpointing)
  fi
  echo "[smoke] trying batch=$bs gc=$gc max_steps=$MAX_STEPS" | tee -a "$LOG"
  # Unset TROCR_BATCH_SIZE so finetune does not re-override --batch_size.
  if env -u TROCR_BATCH_SIZE python -u finetune_trocr.py \
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
    "${gc_flag[@]}" \
    --save_total_limit 1 \
    2>&1 | tee -a "$LOG"
  then
    echo "[smoke] OK batch=$bs gc=$gc" | tee -a "$LOG"
    echo "$bs" > /workspace/output/trocr_v4_smoke/best_batch.txt
    echo "$gc" > /workspace/output/trocr_v4_smoke/best_gc.txt
    return 0
  fi
  echo "[smoke] FAIL batch=$bs gc=$gc" | tee -a "$LOG"
  return 1
}

# Largest-first so we fill shared DRAM; helper hint is a floor, not a cap.
CANDIDATES=()
for b in 16 12 8 "$HINT_BS" 4 2; do
  [[ "$b" =~ ^[0-9]+$ ]] || continue
  skip=0
  for c in "${CANDIDATES[@]:-}"; do
    [[ "$c" == "$b" ]] && skip=1 && break
  done
  [[ $skip -eq 1 ]] && continue
  CANDIDATES+=("$b")
done
echo "[smoke] candidate order (largest first): ${CANDIDATES[*]}" | tee -a "$LOG"

for gc in 1 0; do
  for bs in "${CANDIDATES[@]}"; do
    if try_batch "$bs" "$gc"; then
      echo "[smoke] selected TROCR_BATCH_SIZE=$bs GRADIENT_CHECKPOINTING=$gc" | tee -a "$LOG"
      exit 0
    fi
  done
done

echo "[smoke] ERROR: no batch/gc combo worked" | tee -a "$LOG"
exit 1
