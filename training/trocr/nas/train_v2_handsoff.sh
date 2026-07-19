#!/usr/bin/env bash
# Hands-off v2 retrain: start from v1, mix OG+180k+Nusa, train until early-stop
# or max epochs, push best weights to Hub v2 (overwrite).
set -euo pipefail

cd /workspace/trocr

# GPU selection (discrete Arc preferred).
mapfile -t DETECT_LINES < <(detect_gpu.sh | tee /dev/stderr)
for line in "${DETECT_LINES[@]}"; do
  case "$line" in
    ONEAPI_DEVICE_SELECTOR=*|ZE_AFFINITY_MASK=*)
      export "${line?}"
      ;;
  esac
done
echo "[train] ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-unset}"

: "${HF_TOKEN:?HF_TOKEN must be set (docker -e HF_TOKEN=... or env_file)}"

BASE_MODEL="${BASE_MODEL:-thesimonharms/trocr-javanese-synthetic}"
HUB_MODEL_ID="${HUB_MODEL_ID:-thesimonharms/trocr-javanese-synthetic-v2}"
DATASET_NAME="${DATASET_NAME:-thesimonharms/javanese-dataset-180k}"
# Replay original val-matched data + real OCR lines.
EXTRA_DATASETS="${EXTRA_DATASETS:-thesimonharms/javanese-dataset,thesimonharms/javanese-nusaaksara-ocr}"
EXTRA_UPSAMPLE="${EXTRA_UPSAMPLE:-1,8}"

EPOCHS="${EPOCHS:-15}"
LR="${LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
EARLY_STOP="${EARLY_STOP:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/trocr_v2}"

# Batch: env TROCR_BATCH_SIZE wins; else python heuristic.
if [[ -z "${BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="$(python -c 'from device_utils import recommend_batch_size; print(recommend_batch_size())')"
fi

mkdir -p "$OUTPUT_DIR" /workspace/hf-cache /workspace/logs
LOG="/workspace/logs/train_v2_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[train] base=$BASE_MODEL hub=$HUB_MODEL_ID"
echo "[train] data=$DATASET_NAME extras=$EXTRA_DATASETS ups=$EXTRA_UPSAMPLE"
echo "[train] epochs=$EPOCHS lr=$LR batch=$BATCH_SIZE early_stop=$EARLY_STOP"
echo "[train] output=$OUTPUT_DIR log=$LOG"

# Resume if a prior container died mid-run.
RESUME_FLAG=()
if compgen -G "$OUTPUT_DIR/checkpoint-*" > /dev/null; then
  RESUME_FLAG=(--resume_from_checkpoint)
  echo "[train] found checkpoint(s) — will resume"
fi

# Linux XPU: try bf16+default attn; fall back knobs via env if needed.
# TROCR_FORCE_FP32=1 TROCR_ATTN=eager to mirror Windows-safe mode.
set -o pipefail
python -u finetune_trocr.py \
  --base_model "$BASE_MODEL" \
  --dataset_name "$DATASET_NAME" \
  --extra_dataset_name "$EXTRA_DATASETS" \
  --extra_dataset_upsample "$EXTRA_UPSAMPLE" \
  --hub_model_id "$HUB_MODEL_ID" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --warmup_ratio "$WARMUP_RATIO" \
  --eval_every_epochs 1 \
  --early_stopping_patience "$EARLY_STOP" \
  --load_best_model_at_end \
  --skip_final_cer \
  --pdf_labeled_dir none \
  --no-gradient_checkpointing \
  "${RESUME_FLAG[@]}" \
  2>&1 | tee -a "$LOG"

echo "[train] DONE — best/final weights should be on Hub: $HUB_MODEL_ID"
