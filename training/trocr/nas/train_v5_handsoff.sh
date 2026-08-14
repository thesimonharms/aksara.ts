#!/usr/bin/env bash
# Hands-off v5: synthetic-HQ first cook (no old 60k/180k mix).
#   Phase A: freeze encoder → Phase B: unfreeze (2 ep) → score on HQ val.
# Hub: thesimonharms/trocr-javanese-synthetic-v5
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
echo "[train] ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-unset}"

: "${HF_TOKEN:?HF_TOKEN must be set}"

BASE_MODEL="${BASE_MODEL:-microsoft/trocr-large-printed}"
HUB_MODEL_ID="${HUB_MODEL_ID:-thesimonharms/trocr-javanese-synthetic-v5}"
# First synthetic-HQ cook: HQ ×1 only (do not drown with old mixes).
DATASET_NAME="${DATASET_NAME:-thesimonharms/javanese-synthetic-hq}"
EXTRA_DATASETS="${EXTRA_DATASETS:-}"
EXTRA_UPSAMPLE="${EXTRA_UPSAMPLE:-1}"
SCORE_DATASET="${SCORE_DATASET:-$DATASET_NAME}"

STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-2}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-2}"
STAGE_A_LR="${STAGE_A_LR:-3e-5}"
STAGE_B_LR="${STAGE_B_LR:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"

if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ -f /workspace/output/trocr_v4_smoke/best_batch.txt ]]; then
    BATCH_SIZE="$(tr -d '[:space:]' </workspace/output/trocr_v4_smoke/best_batch.txt)"
    echo "[train] batch from smoke best_batch.txt=$BATCH_SIZE"
  elif [[ -n "${TROCR_BATCH_SIZE:-}" ]]; then
    BATCH_SIZE="$TROCR_BATCH_SIZE"
  else
    BATCH_SIZE="$(python -c 'from device_utils import recommend_batch_size; print(recommend_batch_size())')"
    echo "[train] batch from xpu-helper/heuristic=$BATCH_SIZE"
  fi
fi

if [[ -f /workspace/output/trocr_v4_smoke/best_gc.txt ]]; then
  GRADIENT_CHECKPOINTING="$(tr -d '[:space:]' </workspace/output/trocr_v4_smoke/best_gc.txt)"
fi
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
GC_FLAG=(--gradient_checkpointing)
if [[ "$GRADIENT_CHECKPOINTING" == "0" ]]; then
  GC_FLAG=(--no-gradient_checkpointing)
fi

OUT_A="${OUTPUT_DIR_A:-/workspace/output/trocr_v5_stage_a}"
OUT_B="${OUTPUT_DIR_B:-/workspace/output/trocr_v5_stage_b}"
SCORES_CSV="${SCORES_CSV:-/workspace/output/trocr_v5_evals/scores.csv}"
N_SCORE="${N_SCORE:-1500}"
SCORE_ONLY="${SCORE_ONLY:-0}"

mkdir -p "$OUT_A" "$OUT_B" "$(dirname "$SCORES_CSV")" /workspace/hf-cache /workspace/logs
LOG="/workspace/logs/train_v5_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[train] hub=$HUB_MODEL_ID base=$BASE_MODEL score_only=$SCORE_ONLY" | tee -a "$LOG"
echo "[train] data=$DATASET_NAME extras=${EXTRA_DATASETS:-none} ups=$EXTRA_UPSAMPLE" | tee -a "$LOG"
echo "[train] score_dataset=$SCORE_DATASET" | tee -a "$LOG"
echo "[train] batch=$BATCH_SIZE A: epochs=$STAGE_A_EPOCHS lr=$STAGE_A_LR freeze_encoder" | tee -a "$LOG"
echo "[train] B: epochs=$STAGE_B_EPOCHS lr=$STAGE_B_LR unfrozen from Stage A final" | tee -a "$LOG"
echo "[train] gc=${GC_FLAG[*]}" | tee -a "$LOG"

score_sweep() {
  echo "[train] === SCORE SWEEP (iGPU) ===" | tee -a "$LOG"
  SCORE_ROOT="/workspace/output/trocr_v5_all_ckpts"
  rm -rf "$SCORE_ROOT"
  mkdir -p "$SCORE_ROOT"
  i=0
  for d in \
    $(ls -d "$OUT_A"/checkpoint-* 2>/dev/null | sort -V) \
    $(ls -d "$OUT_B"/checkpoint-* 2>/dev/null | sort -V) \
    "$OUT_A"/final \
    "$OUT_B"/final
  do
    [[ -d "$d" ]] || continue
    i=$((i + 1))
    ln -sfn "$d" "$SCORE_ROOT/$(printf '%02d' "$i")-$(basename "$(dirname "$d")")-$(basename "$d")"
  done
  echo "[train] linked $i checkpoints under $SCORE_ROOT" | tee -a "$LOG"
  if [[ "$i" -eq 0 ]]; then
    echo "[train] ERROR: no checkpoints under $OUT_A or $OUT_B" | tee -a "$LOG"
    exit 1
  fi

  python -u score_epoch_checkpoints.py \
    --ckpt_root "$SCORE_ROOT" \
    --dataset_name "$SCORE_DATASET" \
    --n_samples "$N_SCORE" \
    --out_csv "$SCORES_CSV" \
    --hub_model_id "$HUB_MODEL_ID" \
    2>&1 | tee -a "$LOG"
}

if [[ "$SCORE_ONLY" == "1" ]]; then
  score_sweep
  echo "[train] DONE score-only — scores=$SCORES_CSV" | tee -a "$LOG"
  exit 0
fi

COMMON=(
  --dataset_name "$DATASET_NAME"
  --hub_model_id "$HUB_MODEL_ID"
  --batch_size "$BATCH_SIZE"
  --warmup_ratio "$WARMUP_RATIO"
  --eval_every_epochs 1
  --skip_final_cer
  --pdf_labeled_dir none
  --save_total_limit 0
  "${GC_FLAG[@]}"
)
if [[ -n "${EXTRA_DATASETS}" ]]; then
  COMMON+=(--extra_dataset_name "$EXTRA_DATASETS" --extra_dataset_upsample "$EXTRA_UPSAMPLE")
fi

# ----- Phase A: freeze encoder -----
RESUME_A=()
if compgen -G "$OUT_A/checkpoint-*" > /dev/null; then
  RESUME_A=(--resume_from_checkpoint)
  echo "[train] Phase A resume from checkpoint under $OUT_A" | tee -a "$LOG"
fi

echo "[train] === PHASE A (freeze encoder, large-printed) ===" | tee -a "$LOG"
python -u finetune_trocr.py \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUT_A" \
  --epochs "$STAGE_A_EPOCHS" \
  --lr "$STAGE_A_LR" \
  --freeze_encoder \
  "${COMMON[@]}" \
  "${RESUME_A[@]}" \
  2>&1 | tee -a "$LOG"

STAGE_A_FINAL="$OUT_A/final"
if [[ ! -d "$STAGE_A_FINAL" ]]; then
  echo "[train] ERROR: Phase A final missing at $STAGE_A_FINAL" | tee -a "$LOG"
  exit 1
fi

# ----- Phase B: unfreeze -----
RESUME_B=()
STAGE_B_BASE="$STAGE_A_FINAL"
if compgen -G "$OUT_B/checkpoint-*" > /dev/null; then
  RESUME_B=(--resume_from_checkpoint)
  echo "[train] Phase B resume from checkpoint under $OUT_B" | tee -a "$LOG"
fi

echo "[train] === PHASE B (unfreeze, low LR) base=$STAGE_B_BASE ===" | tee -a "$LOG"
python -u finetune_trocr.py \
  --base_model "$STAGE_B_BASE" \
  --output_dir "$OUT_B" \
  --epochs "$STAGE_B_EPOCHS" \
  --lr "$STAGE_B_LR" \
  --no-expand_javanese_tokenizer \
  "${COMMON[@]}" \
  "${RESUME_B[@]}" \
  2>&1 | tee -a "$LOG"

score_sweep

echo "[train] DONE v5 — Hub root=$HUB_MODEL_ID scores=$SCORES_CSV" | tee -a "$LOG"
