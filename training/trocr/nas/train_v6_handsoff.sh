#!/usr/bin/env bash
# Hands-off v6: exact-match cook on short clean synthetic.
#   Phase 0: unfrozen overfit 256 (stack gate, weights discarded)
#   Phase A/B: unfrozen cook (small DeiT must adapt to aksara; freeze was fatal)
# Hub: thesimonharms/trocr-javanese-synthetic-v6
# Base: microsoft/trocr-small-printed
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

BASE_MODEL="${BASE_MODEL:-microsoft/trocr-small-printed}"
HUB_MODEL_ID="${HUB_MODEL_ID:-thesimonharms/trocr-javanese-synthetic-v6}"
DATASET_NAME="${DATASET_NAME:-thesimonharms/javanese-synthetic-exact}"
EXTRA_DATASETS="${EXTRA_DATASETS:-}"
EXTRA_UPSAMPLE="${EXTRA_UPSAMPLE:-1}"
SCORE_DATASET="${SCORE_DATASET:-$DATASET_NAME}"

STAGE_0_EPOCHS="${STAGE_0_EPOCHS:-200}"
STAGE_0_SAMPLES="${STAGE_0_SAMPLES:-32}"
STAGE_0_GATE="${STAGE_0_GATE:-0.80}"
STAGE_0_LR="${STAGE_0_LR:-1e-4}"
STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-3}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-12}"
STAGE_A_LR="${STAGE_A_LR:-5e-5}"
STAGE_B_LR="${STAGE_B_LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-24}"
MAX_LABEL_CHARS="${MAX_LABEL_CHARS:-12}"
SHORT_FRAC="${SHORT_FRAC:-0.35}"
SHORT_MAX="${SHORT_MAX:-6}"

if [[ -z "${BATCH_SIZE:-}" ]]; then
  if [[ -n "${TROCR_BATCH_SIZE:-}" ]]; then
    BATCH_SIZE="$TROCR_BATCH_SIZE"
  else
    BATCH_SIZE="16"
  fi
fi

GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
GC_FLAG=(--no-gradient_checkpointing)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  GC_FLAG=(--gradient_checkpointing)
fi

OUT_0="${OUTPUT_DIR_0:-/workspace/output/trocr_v6_stage_0}"
OUT_A="${OUTPUT_DIR_A:-/workspace/output/trocr_v6_stage_a}"
OUT_B="${OUTPUT_DIR_B:-/workspace/output/trocr_v6_stage_b}"
SCORES_CSV="${SCORES_CSV:-/workspace/output/trocr_v6_evals/scores.csv}"
N_SCORE="${N_SCORE:-1500}"
SCORE_ONLY="${SCORE_ONLY:-0}"
SKIP_GATE="${SKIP_GATE:-0}"

mkdir -p "$OUT_0" "$OUT_A" "$OUT_B" "$(dirname "$SCORES_CSV")" /workspace/hf-cache /workspace/logs
LOG="/workspace/logs/train_v6_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "[train] hub=$HUB_MODEL_ID base=$BASE_MODEL score_only=$SCORE_ONLY" | tee -a "$LOG"
echo "[train] data=$DATASET_NAME extras=${EXTRA_DATASETS:-none}" | tee -a "$LOG"
echo "[train] score_dataset=$SCORE_DATASET" | tee -a "$LOG"
echo "[train] batch=$BATCH_SIZE max_len=$MAX_TARGET_LENGTH max_chars=$MAX_LABEL_CHARS" | tee -a "$LOG"
echo "[train] 0: overfit n=$STAGE_0_SAMPLES ep=$STAGE_0_EPOCHS lr=$STAGE_0_LR unfrozen gate=$STAGE_0_GATE (discard weights)" | tee -a "$LOG"
echo "[train] A: epochs=$STAGE_A_EPOCHS lr=$STAGE_A_LR unfrozen" | tee -a "$LOG"
echo "[train] B: epochs=$STAGE_B_EPOCHS lr=$STAGE_B_LR unfrozen from Stage A final" | tee -a "$LOG"
echo "[train] gc=${GC_FLAG[*]}" | tee -a "$LOG"

score_sweep() {
  echo "[train] === SCORE SWEEP (exact-match) ===" | tee -a "$LOG"
  SCORE_ROOT="/workspace/output/trocr_v6_all_ckpts"
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
  --batch_size "$BATCH_SIZE"
  --warmup_ratio "$WARMUP_RATIO"
  --eval_every_epochs 1
  --skip_final_cer
  --pdf_labeled_dir none
  --save_total_limit 0
  --max_target_length "$MAX_TARGET_LENGTH"
  --max_label_chars "$MAX_LABEL_CHARS"
  --short_line_fraction "$SHORT_FRAC"
  --short_line_max_chars "$SHORT_MAX"
  "${GC_FLAG[@]}"
)
if [[ -n "${EXTRA_DATASETS}" ]]; then
  COMMON+=(--extra_dataset_name "$EXTRA_DATASETS" --extra_dataset_upsample "$EXTRA_UPSAMPLE")
fi

# ----- Phase 0: overfit gate (do not continue from these weights) -----
if [[ "$SKIP_GATE" != "1" ]]; then
  echo "[train] === PHASE 0 (unfrozen overfit gate, weights discarded) ===" | tee -a "$LOG"
  rm -rf "$OUT_0"
  mkdir -p "$OUT_0"
  python -u finetune_trocr.py \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUT_0" \
    --epochs "$STAGE_0_EPOCHS" \
    --lr "$STAGE_0_LR" \
    --lr_scheduler_type constant_with_warmup \
    --max_train_samples "$STAGE_0_SAMPLES" \
    --short_line_fraction 0 \
    --no_push \
    --no_hub_tag_epochs \
    --dataset_name "$DATASET_NAME" \
    --batch_size "$BATCH_SIZE" \
    --warmup_ratio 0.1 \
    --weight_decay 0 \
    --logging_steps 8 \
    --eval_on_train \
    --eval_every_epochs 1 \
    --skip_final_cer \
    --pdf_labeled_dir none \
    --save_total_limit 1 \
    --max_target_length "$MAX_TARGET_LENGTH" \
    --max_label_chars "$MAX_LABEL_CHARS" \
    "${GC_FLAG[@]}" \
    2>&1 | tee -a "$LOG"

  GATE_MODEL="$OUT_0/final"
  if [[ ! -d "$GATE_MODEL" ]]; then
    echo "[train] ERROR: Phase 0 final missing at $GATE_MODEL" | tee -a "$LOG"
    exit 1
  fi
  echo "[train] Phase 0 exact-match gate on train n=$STAGE_0_SAMPLES need=$STAGE_0_GATE" | tee -a "$LOG"
  HUB_MODEL_ID="$GATE_MODEL" \
  DATASET_NAME="$DATASET_NAME" \
  VERIFY_SPLIT=train \
  VERIFY_SUBSET="$STAGE_0_SAMPLES" \
  VERIFY_SEED=42 \
  N_SAMPLES="$STAGE_0_SAMPLES" \
  EXACT_GATE="$STAGE_0_GATE" \
    python -u local_verify_large.py 2>&1 | tee -a "$LOG"
  echo "[train] Phase 0 GATE PASSED — starting real cook from $BASE_MODEL" | tee -a "$LOG"
fi

# ----- Phase A -----
RESUME_A=()
if compgen -G "$OUT_A/checkpoint-*" > /dev/null; then
  RESUME_A=(--resume_from_checkpoint)
  echo "[train] Phase A resume from checkpoint under $OUT_A" | tee -a "$LOG"
fi

echo "[train] === PHASE A (unfrozen, small-printed) ===" | tee -a "$LOG"
python -u finetune_trocr.py \
  --base_model "$BASE_MODEL" \
  --output_dir "$OUT_A" \
  --epochs "$STAGE_A_EPOCHS" \
  --lr "$STAGE_A_LR" \
  --hub_model_id "$HUB_MODEL_ID" \
  "${COMMON[@]}" \
  "${RESUME_A[@]}" \
  2>&1 | tee -a "$LOG"

STAGE_A_FINAL="$OUT_A/final"
if [[ ! -d "$STAGE_A_FINAL" ]]; then
  echo "[train] ERROR: Phase A final missing at $STAGE_A_FINAL" | tee -a "$LOG"
  exit 1
fi

# ----- Phase B -----
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
  --hub_model_id "$HUB_MODEL_ID" \
  "${COMMON[@]}" \
  "${RESUME_B[@]}" \
  2>&1 | tee -a "$LOG"

score_sweep

echo "[train] DONE v6 — Hub root=$HUB_MODEL_ID scores=$SCORES_CSV" | tee -a "$LOG"
