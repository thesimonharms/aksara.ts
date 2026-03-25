#!/usr/bin/env bash
# training.sh - Training pipeline for aksara.ts
#
# Run from the project root:
#
#   bash training.sh                    # train both models
#   bash training.sh --model segmenter  # train word segmenter only
#   bash training.sh --model ocr        # train OCR model only
#   bash training.sh --force            # re-run all steps even if outputs exist
#
# Prerequisites:
#   - Node.js  (corpus transliteration)
#   - Python 3.11 via uv (venv is created automatically)
#   - training/jv_plain.txt  - Javanese Wikipedia dump (Latin script)
#   - data/jv.txt            - compact Javanese corpus for the segmenter
#   - training/PDFA.pdf      - optional: manuscript PDF for background textures

set -euo pipefail

# -- Argument parsing -------------------------------------------------------
MODEL="both"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$MODEL" != "both" && "$MODEL" != "segmenter" && "$MODEL" != "ocr" ]]; then
    echo "Invalid --model value: $MODEL (expected: both, segmenter, ocr)"
    exit 1
fi

RUN_SEGMENTER=0; RUN_OCR=0
[[ "$MODEL" == "both" || "$MODEL" == "segmenter" ]] && RUN_SEGMENTER=1
[[ "$MODEL" == "both" || "$MODEL" == "ocr"       ]] && RUN_OCR=1

# -- Paths ------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING="$ROOT/training"
MODEL_DIR="$ROOT/model"
DATA="$ROOT/data"
VENV="$TRAINING/venv"
export PYTHONUTF8=1

# Python path: Windows (Git Bash) uses Scripts/, Unix uses bin/
if   [[ -f "$VENV/Scripts/python.exe" ]]; then PY="$VENV/Scripts/python.exe"
elif [[ -f "$VENV/bin/python"         ]]; then PY="$VENV/bin/python"
else                                           PY=""
fi

# -- Helpers ----------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  [OK]${NC} $*"; }
warn() { echo -e "${YELLOW}  [!] ${NC} $*"; }
step() { echo -e "\n${CYAN}--- $*${NC}"; }
die()  { echo -e "\n${RED}  [ERROR]${NC} $*" >&2; exit 1; }

exists() { [[ $FORCE -eq 0 ]] && [[ -e "$1" ]]; }
png_count() { ls "$1"/*.png 2>/dev/null | wc -l || echo 0; }

# -- 0. Prerequisites -------------------------------------------------------
step "Checking prerequisites"

command -v node >/dev/null 2>&1 || die "node not found - install Node.js"

if [[ $RUN_SEGMENTER -eq 1 ]]; then
    [[ -f "$DATA/jv.txt" ]] || die "data/jv.txt not found - needed for segmenter training"
fi
if [[ $RUN_OCR -eq 1 ]]; then
    [[ -f "$TRAINING/jv_plain.txt" ]] || die "training/jv_plain.txt not found - add your Javanese Wikipedia dump"
    if [[ -f "$TRAINING/PDFA.pdf" ]]; then
        ok "Manuscript PDF found - corpus images will use real parchment backgrounds"
    else
        warn "training/PDFA.pdf not found - corpus images will use plain white backgrounds"
    fi
fi
ok "Prerequisites OK"

# -- 1. Python virtual environment ------------------------------------------
step "Python virtual environment"

if [[ -z "$PY" ]]; then
    if command -v uv &>/dev/null; then
        echo "  Creating venv with uv (Python 3.11)..."
        uv venv "$VENV" --python 3.11

        [[ -f "$VENV/Scripts/python.exe" ]] && PY="$VENV/Scripts/python.exe" || PY="$VENV/bin/python"

        echo "  Installing PyTorch (CPU)..."
        uv pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
        echo "  Installing other dependencies..."
        uv pip install --python "$PY" torch-directml pymupdf -r "$TRAINING/requirements.txt"
    else
        PYTHON_SYS=""
        command -v python3 &>/dev/null && PYTHON_SYS="python3"
        [[ -z "$PYTHON_SYS" ]] && command -v python &>/dev/null && PYTHON_SYS="python"
        [[ -z "$PYTHON_SYS" ]] && die "python not found"

        echo "  Creating venv..."
        "$PYTHON_SYS" -m venv "$VENV"
        [[ -f "$VENV/Scripts/python.exe" ]] && PY="$VENV/Scripts/python.exe" || PY="$VENV/bin/python"

        "$PY" -m pip install -q --upgrade pip
        echo "  Installing PyTorch (CPU)..."
        "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
        echo "  Installing other dependencies..."
        "$PY" -m pip install -q -r "$TRAINING/requirements.txt"
        "$PY" -m pip install -q pymupdf
    fi
    ok "Virtual environment ready"
else
    ok "Virtual environment already exists"
fi

mkdir -p "$MODEL_DIR"

# ==========================================================================
# SEGMENTER
# ==========================================================================

if [[ $RUN_SEGMENTER -eq 1 ]]; then
    step "Training word segmenter (BiLSTM)"

    if ! exists "$MODEL_DIR/segmenter.onnx"; then
        "$PY" "$TRAINING/train.py" "$DATA/jv.txt"
        ok "Segmenter trained -> model/segmenter.onnx"
    else
        ok "Segmenter already trained"
    fi
fi

# ==========================================================================
# OCR MODEL
# ==========================================================================

if [[ $RUN_OCR -eq 1 ]]; then
    # -- Transliterate corpus -----------------------------------------------
    step "Transliterating Latin corpus to Aksara Jawa"

    AKSARA_CORPUS="$TRAINING/javanese_aksara.txt"
    if ! exists "$AKSARA_CORPUS"; then
        node "$TRAINING/transliterate_corpus.js" "$TRAINING/jv_plain.txt" "$AKSARA_CORPUS"
        ok "Corpus transliterated -> training/javanese_aksara.txt"
    else
        ok "Aksara corpus already exists"
    fi

    # -- Train character language model -------------------------------------
    step "Training character n-gram language model"

    LM="$TRAINING/javanese_lm.pkl"
    if ! exists "$LM"; then
        "$PY" "$TRAINING/javanese_ocr.py" \
            --mode        train_lm \
            --corpus      "$AKSARA_CORPUS" \
            --output_path "$LM"
        ok "Language model trained -> training/javanese_lm.pkl"
    else
        ok "Language model already trained"
    fi

    # -- Generate OCR training data -----------------------------------------
    step "Generating OCR training data from corpus"

    OCR_CORPUS="$TRAINING/ocr_corpus"
    COUNT=$(png_count "$OCR_CORPUS")
    if [[ "$COUNT" -lt 10000 ]] || [[ $FORCE -eq 1 ]]; then
        BG_ARG=""
        [[ -f "$TRAINING/PDFA.pdf" ]] && BG_ARG="--background_pdf $TRAINING/PDFA.pdf"
        # shellcheck disable=SC2086
        "$PY" "$TRAINING/javanese_ocr.py" \
            --mode        generate_from_corpus \
            --corpus      "$AKSARA_CORPUS" \
            --data_dir    "$OCR_CORPUS" \
            --num_samples 20000 \
            $BG_ARG
        ok "Generated 20000 training images -> training/ocr_corpus/"
    else
        ok "OCR corpus already generated ($COUNT images)"
    fi

    # -- Train OCR model ----------------------------------------------------
    step "Training OCR model (CRNN + CTC)"

    DATA_DIRS=()
    for d in "$TRAINING/ocr_data" "$TRAINING/ocr_corpus" "$TRAINING/pseudo_labeled"; do
        [[ -d "$d" ]] && DATA_DIRS+=("$d")
    done
    [[ ${#DATA_DIRS[@]} -eq 0 ]] && die "No OCR training data found in training/"

    OCR_PTH="$TRAINING/javanese_ocr.pth"
    "$PY" "$TRAINING/javanese_ocr.py" \
        --mode        train \
        --data_dir    "${DATA_DIRS[@]}" \
        --epochs      50 \
        --lr          0.0003 \
        --output_path "$OCR_PTH"
    ok "OCR model trained -> training/javanese_ocr.pth"

    # -- Export OCR model to ONNX -------------------------------------------
    step "Exporting OCR model to ONNX"

    "$PY" "$TRAINING/javanese_ocr.py" \
        --mode        export_onnx \
        --model_path  "$OCR_PTH" \
        --output_path "$MODEL_DIR/javanese_ocr.onnx"
    ok "OCR model exported -> model/javanese_ocr.onnx"
fi

# -- Done -------------------------------------------------------------------
echo ""
echo -e "${GREEN}--- Training complete ---${NC}"
echo ""
[[ $RUN_SEGMENTER -eq 1 ]] && echo "  model/segmenter.onnx     word segmenter (used by TypeScript)"
[[ $RUN_SEGMENTER -eq 1 ]] && echo "  model/vocab.json         segmenter vocabulary"
[[ $RUN_OCR -eq 1       ]] && echo "  model/javanese_ocr.onnx  OCR model"
echo ""
if [[ $RUN_OCR -eq 1 ]]; then
    echo "Test OCR:"
    echo "  cd training && python javanese_ocr.py --mode predict \\"
    echo "      --pdf PDFA.pdf --lm_path javanese_lm.pkl --beam_width 10"
fi
