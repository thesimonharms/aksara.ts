# training.ps1 - Training pipeline for aksara.ts
#
# Run from the project root:
#
#   .\training.ps1                  # train both models
#   .\training.ps1 -Model segmenter # train word segmenter only
#   .\training.ps1 -Model ocr       # train OCR model only
#   .\training.ps1 -Force           # re-run all steps even if outputs exist
#
# Prerequisites:
#   - Node.js  (corpus transliteration)
#   - Python 3.11 via uv (venv is created automatically)
#   - training\jv_plain.txt  - Javanese Wikipedia dump (Latin script)
#   - data\jv.txt            - compact Javanese corpus for the segmenter
#   - training\PDFA.pdf      - optional: manuscript PDF for background textures

param(
    [ValidateSet("both", "segmenter", "ocr")]
    [string]$Model = "both",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = "1"

$RunSegmenter = $Model -eq "both" -or $Model -eq "segmenter"
$RunOcr       = $Model -eq "both" -or $Model -eq "ocr"

# -- Paths ------------------------------------------------------------------
$Root     = $PSScriptRoot
$Training = Join-Path $Root "training"
$ModelDir = Join-Path $Root "model"
$Data     = Join-Path $Root "data"
$Venv     = Join-Path $Training "venv"
$Py       = Join-Path $Venv "Scripts\python.exe"

# -- Helpers ----------------------------------------------------------------
function Ok($msg)   { Write-Host "  [OK] $msg"      -ForegroundColor Green  }
function Warn($msg) { Write-Host "  [!]  $msg"      -ForegroundColor Yellow }
function Step($msg) { Write-Host "`n---  $msg"      -ForegroundColor Cyan   }
function Die($msg)  { Write-Host "`n  [ERROR] $msg" -ForegroundColor Red; exit 1 }

function Count-Pngs($dir) {
    if (-not (Test-Path $dir)) { return 0 }
    return (Get-ChildItem -Path $dir -Filter "*.png" -ErrorAction SilentlyContinue).Count
}

function Exists($path) {
    return (-not $Force) -and (Test-Path $path)
}

# -- 0. Prerequisites -------------------------------------------------------
Step "Checking prerequisites"

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Die "node not found - install Node.js from https://nodejs.org"
}
if ($RunSegmenter -and -not (Test-Path (Join-Path $Data "jv.txt"))) {
    Die "data\jv.txt not found - needed for segmenter training"
}
if ($RunOcr -and -not (Test-Path (Join-Path $Training "jv_plain.txt"))) {
    Die "training\jv_plain.txt not found - add your Javanese Wikipedia dump"
}
if ($RunOcr) {
    if (Test-Path (Join-Path $Training "PDFA.pdf")) {
        Ok "Manuscript PDF found - corpus images will use real parchment backgrounds"
    } else {
        Warn "training\PDFA.pdf not found - corpus images will use plain white backgrounds"
    }
}
Ok "Prerequisites OK"

# -- 1. Python virtual environment ------------------------------------------
Step "Python virtual environment"

if (-not (Test-Path $Py) -or $Force) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "  Creating venv with uv (Python 3.11)..."
        uv venv $Venv --python 3.11

        Write-Host "  Installing PyTorch (CPU)..."
        uv pip install --python $Py torch --index-url https://download.pytorch.org/whl/cpu

        Write-Host "  Installing other dependencies..."
        uv pip install --python $Py torch-directml pymupdf -r (Join-Path $Training "requirements.txt")
    } else {
        $PythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
        if (-not $PythonCmd) { $PythonCmd = Get-Command python -ErrorAction SilentlyContinue }
        if (-not $PythonCmd) { Die "python not found" }
        $PythonSys = $PythonCmd.Source

        Write-Host "  Creating venv..."
        & $PythonSys -m venv $Venv

        Write-Host "  Installing PyTorch (CPU)..."
        & $Py -m pip install -q --upgrade pip
        & $Py -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu

        Write-Host "  Installing other dependencies..."
        & $Py -m pip install -q -r (Join-Path $Training "requirements.txt")
        & $Py -m pip install -q pymupdf
    }

    Ok "Virtual environment ready"
} else {
    Ok "Virtual environment already exists"
}

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

# ==========================================================================
# SEGMENTER
# ==========================================================================

if ($RunSegmenter) {
    # -- 2. Train word segmenter --------------------------------------------
    Step "Training word segmenter (BiLSTM)"

    $SegmenterOnnx = Join-Path $ModelDir "segmenter.onnx"
    if (-not (Exists $SegmenterOnnx)) {
        & $Py (Join-Path $Training "train.py") (Join-Path $Data "jv.txt")
        Ok "Segmenter trained -> model\segmenter.onnx"
    } else {
        Ok "Segmenter already trained"
    }
}

# ==========================================================================
# OCR MODEL
# ==========================================================================

if ($RunOcr) {
    # -- 3. Transliterate corpus --------------------------------------------
    Step "Transliterating Latin corpus to Aksara Jawa"

    $AksaraCorpus = Join-Path $Training "javanese_aksara.txt"
    if (-not (Exists $AksaraCorpus)) {
        node (Join-Path $Training "transliterate_corpus.js") `
             (Join-Path $Training "jv_plain.txt") `
             $AksaraCorpus
        Ok "Corpus transliterated -> training\javanese_aksara.txt"
    } else {
        Ok "Aksara corpus already exists"
    }

    # -- 4. Train character language model ----------------------------------
    Step "Training character n-gram language model"

    $LM = Join-Path $Training "javanese_lm.pkl"
    if (-not (Exists $LM)) {
        & $Py (Join-Path $Training "javanese_ocr.py") `
            --mode        train_lm `
            --corpus      $AksaraCorpus `
            --output_path $LM
        Ok "Language model trained -> training\javanese_lm.pkl"
    } else {
        Ok "Language model already trained"
    }

    # -- 5. Generate OCR training data --------------------------------------
    Step "Generating OCR training data from corpus"

    $OcrCorpus = Join-Path $Training "ocr_corpus"
    $PngCount  = Count-Pngs $OcrCorpus
    if ($PngCount -lt 10000 -or $Force) {
        $BgArg = @()
        $PdfPath = Join-Path $Training "PDFA.pdf"
        if (Test-Path $PdfPath) { $BgArg = @("--background_pdf", $PdfPath) }

        & $Py (Join-Path $Training "javanese_ocr.py") `
            --mode        generate_from_corpus `
            --corpus      $AksaraCorpus `
            --data_dir    $OcrCorpus `
            --num_samples 20000 `
            @BgArg
        Ok "Generated 20000 training images -> training\ocr_corpus\"
    } else {
        Ok "OCR corpus already generated ($PngCount images)"
    }

    # -- 6. Train OCR model -------------------------------------------------
    Step "Training OCR model (CRNN + CTC)"

    $DataDirs = @()
    foreach ($sub in @("ocr_data", "ocr_corpus", "pseudo_labeled")) {
        $p = Join-Path $Training $sub
        if (Test-Path $p) { $DataDirs += $p }
    }
    if ($DataDirs.Count -eq 0) { Die "No OCR training data found in training\" }

    $OcrPth = Join-Path $Training "javanese_ocr.pth"
    & $Py (Join-Path $Training "javanese_ocr.py") `
        --mode        train `
        --data_dir    @DataDirs `
        --epochs      50 `
        --lr          0.0003 `
        --output_path $OcrPth
    Ok "OCR model trained -> training\javanese_ocr.pth"

    # -- 7. Export OCR model to ONNX ----------------------------------------
    Step "Exporting OCR model to ONNX"

    $OcrOnnx = Join-Path $ModelDir "javanese_ocr.onnx"
    & $Py (Join-Path $Training "javanese_ocr.py") `
        --mode        export_onnx `
        --model_path  $OcrPth `
        --output_path $OcrOnnx
    Ok "OCR model exported -> model\javanese_ocr.onnx"
}

# -- Done -------------------------------------------------------------------
Write-Host ""
Write-Host "--- Training complete ---" -ForegroundColor Green
Write-Host ""
if ($RunSegmenter) {
    Write-Host "  model\segmenter.onnx     word segmenter (used by TypeScript)"
    Write-Host "  model\vocab.json         segmenter vocabulary"
}
if ($RunOcr) {
    Write-Host "  model\javanese_ocr.onnx  OCR model"
}
Write-Host ""
if ($RunOcr) {
    Write-Host "Test OCR:"
    Write-Host "  cd training && python javanese_ocr.py --mode predict ``"
    Write-Host "      --pdf PDFA.pdf --lm_path javanese_lm.pkl --beam_width 10"
}
