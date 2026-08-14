# Aksara.ts Training & Self-Supervised OCR Pipeline

This directory contains the training pipeline for `aksara.ts`, including:

- **Javanese Aksara OCR** — the new TrOCR fine-tune pipeline (`trocr/`)
- **Word segmenter** — BiLSTM model exported to ONNX (`train.py`, `export.py`)
- **CRNN + CTC OCR** — the legacy self-supervised pseudo-labelling loop (`crnn/`)

See [`trocr/README.md`](./trocr/README.md) for the current OCR training path
(fine-tune `microsoft/trocr-small-printed` → Hub
[`thesimonharms/trocr-javanese-synthetic-v6`](https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6)).
The CRNN pipeline below is kept for the TypeScript ONNX runtime and still
works via `training.sh` / `training.ps1`.

---

## 1. Prerequisites & Environment Setup

### Windows (cmd / PowerShell)
Run the setup batch file to create a virtual environment and install dependencies:
```cmd
training\setup.bat
training\venv\Scripts\activate
```

### Linux / macOS / Git Bash
Use `uv` or standard Python `venv`:
```bash
python3 -m venv training/venv
source training/venv/bin/activate
pip install -r training/requirements.txt
pip install pymupdf pillow torchvision
```

Verify your environment with:
```bash
python -c "import fitz, PIL, torch, onnxruntime; print('Environment verified')"
```

---

## 2. The 6-Step Self-Training OCR Loop

Real Javanese manuscripts vary widely in ink bleed, parchment texture, ligature variance, and scribal style. Rather than relying solely on manual transcription, the OCR pipeline uses a **self-supervised iterative pseudo-labelling loop**:

```
[1] generate_from_corpus ──┐
[2] train_lm             ──┼─► [3] train (initial CRNN)
                           │
[4] ingest manuscripts   ──┴─► [5] pseudo_label ──► [6] train (retrain CRNN) ──┐
                                     ▲                                          │
                                     └──────────────────────────────────────────┘
```

### Step 1: Generate Synthetic Training Data (`generate_from_corpus`)
Renders synthetic line strips from plain Javanese Aksara corpus text using real manuscript background textures (`PDFA.pdf` or custom scans).
```bash
python training/crnn/javanese_ocr.py --mode generate_from_corpus \
  --corpus training/javanese_aksara.txt \
  --background_pdf training/PDFA.pdf \
  --data_dir ./ocr_corpus \
  --num_samples 10000
```
- **Output**: `ocr_corpus/*.png` + matching `.txt` label files.

### Step 2: Train Character N-Gram Language Model (`train_lm`)
Trains a character-level 3-gram model with Laplace smoothing for shallow fusion during CTC beam search.
```bash
python training/crnn/javanese_ocr.py --mode train_lm \
  --corpus training/javanese_aksara.txt \
  --output_path training/javanese_lm.pkl
```

### Step 3: Train Initial Acoustic Model (`train`)
Trains the CRNN checkpoint (`javanese_ocr.pth`) on synthetic strips.
```bash
python training/crnn/javanese_ocr.py --mode train \
  --data_dir ./ocr_corpus \
  --epochs 20 \
  --lr 0.001
```

### Step 4: Ingest Unlabeled Manuscript Scans (`ingest`)
Segments raw manuscript PDFs or images into normalized $128 \times 32$ greyscale line strips.
```bash
python training/crnn/javanese_ocr.py --mode ingest \
  --input_dir ./manuscripts \
  --data_dir ./manuscript_strips
```

### Step 5: Auto-Label High-Confidence Strips (`pseudo_label`)
Runs the trained CRNN with LM-assisted beam search on unlabelled strips. Strips exceeding confidence threshold (`--threshold 0.92`) are copied alongside generated `.txt` labels.
```bash
python training/crnn/javanese_ocr.py --mode pseudo_label \
  --unlabeled_dir ./manuscript_strips \
  --data_dir ./pseudo_labeled \
  --lm_path training/javanese_lm.pkl \
  --threshold 0.92
```

### Step 6: Retrain with Expanded Dataset (`train`)
Retrains the CRNN combining synthetic pre-training data and confident pseudo-labelled real manuscript strips.
```bash
python training/crnn/javanese_ocr.py --mode train \
  --data_dir ./ocr_corpus ./pseudo_labeled \
  --epochs 25 \
  --output_path training/javanese_ocr.pth
```
Repeat Steps 5–6 iteratively: each improved model labels additional difficult strips.

---

## 3. Minimum Viable Fast-Track Sequence (< 1 Hour CPU)

To produce a functional model from scratch on CPU in under 45 minutes:

```bash
# 1. Generate 2,000 synthetic samples (~3 minutes)
python training/crnn/javanese_ocr.py --mode generate_from_corpus \
  --corpus training/javanese_aksara.txt --background_pdf training/PDFA.pdf \
  --data_dir ./ocr_fast --num_samples 2000

# 2. Train 3-gram language model (< 5 seconds)
python training/crnn/javanese_ocr.py --mode train_lm \
  --corpus training/javanese_aksara.txt --output_path training/javanese_lm.pkl

# 3. Train CRNN for 10 epochs (~25 minutes on CPU)
python training/crnn/javanese_ocr.py --mode train \
  --data_dir ./ocr_fast --epochs 10 --output_path training/javanese_ocr.pth

# 4. Export canonical ONNX artifact to model/javanese_ocr.onnx (< 5 seconds)
python training/crnn/javanese_ocr.py --mode export_onnx \
  --model_path training/javanese_ocr.pth \
  --output_path ../model/javanese_ocr.onnx
```

---

## 4. Resource & Time Budgets

| Task | CPU Time (Standard 8-core) | GPU Time (DirectML / CUDA) | Disk Storage |
| :--- | :--- | :--- | :--- |
| **Synthetic Strip Gen (10k samples)** | 8–12 min | 8–12 min (CPU-bound) | ~45 MB |
| **LM Training (50 MB corpus)** | < 10 sec | < 10 sec | ~2 MB (`.pkl`) |
| **CRNN Training (10k strips, 20 epochs)** | 40–60 min | 5–8 min | ~3.5 MB (`.pth`) |
| **ONNX Export** | < 5 sec | < 5 sec | ~3.3 MB (`.onnx`) |
