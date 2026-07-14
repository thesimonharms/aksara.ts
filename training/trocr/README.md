# TrOCR Fine-tune — Javanese Aksara

Fine-tune `microsoft/trocr-base-handwritten` for Javanese Aksara OCR via the
transformers `Seq2SeqTrainer` (AutoTrain Advanced is deprecated and never
supported a VisionEncoderDecoder task).

This folder is the new home for the OCR training pipeline. The previous CRNN +
CTC pipeline still works — its scripts are kept under
[`../crnn/`](../crnn/) and the existing `training.sh` / `training.ps1`
entries have been repointed there.

## Layout

```
training/trocr/
  generate_trocr_dataset.py   Synthetic dataset generator (fonts + PDFs → imagefolder)
  label_pdfs.py               HITL labeler — auto-detects manuscript line strips and
                              lets you type Latin transliterations via a Gradio UI
  finetune_trocr.py           Seq2SeqTrainer fine-tuning script
  requirements.txt            Pinned deps for the importable pipeline (torch-wheel comment inside)
  README.md                   This document
  trocr_dataset/              (generated, gitignored) — output of generate_trocr_dataset.py
  trocr_ckpt/                 (generated, gitignored) — Trainer checkpoints + final model
  space/                      HF Space SDK=docker config — see "HF Space path" below
```

```
training/fonts/               Drop your .ttf / .otf Javanese Aksara fonts here (gitignored)
training/pdfs/                Drop scanned manuscript .pdf files here (gitignored)
```

## Two paths

| Path | Compute | Use when |
|------|---------|----------|
| **Local** | your CPU / DirectML (no CUDA needed) | Quick 1-epoch smoke runs, debugging the dataset, iterating on augmentation |
| **HF Space** | rented T4 GPU in a HF Space | Real fine-tune runs that actually beat the CRNN baseline |

Both paths share the same `finetune_trocr.py`; only the entry-point differs.

## Secrets / .env

Locally, copy `../../.env.example` → `../../.env` and fill in:

```dotenv
HF_TOKEN=hf_<your write-perm token>
HF_USERNAME=<your-hf-username>
AUTOTRAIN_PROJECT_NAME=javanese-trocr-handwritten
BASE_MODEL=microsoft/trocr-base-handwritten
```

`.env` is in the repo `.gitignore`; `.env.example` is committed so the schema
stays visible.

In a HF Space, do **not** upload `.env`. Instead set `HF_TOKEN` and
`HF_USERNAME` as **Space Secrets** — the script's `load_dotenv(...)` becomes a
no-op and the secrets populate the environment for the script exactly the same
way.

## Local path (DirectML or CPU)

### 1. Install deps

From the repo root, create the venv as `training.sh` does, then add the trocr pipeline:

```powershell
# Windows / DirectML (AMD Strix Halo, Radeon, etc.)
.\training.ps1 -Model segmenter   # bootstraps venv + trains segmenter (one-time)
.\training\venv\Scripts\python.exe -m pip install torch torch-directml
.\training\venv\Scripts\python.exe -m pip install -r training\trocr\requirements.txt
```

```bash
# Linux / macOS / CPU
bash training.sh --model segmenter   # bootstraps venv
./training/venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
./training/venv/bin/python -m pip install -r training/trocr/requirements.txt
```

### 2. Drop fonts and manuscript scans (PDF / PNG / JPG)

```
training/fonts/*.{ttf,otf}             ← fonts render the Aksara text into line images
training/pdfs/*.{pdf,png,jpg,jpeg}     ← manuscript scans used as background textures & HITL labeling sources
```

### 3. Generate the dataset

```bash
cd training/trocr
python generate_trocr_dataset.py \
    --corpus  ../javanese_corpus_clean.txt \
    --fonts_dir ../fonts \
    --pdfs_dir  ../pdfs \
    --output_dir trocr_dataset \
    --num_train 5000 \
    --num_val 500
```

This writes `trocr_dataset/train/{*.png, metadata.jsonl}` and
`trocr_dataset/validation/{*.png, metadata.jsonl}` — Hugging Face `imagefolder`
format, directly usable by `load_dataset("imagefolder", ...)`.

### 4. (Optional, but high-ROI) Label real handwriting with the HITL labeler

Synthetic data alone hits a ceiling — the model never sees real ink, real
glyph variation, or real parchment texture as the *target* to read. A few
hundred lines of **real handwriting with paired transliterations** close that
domain gap dramatically.

```bash
# Make sure gradio is installed (already in space/requirements.txt)
cd training/trocr
python label_pdfs.py --pdfs_dir ../pdfs --output_dir ../pdf_labeled
# then open http://127.0.0.1:7861 in your browser
```

Workflow in the browser:

1. The labeler auto-detects each text-line strip in your manuscript scans
   using a horizontal dark-pixel projection (works well on palm-leaf lontar
   and codex scans at ≥150 DPI).
2. The strip preview shows what will be saved; the page on the right has a
   red box around the candidate strip.
3. Type the Latin transliteration, press **Submit + next**.
4. If auto-detect got the strip wrong (too tight / merged two lines /
   included a margin), drag the **Y start / Y end** sliders to fix it and
   press **Apply manual bounds** before submitting.
5. **Skip** drops a strip without labeling — use freely for marginalia,
   page numbers, or unreadable ink.

Pairs save as `training/pdf_labeled/label_XXXXXX.png` + `.txt`. The
directory is **gitignored** like the old `training/human_labeled/` — your
hand-labeled pairs stay local unless you deliberately publish via
`finetune_trocr.py --push_dataset`.

The labeler is **resumable** — close with Ctrl+C, restart it later, and
already-labeled strips are skipped automatically.

**Target ROI:** ~200–500 hand-labeled lines mixed into 5k synthetic samples
produces a model that generalizes to real handwriting far better than 10
more fonts ever could. Budget ~2 hours of labeling for 300 lines.

### 5. Fine-tune (CPU / DirectML — slow, but works)

```bash
cd training/trocr
python finetune_trocr.py \
    --dataset_dir trocr_dataset \
    --pdf_labeled_dir ../pdf_labeled \
    --upsample_labeled 5 \
    --epochs 3 \
    --no_push                 # toggle off once HF_TOKEN / HF_USERNAME are set
```

Expect roughly 10–15 min/epoch on 5k samples with DirectML (AMD Strix Halo iGPU).
The `--upsample_labeled 5` flag duplicates your ~hundreds of real-handwriting
pairs 5× so they aren't drowned out by the thousands of synthetic samples;
tune to taste once you see val-CER.

## HF Space path (T4 GPU — recommended for the "real" run)

The `space/` subdirectory has a complete HF Space SDK=docker config: a
Dockerfile, a Gradio app front-end, pinned CUDA requirements, and a README with
HF frontmatter. See [`space/README.md`](space/README.md) for the step-by-step.

Short version:

1. Create a HF Space (SDK **docker**, hardware **T4 small**) — copy the four
   scripts in `space/` plus `generate_trocr_dataset.py` and `finetune_trocr.py`
   from this folder into the Space repo root.
2. Set `HF_TOKEN` and `HF_USERNAME` as Space Secrets.
3. Upload your `.ttf` / `.otf` to `fonts/` and your `.pdf`s to `pdfs/` *in the
   Space repo* (your gitignored local copies never need to leave your machine).
4. Click **1. Generate dataset** → **2. Push dataset** → **3. Run fine-tuning**.

Net cost on a $10 budget at T4 small (~$0.40/h):
**~25 hours of training**, e.g. 5 epochs on 5k samples ≈ $0.40 / run.

## Publishing targets

| Artifact | Default HF Hub id |
|----------|------|
| Dataset  | `{HF_USERNAME}/javanese-dataset` (toggle with `--push_dataset`) |
| Model    | `{HF_USERNAME}/javanese-trocr-handwritten` (override with `--hub_model_id`) |

## Tips for improving the dataset

- **More fonts = more style coverage.** TrOCR generalizes across typefaces
  much better when the synthetic corpora vary in stroke weight, glyph
  proportions, and serif/sans style. Aim for ≥ 3 fonts before training.
- **PDF backgrounds are doing real work.** Each manuscript scan provides
  textured paper backgrounds that teach the model to ignore non-text ink, EMS,
  aging, etc. Even a handful of pages materially helps.
- `generate_trocr_dataset.py` is intentionally simple — augment further
  (noise, aliasing, scissors/curvature) by editing its `_apply_augmentations`.
- The base model is pre-trained on English handwriting, so the model mainly has
  to **unlearn** its English decoder prior. **5+ epochs** is a reasonable starting
  point; watch CER on the validation split. If val-CER stops dropping before
  train-loss converges, you're over-fitting fonts — back off epochs / add fonts.