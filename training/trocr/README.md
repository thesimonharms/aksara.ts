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
  finetune_trocr.py           Seq2SeqTrainer fine-tuning script (Jobs + local)
  push_dataset.py             Push local imagefolder to a **private** Hub dataset
  jobs.md                     HF Jobs smoke + full-train recipes (production path)
  nas/                        Linux NAS Docker hands-off v2 retrain (Intel XPU)
  requirements.txt            Pinned deps for the importable pipeline (torch-wheel comment inside)
  README.md                   This document
  trocr_dataset/              (generated, gitignored) — output of generate_trocr_dataset.py
  trocr_ckpt/                 (generated, gitignored) — Trainer checkpoints + final model
  space/                      HF Space SDK=docker — optional UI only; do not train here
```

```
training/fonts/               Drop your .ttf / .otf Javanese Aksara fonts here (gitignored)
training/pdfs/                Drop scanned manuscript .pdf files here (gitignored)
```

## Three paths

| Path | Compute | Use when |
|------|---------|----------|
| **Local Intel Arc (XPU)** | Arc Pro B60 eGPU via `torch.xpu` | **Inference** + local fine-tunes (`.venv-xpu`) |
| **HF Jobs** | rented L4 / A10G via `hf jobs` | Cloud fine-tunes when local GPU is busy |
| **HF Space** | Gradio docker Space | Optional demo / labeling UI — **not for training** |

Setup + run commands: **[`jobs.md`](jobs.md)** (Local Intel Arc section first).

Both Jobs and local share `finetune_trocr.py`. Verify scripts (`verify_trocr.py`,
`local_verify_large.py`) auto-pick `xpu` when the Arc stack is installed.

**Licensing:** the Hub dataset (`javanese-dataset`) is pushed **private** by
default — you can train on it, but it is not redistributed publicly. The
fine-tuned **model** weights may still be published openly if that matches your
goals.

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

## HF Jobs path (recommended for real training)

See **[`jobs.md`](jobs.md)**. Short version:

1. Push a **private** dataset: `python push_dataset.py --repo_id <you>/javanese-dataset`
2. Run a smoke Job (200 samples, 15m timeout on `l4x1`), then the full 5-epoch
   Job on **`a10g-large`** with batch 24 (L4 OOMs at that batch).
3. Confirm the Hub model repo is non-empty; pause any leftover GPU Space.

## HF Space path (optional UI only — do not train here)

The `space/` subdirectory has a Gradio docker Space for demos / labeling.
Training on Spaces previously hung for hours without reliable Hub push — prefer
Jobs. If you still use the Space UI, see [`space/README.md`](space/README.md)
and pause the Space when idle.

## Publishing targets

| Artifact | Default HF Hub id | Visibility |
|----------|------|---|
| Dataset  | `{HF_USERNAME}/javanese-dataset` (`push_dataset.py` / `--push_dataset`) | **Private** by default |
| Model    | `{HF_USERNAME}/javanese-trocr-handwritten` (override with `--hub_model_id`) | Whatever you set on the model repo |

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