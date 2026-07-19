---
title: Javanese TrOCR Fine-tune
emoji: 🔤
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
hardware: t4-small
license: mit
tags:
  - trocr
  - javanese
  - aksara
  - ocr
  - fine-tuning
---

# Javanese TrOCR Fine-tune Space

Fine-tunes `microsoft/trocr-base-handwritten` on a Javanese Aksara (Old Javanese script)
OCR dataset, then pushes the fine-tuned model to your HF account.

## Why this Space

Hugging Face AutoTrain Advanced is deprecated and never had a first-class
VisionEncoderDecoder task. This Space runs the proper path — transformers'
`Seq2SeqTrainer` — on a rented T4 GPU.

## Hardware

Set **Space → Settings → Hardware** to **T4 small** (~$0.40/h). A 5-epoch
fine-tune on ~5k samples finishes in roughly one hour, so your $10 of HF credits
covers many runs.

## Required Secrets (Space → Settings → Repository secrets)

| Secret | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | yes | Write-scoped token — push model + dataset |
| `HF_USERNAME` | yes | HF user/org name; model publishes as `{username}/javanese-trocr-handwritten` |

Optional secrets:
- `BASE_MODEL` (default `microsoft/trocr-base-handwritten`)
- `EPOCHS` (default `5`)
- `PER_DEVICE_TRAIN_BATCH_SIZE` (default `8`)
- `DATASET_NAME` (HF Hub dataset id — skips local generation, loads from Hub)
- `HUB_MODEL_ID` (HF Hub model id — overrides the default `{HF_USERNAME}/javanese-trocr-handwritten`.
  Use this for experiment variants, e.g. `{HF_USERNAME}/trocr-javanese-synthetic` for a
  synthetic-only preliminary run. You can also override it per-run via the UI text box;
  the literal string `{HF_USERNAME}/...` is resolved against the `HF_USERNAME` secret.)

## What to upload to the Space repo

This Space expects the following layout (files live in the Space repo root, not
in aksara.ts's `training/trocr/space/`):

```
Dockerfile
app.py
finetune_trocr.py
generate_trocr_dataset.py
requirements.txt
README.md           (this file)
fonts/              ← upload your .ttf / .otf Javanese Aksara fonts here
pdfs/               ← upload scanned manuscript .pdf files here
```

`fonts/` and `pdfs/` are empty in the Space's git repo until you upload your
files. They're independent of aksara.ts's local gitignored `fonts/`/`pdfs/` —
**nothing from your gitignored folders needs to leak to GitHub**; the Space is
its own private repo on HF.

## Workflow

1. **Create the Space.** Mirror the files from `training/trocr/space/` to its
   repo root *plus* copy `training/trocr/generate_trocr_dataset.py` and
   `training/trocr/finetune_trocr.py`.
2. **Set Hardware** to T4 small and the two Secrets above.
3. **(Optional) Test first:** click **1. Generate dataset** with a small sample
   count (e.g. 500 train / 100 val) to confirm fonts/PDFs load.
4. **Generate the full dataset:** `5000` train / `500` val is a sensible start.
5. **Push dataset to Hub:** click **2. Push** so future runs can skip step 4
   by setting `DATASET_NAME` = `<your-username>/javanese-dataset`.
6. **Fine-tune:** click **3. Run fine-tuning**. Watch **Space → Logs** for live
   loss/CER. The finished model lands at `<your-username>/javanese-trocr-handwritten`.

## Notes

- Training outputs are  written to `/app/trocr_ckpt/` inside the container —
  ephemeral, but the model is also pushed to Hub so it survives Space restarts.
- Drafts and intermediate Trainer checkpoints (`checkpoint-*`) live in
  `/app/trocr_ckpt/`; fine-tuned model artifacts live at `/app/trocr_ckpt/final/`.

## Cost guardrails

With $10 of HF credit on T4 small (~$0.40/h):

| Run length | Approx cost |
|---|---|
| 1 epoch, 1k samples (~5 min) | ~$0.04 |
| 5 epochs, 5k samples (~1 h)   | ~$0.40 |
| 20 epochs, 10k samples (~4 h) | ~$1.60 |