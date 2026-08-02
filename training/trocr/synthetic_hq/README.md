# Synthetic HQ Javanese Aksara OCR dataset

Higher-quality **synthetic** line data for a future cook after v4. Fonts are
already exhausted; this pipeline spends variety budget on **length mix**,
**held-out text**, **backgrounds**, and **degrade augments**.

**Do not run a full 500k build / Hub upload until explicitly asked.**

## Targets

| Knob | Value |
|------|--------|
| Hub dataset | `thesimonharms/javanese-synthetic-hq` (**private**) |
| Scale | 500_000 train / 5_000 val (flags adjustable) |
| Local dir | `training/trocr/trocr_dataset_hq/` (gitignored) |
| Old generators | unchanged (`generate_trocr_dataset.py`, `build_large_dataset.py`) |

## Process

```
javanese_corpus_ocr.txt (+ optional dumps)
        │  corpus_hq_prepare.py
        ▼
corpus_hq/{corpus_hq_train,val}.{txt,jsonl}   # hash holdout 5% val text
        │  build_synthetic_hq.py
        ▼
trocr_dataset_hq/{train,validation}/          # PNG+TXT+manifest
        │  --export-parquet [--push]
        ▼
HF private: thesimonharms/javanese-synthetic-hq
```

### Quotas (generate-time)

- **15% short** (≤8 chars) / **55% mid** / **30% long**
- Soft oversample lines tagged `rare` (pengkal / less-common aksara)
- Prefer unique text before repeats (`ceil(N / pool)` cap)

### Render upgrades vs v1–v4 generator

- Load up to **50 pages/PDF** + loose PNG/JPG textures
- ~70% manuscript crop / ~30% procedural paper
- CRNN-grade degrade: Gaussian + S&P noise, ink bleed/erosion, JPEG recompress, mild aliasing
- Quality gate: reject blank / low-contrast / border-clipped; retry up to 6×

## Commands (scaffold — run when ready)

```bash
cd training/trocr

# 1) Text pools (fast, safe anytime)
python corpus_hq_prepare.py \
  --inputs ../javanese_corpus_ocr.txt \
  --out-dir ../corpus_hq

# 2) Dry-run plans only (no images)
python build_synthetic_hq.py --dry-run

# 3) Full local render (long)
python build_synthetic_hq.py \
  --num_train 500000 --num_val 5000 \
  --workers 8 --seed 42

# 4) Private Hub publish (explicit)
python build_synthetic_hq.py \
  --skip_generate --export-parquet --push \
  --repo_id thesimonharms/javanese-synthetic-hq
```

Uploads always call `create_repo(..., private=True)`. There is no `--public` path.

## Future cook consumption

First cook on this set should be **hq ×1 only** (no upsample of old 60k/180k), so the new distribution is not drowned. Optionally add Nusa ×1 later as a real-domain sniff — keep HITL / Nusa as separate tracks.

Example (later):

```bash
DATASET_NAME=thesimonharms/javanese-synthetic-hq
# no EXTRA_DATASETS / upsample until hq alone is scored
```

## Layout

```
synthetic_hq/
  __init__.py
  sampler.py      # stratified + rare boost
  render.py       # backgrounds + augments
  quality.py      # reject gates
  README.md       # this file
../corpus_hq_prepare.py
../build_synthetic_hq.py
```
