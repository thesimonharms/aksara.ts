# TrOCR on Hugging Face Jobs

> **Current model is v6**, trained on the NAS, not Jobs:
> [`thesimonharms/trocr-javanese-synthetic-v6`](https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6).
> Recipes below are historical (v1/v2/smoke). Do not publish over v6.

Production fine-tunes run as **HF Jobs**, not Spaces. Jobs survive laptop sleep,
keep full logs on the [Jobs page](https://huggingface.co/jobs), and hard-stop via
`--timeout` so a hung run cannot burn hours of credits.

| Path | Use for |
|------|---------|
| **Local Intel Arc (XPU)** | Inference + local fine-tunes on Arc Pro B60 eGPU |
| **NAS Docker (Linux XPU)** | **Hands-off v6 cook** — see [`nas/README.md`](nas/README.md) |
| **HF Jobs** (this doc) | Cloud GPU fine-tunes when local GPU is busy / unavailable |

## Local Intel Arc (Windows XPU)

Driver must expose the Arc card (Arc Pro B60 is Battlemage — supported by
PyTorch `torch.xpu`). Use a dedicated venv; do **not** mix with CUDA/DirectML
torch wheels.

```powershell
cd training\trocr
py -3.12 -m venv .venv-xpu
.\.venv-xpu\Scripts\pip install --upgrade pip
.\.venv-xpu\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
.\.venv-xpu\Scripts\pip install -r requirements.txt
```

Smoke that the B60 is visible:

```powershell
.\.venv-xpu\Scripts\python.exe -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0) if torch.xpu.is_available() else None)"
```

### Inference (prefer this over HF Jobs verify)

```powershell
cd training\trocr
$env:HUB_MODEL_ID = "thesimonharms/trocr-javanese-synthetic-v6"
$env:N_SAMPLES = "1500"
.\.venv-xpu\Scripts\python.exe local_verify_large.py
# or quick Hub smoke:
.\.venv-xpu\Scripts\python.exe verify_trocr.py
```

### Local train (1-epoch chunk)

**Working recipe on Arc Pro B60 (Windows `torch.xpu`):**
- `attn_implementation=eager` (auto on XPU) — default SDPA hangs/crashes in backward
- **fp32** (no bf16/fp16 AMP — bf16 breaks TrOCR backward on this stack)
- `dataloader_num_workers=0` on Windows
- batch **8** stable; batch **16** can `DEVICE_LOST` after a step

Expect ~1.5–3 s/it after warmup → roughly **10–15 h** per epoch on the 180k+Nusa mix
(vs ~1 h on A10). Script auto-applies eager + fp32 when XPU is detected.

```powershell
cd training\trocr
.\.venv-xpu\Scripts\python.exe finetune_trocr.py `
  --base_model thesimonharms/trocr-javanese-synthetic-v2 `
  --dataset_name thesimonharms/javanese-dataset-180k `
  --extra_dataset_name thesimonharms/javanese-nusaaksara-ocr `
  --extra_dataset_upsample 8 `
  --hub_model_id thesimonharms/trocr-javanese-synthetic-v2 `
  --epochs 1 `
  --batch_size 8 `
  --lr 2e-5 `
  --warmup_ratio 0 `
  --no-gradient_checkpointing `
  --eval_every_epochs 1 `
  --skip_final_cer `
  --pdf_labeled_dir none
```

Use `--no_push` for a dry run that never touches the Hub.
Quick probe: `.\.venv-xpu\Scripts\python.exe probe_xpu_train_step.py --attn eager --amp off --steps 2 --batch 8`

---

# TrOCR on Hugging Face Jobs

Cloud fine-tunes via **HF Jobs** when you need fire-and-forget rented GPUs.
Jobs survive laptop sleep, keep full logs on the [Jobs page](https://huggingface.co/jobs),
and hard-stop via `--timeout`.

## Prerequisites

1. Top up **$10–15** HF credits ([billing](https://huggingface.co/settings/billing)).
2. Write-scoped token; locally `hf auth login` (or `HF_TOKEN` in `../../.env`).
3. Private dataset on the Hub (manuscript-derived data must stay private):

```powershell
cd training/trocr
..\venv\Scripts\python.exe push_dataset.py --repo_id thesimonharms/javanese-dataset
# default is private; only pass --public if you have redistribution rights
```

Current dataset: `thesimonharms/javanese-dataset` (**private**). Jobs load it
with your `HF_TOKEN` secret — other accounts cannot download it.

4. Pause any GPU **Space** so it cannot idle-bill.

## Smoke Job (~5 min, ~$0.07 on L4)

Verifies CUDA + private dataset download + Hub model push before a long run.
Detach anytime with Ctrl+C — the job keeps running.

```powershell
hf jobs uv run `
  --flavor l4x1 `
  --timeout 15m `
  --secrets HF_TOKEN `
  training/trocr/finetune_trocr.py `
  --dataset_name thesimonharms/javanese-dataset `
  --hub_model_id thesimonharms/trocr-javanese-synthetic-smoke `
  --epochs 1 `
  --batch_size 24 `
  --max_train_samples 200 `
  --pdf_labeled_dir none
```

(`finetune_trocr.py` carries UV inline deps, so extra `--with` flags are optional.
Use `--pdf_labeled_dir none` rather than an empty string — Jobs/CLI strips `""`.)

Watch: [huggingface.co/jobs](https://huggingface.co/jobs) or `hf jobs logs <JOB_ID>`.
Cancel: `hf jobs cancel <JOB_ID>`.

Success looks like logs containing `[OK] Model pushed` and a **non-empty** model
repo at `thesimonharms/trocr-javanese-synthetic-smoke`.

## Full train (cleaned ~50k dataset, a10g-large)

Corpus is cleaned via `clean_javanese_corpus.py` → `javanese_corpus_ocr.txt`
(max 48 chars, wiki/HTML stripped). Private Hub dataset keeps **~60k train
shards / 2k val**; Jobs cap with `--max_train_samples 50000`.

Batch **24** without gradient checkpointing on `a10g-large`, with **on-the-fly**
image encode (no pixel_values map cache). Smoke measured ~**2.4 it/s**.
**50k × 20 epochs** ≈ **6–8 hours** / **~$9–$12**.

Tokenizer expansion is **on by default** (`--expand_javanese_tokenizer`): adds the
Javanese Unicode block as atomic tokens so free-run generation does not scramble
UTF-8 byte pieces. Retrain is required after this change — old checkpoints used
the byte-level vocab.

```powershell
hf jobs uv run `
  --flavor a10g-large `
  --timeout 24h `
  --secrets HF_TOKEN `
  training/trocr/finetune_trocr.py `
  --dataset_name thesimonharms/javanese-dataset `
  --hub_model_id thesimonharms/trocr-javanese-synthetic `
  --epochs 20 `
  --batch_size 24 `
  --no-gradient_checkpointing `
  --max_train_samples 50000 `
  --eval_every_epochs 2 `
  --pdf_labeled_dir none
```

Defaults that keep wall-clock down:

- Mid-train eval is **loss-only** every 2 epochs (no beam-search CER).
- CER runs once at the end.
- Checkpoints **push every save** (`hub_strategy=every_save`), so a timeout
  still leaves weights on the Hub.

Pass `--predict_with_generate` only if you want slow mid-train CER.

## Continue-finetune v2 (length curriculum → `trocr-javanese-synthetic-v2`)

Resume from the **repo-root** expanded checkpoint (vocab ~50361), not `final/`
(stale 50265). Builds a **120k** train mix with **~25% short lines (≤8 chars)**
via sampling-with-replacement from the Hub 60k pool, then continues for fewer
epochs at a lower LR.

```powershell
hf jobs uv run `
  --flavor a10g-large `
  --timeout 16h `
  --secrets HF_TOKEN `
  training/trocr/finetune_trocr.py `
  --base_model thesimonharms/trocr-javanese-synthetic `
  --dataset_name thesimonharms/javanese-dataset `
  --hub_model_id thesimonharms/trocr-javanese-synthetic-v2 `
  --epochs 8 `
  --batch_size 24 `
  --lr 2e-5 `
  --no-gradient_checkpointing `
  --max_train_samples 120000 `
  --short_line_fraction 0.25 `
  --short_line_max_chars 8 `
  --eval_every_epochs 2 `
  --pdf_labeled_dir none
```

(Base model already has the expanded Javanese vocab; leave tokenizer expansion at its default no-op.)

Expect ~**5–7 hours** on `a10g-large`. Final weights are uploaded to the **v2
repo root** (load with no `subfolder=`).

If a run is killed mid-train (exit 143), resume from the v2 Hub root for the
remaining epochs, e.g. `--base_model thesimonharms/trocr-javanese-synthetic-v2
--epochs 5` with the same data/length flags.

### Chunked HF Jobs (when ~2h kills hit)

HF may SIGTERM (~exit 143) around 1.5–2h. Use **1-epoch** chunks (~50–60 min)
and mix NusaAksara OCR. Skip final CER so the Hub push finishes before the cutoff.

```powershell
# Chunk 1 (from v1)
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN `
  training/trocr/finetune_trocr.py `
  --base_model thesimonharms/trocr-javanese-synthetic `
  --dataset_name thesimonharms/javanese-dataset-180k `
  --extra_dataset_name thesimonharms/javanese-nusaaksara-ocr `
  --extra_dataset_upsample 8 `
  --hub_model_id thesimonharms/trocr-javanese-synthetic-v2 `
  --epochs 1 --batch_size 24 --lr 2e-5 --warmup_ratio 0.05 `
  --no-gradient_checkpointing --eval_every_epochs 1 `
  --skip_final_cer --pdf_labeled_dir none

# Chunks 2–10 (resume v2, no warmup)
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN `
  training/trocr/finetune_trocr.py `
  --base_model thesimonharms/trocr-javanese-synthetic-v2 `
  --dataset_name thesimonharms/javanese-dataset-180k `
  --extra_dataset_name thesimonharms/javanese-nusaaksara-ocr `
  --extra_dataset_upsample 8 `
  --hub_model_id thesimonharms/trocr-javanese-synthetic-v2 `
  --epochs 1 --batch_size 24 --lr 2e-5 --warmup_ratio 0 `
  --no-gradient_checkpointing --eval_every_epochs 1 `
  --skip_final_cer --pdf_labeled_dir none
```

## After the job

1. Confirm `https://huggingface.co/thesimonharms/trocr-javanese-synthetic` has
   `config.json` / weights (not an empty page).
2. Job logs should show `[OK] Model pushed to HF Hub`.
3. You can close the laptop anytime after submit — billing stops when the job
   finishes or hits `--timeout`.
