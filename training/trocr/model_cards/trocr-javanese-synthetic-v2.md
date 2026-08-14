---
# SUPERSEDED — Hub repo deleted. Current model: thesimonharms/trocr-javanese-synthetic-v6
language:
- jv
license: mit
library_name: transformers
pipeline_tag: image-to-text
tags:
- trocr
- ocr
- javanese
- aksara
- handwritten
- vision-encoder-decoder
- generated_from_trainer
base_model:
- thesimonharms/trocr-javanese-synthetic
datasets:
- thesimonharms/javanese-dataset-180k
- thesimonharms/javanese-dataset
- thesimonharms/javanese-nusaaksara-ocr
model-index:
- name: trocr-javanese-synthetic-v2
  results:
  - task:
      type: image-to-text
      name: Optical Character Recognition
    dataset:
      type: thesimonharms/javanese-dataset
      name: Javanese synthetic lines (validation, 1500 samples)
      split: validation
    metrics:
    - type: cer
      value: 0.6671
      name: Character Error Rate (epoch-3 Hub snapshot)
    - type: exact_match
      value: 0.0033
      name: Exact match rate (epoch-3 Hub snapshot)
---

# TrOCR Javanese Aksara (v2)

Continue-finetune of [`thesimonharms/trocr-javanese-synthetic`](https://huggingface.co/thesimonharms/trocr-javanese-synthetic) (**v1**) for Javanese Aksara line OCR, with a broader train mix (180k synthetic + original replay + real NusaAksara lines) and a gentler learning rate.

> **Status:** training / iterative publish. Weights at repo root are overwritten each epoch. Prefer the checkpoint with best `eval_loss` (currently **epoch 2**) until a later epoch beats it **and** free-generation CER on the original-val gate improves over v1.

## Why v2 exists

An earlier continue-FT of v1 (hot LR `2e-5`, 180k+Nusa without original replay) **raised** original-val CER (~0.63 → ~0.75) and collapsed short-line accuracy. Domain A/B on 180k-val showed the same regression in-domain, so it was not a domain-shift artifact.

This v2 run restarts from **v1** with:

| Choice | Value | Why |
|--------|--------|-----|
| Base | `trocr-javanese-synthetic` (v1) | Keep the tokenizer-expanded basin |
| Data | 180k + original `×1` + Nusa `×8` | New unique lines + replay scoreboard domain + real OCR |
| LR | `1e-5` | Avoid walking off v1 |
| Schedule | 15 epochs max, early-stop patience 3 on `eval_loss` | Hands-off; stop when val stalls |
| Push | every epoch → this repo | Survive restarts; one Hub id to watch |

## Model description

Same architecture as v1: TrOCR encoder-decoder with an expanded Javanese tokenizer (vocab ≈ **50361**). Input is a single line image; output is Unicode Javanese text.

**Base model:** `thesimonharms/trocr-javanese-synthetic` (itself fine-tuned from `microsoft/trocr-base-handwritten`)

## Intended uses and limitations

**Use for**
- Line-level OCR experiments on synthetic / clean printed Aksara
- Comparing continue-FT recipes against the v1 scoreboard
- Bootstrapping tools that need a frequently updated Hub checkpoint

**Limitations**
- **Work in progress.** Free-generation CER on the original 1500-line gate is **not** yet better than v1 (see Evaluation)
- Exact match rate remains low; treat outputs as noisy hypotheses
- Real manuscripts still need more domain data than the current Nusa upsample provides
- Load from **repo root** (no `subfolder=`)

## How to use

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

repo = "thesimonharms/trocr-javanese-synthetic-v2"
processor = TrOCRProcessor.from_pretrained(repo)
model = VisionEncoderDecoderModel.from_pretrained(repo)
model.eval()

image = Image.open("line.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

with torch.no_grad():
    ids = model.generate(pixel_values, max_new_tokens=64, num_beams=4)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

## Training data

Per-epoch train size ≈ **246,072** lines after mixing:

| Dataset | Role | Upsample |
|---------|------|----------|
| [`javanese-dataset-180k`](https://huggingface.co/datasets/thesimonharms/javanese-dataset-180k) | Primary synthetic mix | 1× (180k train / 3k val) |
| [`javanese-dataset`](https://huggingface.co/datasets/thesimonharms/javanese-dataset) | Replay v1 scoreboard domain | 1× (~60k) |
| [`javanese-nusaaksara-ocr`](https://huggingface.co/datasets/thesimonharms/javanese-nusaaksara-ocr) | Real OCR lines | 8× (~759 → ~6k) |

Validation mix ≈ **5050** lines (loss-only mid-train eval).

## Training procedure

Hands-off Docker train on an UGOS Pro NAS using the Intel XPU stack (`intel/pytorch` image), batch size **24**, no gradient checkpointing, on-the-fly image encode.

### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Base checkpoint | `thesimonharms/trocr-javanese-synthetic` |
| Learning rate | `1e-5` |
| Train batch size | 24 |
| Max epochs | 15 |
| Warmup ratio | 0.05 |
| Early stopping | patience **3** on `eval_loss` |
| Eval cadence | every epoch (loss-only) |
| Hub strategy | `every_save` (overwrite this repo) |

### Mid-train validation loss

| Epoch | eval_loss |
|------:|----------:|
| 1 | 0.1127 |
| 2 | **0.1035** |
| 3 | 0.1052 |
| 4 | 0.1043 |

Best by `eval_loss` so far: **epoch 2**. Later epochs may still be selected only if they beat this and CER on the original-val gate improves.

## Evaluation

Free-generation CER on the original private validation gate (greedy decode, 1500 lines), measured on an epoch-3 Hub snapshot:

| Metric | v1 | v2 (ep3 snapshot) |
|--------|---:|------------------:|
| Mean CER | **0.6266** | 0.6671 |
| Exact match | 0.13% | **0.33%** |

> Lower CER is better. Exact match improved, but mean CER did **not** beat v1 on this gate yet. Prefer v1 for scoreboard CER until a later v2 epoch clears that bar.

In-domain `eval_loss` (train mix val) is tracked each epoch in Training procedure above.

## Hardware note

This run used the NAS **iGPU** (`Intel Arc Graphics` via `/dev/dri/renderD128`). The discrete Arc Pro B60 enumerates on PCI (`8086:e211`) but did not bind to `xe` under UGOS Pro at bring-up (VF BAR / enable failures). Docker recipe for Linux XPU lives in the project `training/trocr/nas/` folder.

## Framework versions

- Transformers 4.57.6
- PyTorch 2.11.0+xpu (train image)
- Datasets 5.0.0
- Tokenizers 0.22.2

## Citation

```bibtex
@misc{trocr-javanese-synthetic-v2,
  author = {Harms, Simon},
  title = {TrOCR Javanese Aksara (synthetic v2 continue-FT)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v2}}
}
```
