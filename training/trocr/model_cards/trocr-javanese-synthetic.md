---
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
- microsoft/trocr-base-handwritten
datasets:
- thesimonharms/javanese-dataset
widget:
- src: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ocr.png
  example_title: OCR example
model-index:
- name: trocr-javanese-synthetic
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
      value: 0.6266
      name: Character Error Rate
    - type: exact_match
      value: 0.0013
      name: Exact match rate
---

# TrOCR Javanese Aksara (v1)

Fine-tuned [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) for **printed/synthetic Javanese Aksara (Hanacaraka) line OCR**.

This is the **v1** scoreboard model: expanded Javanese tokenizer, trained on cleaned synthetic lines, and the strongest free-generation CER we measured on the original validation gate before the v2 retrain.

## Model description

Javanese Aksara is a Brahmic abugida: consonants carry a default vowel, and stacked diacritics change or mute it. Standard TrOCR tokenizers fragment those codepoints, which collapses free-run CER. This checkpoint expands the decoder tokenizer with atomic Javanese Unicode characters (vocab size **50361**) and fine-tunes the full encoder-decoder on synthetic Aksara line images.

**Base model:** `microsoft/trocr-base-handwritten`  
**Output:** Unicode Javanese text (one line image → one string)  
**Intended use:** research / prototype OCR for synthetic or clean printed Aksara lines

## Intended uses and limitations

**Use for**
- Line-level OCR on synthetic or relatively clean printed Javanese Aksara
- Bootstrapping labeling tools and further fine-tunes

**Limitations**
- Not a production manuscript reader: real handwriting, palm-leaf noise, stains, and layout still hurt badly without domain adaptation
- Exact line match rate is low (~0.1% on the 1500-line original val gate); many predictions are partially correct
- Trained mainly on synthetic renderings; real OCR lines need extra fine-tuning (see v2)
- Load weights from the **repo root**, not the stale `final/` subfolder

## How to use

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

repo = "thesimonharms/trocr-javanese-synthetic"
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

Private Hub dataset [`thesimonharms/javanese-dataset`](https://huggingface.co/datasets/thesimonharms/javanese-dataset):

- ~**60k** cleaned synthetic train lines / **2k** validation
- Rendered Aksara lines; corpus cleaned (max length cap, wiki/HTML stripped)
- Job runs typically capped at **50k** train samples for wall-clock

## Training procedure

Fine-tuned from `microsoft/trocr-base-handwritten` on Hugging Face Jobs (`a10g-large`) with on-the-fly image encoding (no pre-cached `pixel_values`).

### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | `4e-5` |
| Train batch size | 24 |
| Eval batch size | 16 |
| Epochs | 20 |
| LR scheduler | linear |
| Warmup ratio | 0.05 |
| Optimizer | AdamW (fused) |
| Seed | 42 |
| Mixed precision | native AMP |
| Tokenizer expansion | Javanese block as atomic tokens (vocab ≈ 50361) |

### Training loss curve

| Training Loss | Epoch | Step  | Validation Loss |
|:-------------:|:-----:|:-----:|:---------------:|
| 0.0810 | 2.0  | 4168  | 0.0992 |
| 0.0441 | 4.0  | 8336  | 0.0638 |
| 0.0363 | 6.0  | 12504 | 0.0497 |
| 0.0277 | 8.0  | 16672 | **0.0483** |
| 0.0187 | 10.0 | 20840 | 0.0512 |
| 0.0114 | 12.0 | 25008 | 0.0536 |
| 0.0042 | 14.0 | 29176 | 0.0574 |
| 0.0020 | 16.0 | 33344 | 0.0646 |
| 0.0006 | 18.0 | 37512 | 0.0711 |
| 0.0009 | 20.0 | 41680 | 0.0732 |

Best checkpoint by validation loss: **epoch 8** (val loss **0.0483**). Later epochs overfit train loss while val loss rose.

## Evaluation

Free-generation CER on the original private validation gate (greedy decode, 1500 lines):

| Metric | Value |
|--------|------:|
| Mean CER | **0.6266** |
| Exact match | 0.13% |
| CER p50 | 0.5833 |
| CER p90 | 0.7857 |
| Short lines (≤8 chars) CER | 0.5306 |
| Long lines (>8 chars) CER | 0.6374 |

In-domain check on `javanese-dataset-180k` validation (1500 lines): mean CER **~0.611**.

> Load from the **repository root**. The `final/` subfolder is a stale export (smaller vocab) and should not be used.

## Framework versions

- Transformers 4.57.6
- PyTorch 2.13.0+cu130
- Datasets 5.0.0
- Tokenizers 0.22.2

## Citation

```bibtex
@misc{trocr-javanese-synthetic,
  author = {Harms, Simon},
  title = {TrOCR Javanese Aksara (synthetic v1)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/thesimonharms/trocr-javanese-synthetic}}
}
```
