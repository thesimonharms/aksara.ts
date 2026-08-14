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
- hanacaraka
- printed
- vision-encoder-decoder
base_model: microsoft/trocr-small-printed
base_model_relation: finetune
datasets:
- thesimonharms/javanese-synthetic-exact
widget:
- src: images/example.png
  example_title: Short printed Aksara line (384×384)
model-index:
- name: trocr-javanese-synthetic-v6
  results:
  - task:
      type: image-to-text
      name: Optical Character Recognition
    dataset:
      type: thesimonharms/javanese-synthetic-exact
      name: Javanese synthetic-exact (held-out validation, 1500 lines)
      split: validation
    metrics:
    - type: exact_match
      value: 0.96
      name: Exact match
    - type: cer
      value: 0.006834
      name: Character Error Rate
    - type: near_match
      value: 0.999333
      name: Near match (Levenshtein ≤ 2)
---

# TrOCR Javanese Aksara (v6)

Line-level OCR for **short, clean, printed Javanese Aksara (Hanacaraka)**. Fine-tuned from [`microsoft/trocr-small-printed`](https://huggingface.co/microsoft/trocr-small-printed) on a private synthetic set of 384×384 line images with at most 12 aksara per label.

| Gate (1500 held-out lines, greedy decode) | Value |
|---|---|
| **Exact match** | **96.0%** (1440 / 1500) |
| Character error rate | **0.68%** |
| Near match (edit distance ≤ 2) | **99.93%** (1499 / 1500) |

This is a **working recognizer for the synthetic domain it was trained on**. It is not a manuscript reader, not a page OCR system, and not a substitute for a text detector. Earlier public checkpoints (v1–v4) were withdrawn; this is the current model.

## Model description

TrOCR is an encoder–decoder Transformer: a Vision Transformer reads a single image, and an autoregressive decoder emits text ([Li et al., 2023](https://arxiv.org/abs/2109.10282)). This checkpoint keeps the **small printed** stack:

| Part | Detail |
|---|---|
| Encoder | DeiT-small, 12 layers, hidden size 384, 16×16 patches, **384×384** input |
| Decoder | TrOCR decoder, 6 layers, `d_model` 256 |
| Tokenizer | Base SentencePiece + **96 atomic Javanese characters** (vocab **64,098**) |
| Input | One RGB line image (pad to square, then the processor resizes to 384×384) |
| Output | One Unicode Javanese string |

Javanese is an abugida: a base consonant carries an inherent vowel, modified by *sandhangan* (marks above, below, left, or right of the body) and *pasangan* (subscript consonants). Byte-level BPE fragments those codepoints. This model adds each Javanese character as a single token so free-run generation can emit valid orthography.

## Intended use

**Use for**
- Research and tooling on **cropped, printed, short** Aksara lines that look like the training renderer (clean type, cream or white paper, ≤12 characters)
- Bootstrapping a larger pipeline **after** an external line detector has produced crops
- Fine-tuning further toward a narrower printed domain (same square-pad + unshifted-loss recipe)

**Do not use for**
- Full manuscript pages, PDFs, or photos of a book opening
- Palm-leaf / lontar / stained / faded handwriting
- Long lines, multi-column layout, or scene text
- Production archival digitization without a human in the loop

## How to use

Pad every crop to a **square** before `TrOCRProcessor`. TrOCR always resizes to 384×384; stretching a wide, short strip ~6× vertically wrecks sandhangan geometry.

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch


def pad_to_square(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    fill = img.getpixel((0, 0))
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img, (0, (side - h) // 2))
    return canvas


repo = "thesimonharms/trocr-javanese-synthetic-v6"
processor = TrOCRProcessor.from_pretrained(repo)
model = VisionEncoderDecoderModel.from_pretrained(repo)
model.eval()

image = pad_to_square(Image.open("line.png"))
pixel_values = processor(images=image, return_tensors="pt").pixel_values

cls_id = processor.tokenizer.cls_token_id
eos_id = processor.tokenizer.sep_token_id or processor.tokenizer.eos_token_id

with torch.no_grad():
    ids = model.generate(
        pixel_values,
        max_new_tokens=24,
        num_beams=4,
        do_sample=False,
        decoder_start_token_id=cls_id,
        eos_token_id=eos_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        no_repeat_ngram_size=0,
    )
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

Load weights from the **repository root** (not a `final/` subfolder). Official scores below used **greedy** decoding (`num_beams=1`) plus a sandhangan anti-loop logits processor; beam search is shown here as a reasonable default for interactive use.

A line detector is required for pages. Hugging Face maintainers document TrOCR as a **text-line** model and recommend pairing it with a detector such as CRAFT or FAST ([transformers#37639](https://github.com/huggingface/transformers/issues/37639)). Do **not** tile a page into a 384×384 grid: that cuts through glyphs.

## Training data

Private Hub dataset [`thesimonharms/javanese-synthetic-exact`](https://huggingface.co/datasets/thesimonharms/javanese-synthetic-exact) (not redistributed).

| Split | Size |
|---|---|
| Train | 60,000 lines |
| Validation | 2,500 lines (scores use a 1,500-line prefix) |

**Construction**
- Labels are Aksara-only Unicode, length **2–12**, held out by a hash of source lines
- Rendered at **384×384** on white / cream paper with light noise, blur, contrast, and JPEG
- No manuscript backgrounds, no Latin, no English chart labels
- Includes a glyph primer (nglegena, murda, swara, sandhangan, pasangan, digits, common words) mixed with corpus chunks

## Training procedure

Curriculum on an Intel Arc XPU (`torch.xpu`), **fp32**, eager attention, batch size 16, no gradient checkpointing. The encoder was **not** frozen (a freeze collapsed this DeiT-small stack; see Findings).

| Stage | Data | Epochs | LR | Notes |
|---|---|---|---|---|
| Phase 0 | 32-line overfit | 400 | 1e-4 | Stack gate; **32/32 exact**, weights discarded |
| Phase A | 60k train | 3 | 5e-5 | Tokenizer expanded; unfrozen |
| Phase B | 60k train, from A final | 12 | 2e-5 | No further tokenizer expand |

Other knobs: `max_target_length=24`, `max_label_chars=12`, AdamW, linear schedule, warmup ratio 0.05, seed 42.

Phase B teacher-forced validation loss (Trainer):

| Epoch | Train loss | Val loss |
|------:|----------:|---------:|
| 1 | 0.0377 | 0.0647 |
| 2 | 0.0244 | 0.0636 |
| 3 | 0.0168 | 0.0497 |
| 4 | 0.0137 | 0.0529 |
| 5 | 0.0137 | 0.0466 |
| 6 | 0.0114 | 0.0406 |
| 7 | 0.0074 | 0.0409 |
| 8 | 0.0043 | 0.0433 |
| 9 | 0.0010 | 0.0404 |
| 10 | 0.0001 | 0.0365 |
| 11 | 0.0001 | 0.0371 |
| 12 | 0.0000 | 0.0368 |

The jump from Phase A val loss (~0.035) to Phase B epoch 1 (0.065) is a **fresh Adam + higher LR after A had decayed**, not a logging artifact. Free-generation exact match dipped 93.9% → 88.1% then recovered to 96.0%.

### Loss implementation note

Training used a patched `VisionEncoderDecoder` loss: plain token-wise cross-entropy **without** a second label shift. In `transformers` 4.46+, `ForCausalLMLoss` shifts labels after VED has already called `shift_tokens_right`, so the model otherwise learns to skip the first aksara. Teacher-forced loss still looks healthy in that bug; free-run never matches. Anyone continuing this fine-tune on current Transformers should keep that patch.

## Evaluation

Protocol: 1,500 validation lines from `javanese-synthetic-exact`, pad-to-square, greedy decode, `decoder_start_token_id = cls`, `no_repeat_ngram_size = 0`, sandhangan anti-loop processor at inference. Exact match is Unicode-string equality. CER is mean Levenshtein / reference length.

| Checkpoint | Exact | CER | Near (≤ 2 edits) |
|---|---:|---:|---:|
| Phase A epoch 1 | 55.3% | 12.01% | 90.1% |
| Phase A epoch 2 | 85.3% | 2.63% | 98.7% |
| Phase A final | 93.9% | 1.13% | 99.7% |
| Phase B epoch 1 | 88.1% | 2.14% | 99.5% |
| Phase B final | **96.0%** | **0.68%** | **99.93%** |

Residual errors are almost all near-misses (59 lines with edit distance 1–2; one line worse than that).

## Findings

This section is the engineering record of v1–v6, plus what published OCR work already implied we would hit.

### 1. TrOCR is a line recognizer with a square ViT, not a page model

[Li et al. (2023)](https://arxiv.org/abs/2109.10282) resize every input to **384×384** and split it into 16×16 patches. The decoder then emits **one** token sequence. Hugging Face’s TrOCR maintainers state the same operational limit: crop a single text line first; a full sentence or page as one image is the wrong input ([transformers#37639](https://github.com/huggingface/transformers/issues/37639)).

Consequence: this checkpoint cannot OCR a manuscript page. A 384×384 grid over a page will bisect sandhangan and *pasangan*. The correct outer loop is **detect lines → pad each crop to square → recognize**.

### 2. Aspect-ratio distortion is an encoder geometry problem

A 64-pixel-tall, wide strip stretched to 384×384 is scaled ~6× on the vertical axis. Aksara vowels and virama live in that vertical band. Pad-to-square (left-aligned, vertically centered, paper-colored fill) makes the ViT resize isotropic. Training images for v6 are **native 384×384**, so train and inference geometry match.

This is the same class of failure users see when feeding arbitrary-aspect scans to a ViT OCR model. Capacity (`trocr-large`) does not fix a warped encoder canvas.

### 3. Exact match is a different objective from CER

On a ~30-character line, 10% CER is about **four** independent errors, which is ~4% exact match if errors are spread. v1–v5 reported CER on long, noisy, wiki-derived lines (often 40–48 characters) and never had a path to 90% exact. v6 shortens the label to ≤12 aksara so exact match is a meaningful gate, then actually optimizes it.

### 4. Javanese script is a hard OCR domain even in print

Independent printed-Javanese work still struggles at the **segmentation** step that classical OCR assumes. Widiarti et al. (2026) report ~63.5% segmentation accuracy and note that nglegena + *pasangan* + *sandhangan* yield **more than 11,000** visual syllable types. Mahastama and Krisnawati (2019) show why: marks sit above, below, left, and right of the body, so a naive projection profile splits a line into false “diacritic rows” that must be merged back.

Sequence generation (TrOCR) avoids explicit character boxes, which is why v6 can look strong **once the line is already cropped and the canvas is square**. It does not remove the line-detection problem, and it does not create robustness to handwriting, stains, or unseen fonts.

### 5. Model size was the wrong lever

v4/v5 fine-tuned `microsoft/trocr-large-printed` on mixed or “HQ” data. Decoder start-token, runaway sandhangan loops, stretched lines, and long labels dominated the error. After those were fixed, **`trocr-small-printed` reached 96% exact** on the short synthetic set. The task is a few dozen new glyphs on clean paper; DeiT-small is enough and allowed a full 3+12 epoch cook on one NAS run.

### 6. Bugs that looked like “the model cannot learn Aksara”

These blocked Phase 0 until they were removed. They are reproducible footguns for anyone fine-tuning TrOCR on recent Transformers / Intel XPU.

| Failure | What we saw | Fix |
|---|---|---|
| Frozen DeiT-small encoder | Collapse to pangkon / wulu; 0/64 exact | Train encoder + decoder |
| `ForCausalLMLoss` double-shift (`transformers` 4.46+) | Teacher-forced loss OK; free-gen always dropped character 0 | Unshifted CE on VED logits |
| bf16 on Linux XPU | Train loss plateau ~2.6; no memorization | `fp32` + eager attention |
| Square-canvas length heuristic | `max_new_tokens` truncated free-gen | Use the hard cap on near-square images |
| Height-64 wide renders / manuscript-crop HQ | Sandhangan destroyed before the decoder | Native 384×384, no heavy degrade |
| BPE-fragmented Javanese | Invalid mark runs | +96 atomic tokens; `decoder_start=cls` |

Phase 0 after the loss patch: **32/32 exact** on the overfit set. That gate is why v6 was allowed to cook.

### 7. In-domain success is not out-of-domain OCR

A Wikimedia specimen ([Sample Hanacaraka font](https://commons.wikimedia.org/wiki/File:Sample_Hanacaraka_font.png), UDHR line) is Javanese print in a **different typeface, much longer than 12 aksara, transparent PNG**. The model emitted Javanese-looking Unicode that was **wrong** from the first characters. That is the expected domain gap: new font, new length, new canvas, no line crop.

v1–v2 mixing real NusaAksara lines into a broken geometry/loss stack did not buy manuscript skill. Domain transfer needs correct line crops, matching image geometry, and labels in the same script convention — then more real ink.

## Limitations

- **Domain.** Held-out numbers are the same renderer, font pool, paper, and length cap as training. They are not a handwriting or archival benchmark.
- **Length.** Labels longer than 12 aksara were not trained. `generation_config` `max_length` is 24.
- **Pages and layout.** No detector, no reading-order, no region proposal. One image → one short string.
- **Script coverage.** Inventory includes nglegena, murda, swara, sandhangan, pasangan, digits, and pada used in the primer/corpus, but rare conjuncts and unseen font idiosyncrasies remain open.
- **Tokenizer.** The 96 new embeddings were randomly initialized and then trained; they have no MiniLM pre-training signal of their own.
- **Inference contract.** Skip pad-to-square, or start the decoder on the wrong BOS, and quality collapses even on in-domain images.
- **Privacy.** Training images are private synthetic renders. This card does not release that dataset.
- **Not in the npm runtime.** The `aksara-ts` package still ships a separate CRNN ONNX for TypeScript. This Hub model is the research TrOCR line.

## Bias

The renderer uses a small set of desktop Javanese fonts and cream/white paper. It will systematically prefer those stroke weights and proportions. Historical hands, regional glyph variants, and degraded supports are out of distribution. Labels contain no Latin; feeding Latin or bilingual charts will not yield useful text.

## Hardware and software

- Intel Arc via `torch.xpu` (NAS iGPU), fp32, eager attention
- Transformers 4.57.6 · PyTorch 2.11.0+xpu · Datasets 5.0.1 · Tokenizers 0.22.2

Training code: [`training/trocr`](https://github.com/thesimonharms/aksara.ts/tree/main/training/trocr) in [aksara.ts](https://github.com/thesimonharms/aksara.ts).

## License

MIT. Base weights [`microsoft/trocr-small-printed`](https://huggingface.co/microsoft/trocr-small-printed) are also MIT.

## Citation

```bibtex
@misc{harms2026trocr-javanese-v6,
  author = {Harms, Simon},
  title = {TrOCR Javanese Aksara (synthetic v6)},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6}}
}

@article{Li2023TrOCR,
  title = {{TrOCR}: Transformer-based Optical Character Recognition with Pre-trained Models},
  author = {Li, Minghao and Lv, Tengchao and Chen, Jingye and Cui, Lei and Lu, Yijuan and Florencio, Dinei and Zhang, Cha and Li, Zhoujun and Wei, Furu},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {37},
  number = {11},
  pages = {13094--13102},
  year = {2023},
  doi = {10.1609/aaai.v37i11.26538}
}
```

Related work cited in Findings:

- Widiarti, A. R., Adji, S. E. P., Adji, F. T., Indraputra, G. K. B., Trisnanto, Y. A., & Pratama, F. B. Y. (2026). A baseline evaluation of OCR segmentation and classification methods for printed Javanese script. *Engineering, Technology & Applied Science Research, 16*(1), 31699–31705. https://doi.org/10.48084/etasr.15502
- Mahastama, A. W., & Krisnawati, L. D. (2019). Improving projection profile for segmenting characters from Javanese manuscripts. *Proceedings of the International Conference on Computer Vision and Graphics*.
