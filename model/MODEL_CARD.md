# Javanese OCR Model Card (`javanese_ocr.onnx`)

This is the **CRNN + CTC** model shipped with the TypeScript package. For the
current research **TrOCR** line-OCR model (96% exact on short synthetic print),
see [`thesimonharms/trocr-javanese-synthetic-v6`](https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6).

## Overview

- **Architecture**: CRNN (6-layer Convolutional Neural Network backbone + Sinusoidal Positional Encoding + 2-layer Transformer Encoder + CTC Linear Projection).
- **Input**: `Float32Array[1, 1, 32, 128]` — single-channel greyscale image strip, resized to $128 \times 32$ pixels, normalized to $[0, 1]$ (`arr / 255.0`).
- **Output**: `Float32Array[1, T, 51]` — raw logits across $T$ timesteps for 51 CTC classes.
- **Decoding**: Supports greedy CTC decoding and language-model assisted shallow fusion beam search (`CharNgramLM`).

---

## Character Alphabet (`ALPHABET`)

The model outputs 51 classes (`NUM_CLASSES = 51`):
- **Index 0**: `<blank>` (CTC blank token)
- **Indices 1–50**: Unicode Javanese script codepoints `U+A98F` to `U+A9C0` inclusive (`range(0xA98F, 0xA9C1)`).

### Supported Codepoint Range (`0xA98F`–`0xA9C0`)

| Range | Characters | Description |
| :--- | :--- | :--- |
| `0xA98F`–`0xA9A3` | `ꦏ ꦐ ꦑ ꦒ ... ꦣ` | Basic & Murda Consonants |
| `0xA9A4` | `ꦤ` ... `ꦲ` | Consonants through HA |
| `0xA9A5`–`0xA9BD` | `ꦸ ꦹ ꦺ ꦻ ꦼ ꦽ` ... | Sandhangan (Vowel diacritics & anuswara/cecak/wignyan/layar) |
| `0xA9BE` | `ꦾ` | Pengkal (medial `-y-`) |
| `0xA9BF` | `ꦿ` | Cakra (medial `-r-`) |
| `0xA9C0` | `꧀` | Pangkon (virama / vowel killer) |

---

## Known Limitations & Alphabet Audit (Phase 1.4)

### 1. Pengkal (`ꦾ`, `U+A9BE`) Under-Representation
While codepoint `0xA9BE` (`ꦾ`) is explicitly present in the 50-class vocabulary (`ALPHABET[48]`), audit of `training/javanese_aksara.txt` shows **0 occurrences** of `ꦾ`.

**Root cause**: The synthetic corpus generator (`transliterate_corpus.js`) converts Latin text using `Aksara.getAksara()`, which currently serializes medial `-y-` clusters as `꧀ꦪ` (`pangkon + ya`) rather than dedicated pengkal (`ꦾ`).

**Impact**: Because the CRNN seen no pengkal examples during synthetic pre-training (`generate_from_corpus`), it cannot reliably decode `ꦾ` glyphs in real manuscript scans and will typically decode them as `꧀ꦪ` or drop the medial subscript.

**Mitigation (Phase 2)**:
- Update `Aksara.toAksara()` to emit `ꦾ` for medial `-y-` sequences.
- Retrain the CRNN after regenerating synthetic strips (`--mode generate_from_corpus`) with pengkal sequences included.

### 2. Standalone Vowels (`ꦄ` `ꦇ` `ꦉ` `ꦌ` `ꦎ`)
Codepoints outside `0xA98F`–`0xA9C0` (such as independent vowels `0xA984`–`0xA98E` or punctuation `0xA9C1`–`0xA9CF`) are outside the model's 50-class alphabet and are not decoded by this OCR layer. Standalone vowels in standard Javanese orthography are represented via `ꦲ` (`HA`) plus vowel diacritic.
