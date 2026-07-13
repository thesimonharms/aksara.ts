# ROADMAP

This document is the canonical, detailed plan for closing the gaps in `aksara.ts`. It supersedes the brief list in the README's `## Roadmap` section. Each phase lists concrete deliverables, an acceptance test, and the artifacts it touches. Items reference specific files and line ranges so the work is unambiguous.

The single most important constraint: **the pipeline in the README cannot run end-to-end today.** The TypeScript runtime has no OCR leg. Every other item is subordinate to closing that gap.

---

## Status: shipped

Items already merged on `main` (commit `e7031f3`):

- Bidirectional Latin ↔ Aksara transliteration (`src/aksara.ts`)
- Murda consonant support (uppercase Latin → prestige forms)
- Neural word segmenter with ONNX export (`src/segmenter.ts`, `model/segmenter.onnx`, `model/vocab.json`)
- 90-test suite covering the transliteration layer (`tests/aksara.test.ts`)
- Build + npm packaging with `exports` map (`package.json`)

---

## Phase 1 — Close the OCR leg of the pipeline (1–2 weeks)

Without this, the diagram in the README is aspirational. The ONNX model exists in `training/javanese_ocr.onnx` but nothing in `src/` can call it.

### 1.1 TypeScript OCR runtime

Build a `src/ocr.ts` module that mirrors the structure of `src/segmenter.ts`:

- `OcrModel.load(modelPath?, vocabPath?)` — returns a Promise.
- `model.recognize(image: ImageData | Buffer, options?)` — runs the pre-processing pipeline (greyscale → resize to 128×32 → tile if wider → CTC greedy decode by default) and returns `{ text: string, confidence: number, tiles: { text, confidence }[] }`.
- Add an entry in `package.json#exports` for `"./ocr"`.
- Expose the same `IMAGENET`-style normalisation the training script uses (`arr / 255.0`, single channel, shape `[1, 1, 32, 128]`).
- Accept a `CharNgramLM`-equivalent for shallow fusion, or stub it for v1 (LM in TS = port of `CharNgramLM` from `training/crnn/javanese_ocr.py:171–223`; small class, ~50 LOC).

**Acceptance:** `bun run scripts/ocr-demo.ts path/to/manuscript_page.png` prints line-by-line decoded Aksara. The output is non-empty and matches the model output from `python training/crnn/javanese_ocr.py --mode predict` to ≥ 90% of characters on a held-out test image.

**Touches:** `src/ocr.ts` (new), `package.json`, `scripts/ocr-demo.ts` (new).

### 1.2 End-to-end pipeline demo

Extend `scripts/demo.ts` (or add `scripts/ocr-llm-demo.ts`) to take a PDF/PNG path and emit:

```
[1] rendered page → OCR tiles → Aksara strings
[2] Aksara strings → fromAksara → Latin strings
[3] Latin strings → Segmenter → segmented Javanese
[4] segmented Javanese → "would be sent to an LLM"
```

**Acceptance:** demo runs on `training/PDFA.pdf` (already in the repo) and the final segmented output is human-readable Indonesian/Javanese.

**Touches:** `scripts/demo.ts`, `scripts/ocr-llm-demo.ts` (new).

### 1.3 Fix `training/setup.bat`

The setup script is incomplete: it installs `torch`, `onnx`, `onnxruntime`, `numpy` but `javanese_ocr.py:53` imports `fitz` (pymupdf) and `training/ocr_dataset.py:25` uses PIL. Without those, `--mode train_lm` and `--mode generate_from_corpus` fail on a clean machine.

- Add `pip install pymupdf pillow torchvision` to `setup.bat` after the `onnx onnxruntime numpy` line.
- Same for `training.sh` if it has the same gap.
- Add a post-install check that prints `python -c "import fitz, PIL, torch, onnxruntime"` and exits non-zero on failure.

**Acceptance:** on a clean venv, `setup.bat && python training/crnn/javanese_ocr.py --mode train_lm --corpus training/jv_plain.txt --output_path training/javanese_lm.pkl` succeeds end to end.

**Touches:** `training/setup.bat`, `training.sh` if applicable.

### 1.4 Verify and document the OCR alphabet

`training/crnn/javanese_ocr.py:85` defines:

```python
JAVANESE_CHARS = [chr(i) for i in range(0xA98F, 0xA9C1)]
```

The range `0xA98F`–`0xA9C0` is 50 codepoints. Pangkon is `0xA9C0` (the last included), and the pengkal subscript `ꦾ` = `0xA9BE` is in range, so subscript-y clusters are technically decodable — but the trained model has 50 output classes and may not have seen enough pengkal examples to recognise them reliably. Confirm by:

- Counting pengkal occurrences in `training/javanese_aksara.txt` (139 MB corpus).
- Running `predict` on synthetic pengkal-heavy strings and checking character accuracy.

If accuracy is low, either (a) regenerate synthetic data with more pengkal, or (b) document the limitation explicitly in the model card.

**Acceptance:** a written note in the repo (could be in `ROADMAP.md` or a new `model/MODEL_CARD.md`) describing what the OCR can and cannot decode.

**Touches:** `training/crnn/javanese_ocr.py` (regen threshold or filter), `model/MODEL_CARD.md` (new).

---

## Phase 2 — Improve OCR accuracy (2–4 weeks)

> **Update:** Items 2.1, 2.2, and 2.4 are addressed by the new TrOCR fine-tune
> pipeline in `training/trocr/` — see [`training/trocr/README.md`](./training/trocr/README.md).
> That pipeline replaces the two-font / single-PDF synthetic approach with a
> multi-font + multi-PDF generator, adds a HITL labeler for real handwriting
> (`label_pdfs.py`), and publishes the fine-tuned model to HF Hub from a
> Docker-equipped HF Space. The CRNN scripts remain in `training/crnn/` for
> reference. The items below are kept for historical context.

The current model is trained on synthetic data rendered with two fonts. Real manuscripts have ink bleed, parchment texture, ligature variance, and scribal idiosyncrasies. Self-training on real material closes most of the gap.

### 2.1 Document the existing training pipeline

The 9-mode CLI workflow (`javanese_ocr.py:1119–1158`) lives only in `--help` output. New users have no entry point.

- Add a `training/README.md` covering the 6-step self-training loop: `generate_from_corpus` → `train_lm` → `train` → `ingest` → `pseudo_label` → `train` (retrain with expanded data).
- Include expected disk usage, time budgets, and a "minimum viable" command sequence that produces a working model in under an hour on CPU.

**Acceptance:** `training/README.md` exists and is linked from the top-level README.

**Touches:** `training/README.md` (new), `README.md` (one-line link).

### 2.2 Broaden the synthetic training distribution

The `generate_from_corpus` mode samples backgrounds from a single PDF (`training/PDFA.pdf`) and renders with two fonts. Augment with:

- Multiple manuscript background PDFs (collect 3–5 from Perpustakaan Nasional or similar public-domain scans).
- Vary font size, line spacing, and ink colour in `generate_from_corpus` (`javanese_ocr.py:700–859`).
- Increase the default `--num_samples` from 5,000 to 20,000+.

**Acceptance:** retrained model improves character accuracy on a held-out manuscript page by ≥ 5% over the current `javanese_ocr.onnx`.

**Touches:** `training/crnn/javanese_ocr.py`, `training/data/` (new, holds extra PDFs).

### 2.3 Wire the language model into the TS runtime

Port `CharNgramLM` (`crnn/javanese_ocr.py:171–223`) to TypeScript. The class is small: a `Map<string, Map<string, number>>` of context → char → count, plus a Laplace smoothing term. Provide it as `src/lm.ts` and accept it via `OcrModel.recognize({ lm })` to enable shallow-fusion beam search.

**Acceptance:** with LM weight 0.5, OCR output on a degraded manuscript sample is ≥ 10% more accurate (edit distance) than greedy decoding.

**Touches:** `src/lm.ts` (new), `src/ocr.ts` (extend `recognize` signature).

### 2.4 Resolve the "two ONNX homes" problem

`model/javanese_ocr.onnx` and `training/javanese_ocr.onnx` both appear in the working tree (the .gitignore allows them in `model/` and ignores them in `training/`). The committed copy should be the one the TS runtime loads.

- Decide on a single canonical path (`model/javanese_ocr.onnx` is consistent with the segmenter).
- Update the training script's `--output_path` default to write there, and `.gitignore` any other copies.
- Update `export.py` (segmenter export) and the new `export_onnx` mode to write to the same path.

**Acceptance:** `ls model/` shows exactly `segmenter.onnx`, `javanese_ocr.onnx`, `vocab.json`, plus external data files. No ONNX files anywhere in `training/`.

**Touches:** `training/crnn/javanese_ocr.py`, `training/export.py`, `.gitignore`, `model/`.

---

## Phase 3 — Improve transliteration quality (1–2 months)

These are the README's existing roadmap items, kept verbatim but expanded with concrete steps.

### 3.1 Structured token output

Currently `new Aksara(text).getAksara()` returns a flat string. Expose a `getTokens()` (or `tokenize()`) method that returns:

```ts
type AksaraToken =
  | { type: 'base'; char: string; latin: string; cluster?: string[] }
  | { type: 'sandhangan'; char: string; latin: string }
  | { type: 'pangkon'; char: string }
  | { type: 'pasangan'; char: string; base: string }
  | { type: 'pengkal'; char: string }
  | { type: 'vocalic'; char: string; latin: string }
  | { type: 'digit'; char: string; value: number }
  | { type: 'punctuation'; char: string; name: string }
  | { type: 'space' }
  | { type: 'syllable_break' };
```

This makes the output consumable by RAG pipelines and embedding models without lossy re-parsing.

**Acceptance:** `getTokens()` is implemented, tested, and the README's "Why this exists" section can show a downstream consumer using the structured form.

**Touches:** `src/aksara.ts` (refactor), `tests/aksara.test.ts` (new test file `tests/tokens.test.ts`).

### 3.2 Unicode normalisation

OCR engines emit inconsistent codepoint sequences for the same visual glyph (e.g. precomposed vs decomposed NFD, ZWJ vs single codepoint for some conjuncts). Add a `normalizeAksara(input: string): string` static that:

- Collapses NFD/NFC variants.
- Resolves common OCR confusables (e.g. `ꦲ` vs `ꦃ` vs `ꦄ`).
- Is idempotent: `normalize(normalize(x)) === normalize(x)`.

**Acceptance:** a corpus of 100 OCR outputs from `pdf2image` + a real OCR engine round-trips through `normalize → fromAksara` with no character drift.

**Touches:** `src/aksara.ts`, `tests/aksara.test.ts`.

### 3.3 Retrain the segmenter on broader data

Current model is trained on `data/jv.txt` (15,309 lines of MediaWiki localisation strings). Manuscript text uses a different vocabulary (`lamun`, `yén`, `hutama`, etc.) and the model's space-density calibration is off for it.

- Source a prose corpus: Javanese Wikipedia dumps, gutenberg.org Javanese texts, or a curated prose set from a university corpus.
- Retrain with the same architecture (`training/train.py`), evaluate on a held-out prose set.
- Ship a `model/segmenter.prose.onnx` variant if the new model regresses on UI strings, and let `Segmenter.load()` accept a `domain` option.

**Acceptance:** on a 1,000-line prose validation set, segmentation F1 is ≥ 95% (current model is unmeasured on prose but qualitatively poor).

**Touches:** `training/train.py`, `data/`, `model/`.

### 3.4 Resolve the `ꦲ` ambiguity in `fromAksara`

Known limitation: `ꦲ` is both `h` and the carrier for standalone vowels. `fromAksara('ꦲꦗꦶ')` returns `'haji'` instead of `'aji'`. A probabilistic decoder that consults the language model (same `CharNgramLM` from 2.3) can pick the more likely reading per Aksara region.

**Acceptance:** on a sample of 50 ambiguous sequences, the LM-assisted decoder picks the intended reading ≥ 80% of the time.

**Touches:** `src/aksara.ts`, `src/lm.ts`.

---

## Phase 4 — Polish and distribution (3+ months)

Lower-priority. Ship only after Phase 1–3 are stable.

- **CI:** add a GitHub Actions workflow that runs `bun test` and `bun run build` on push and PR.
- **Coverage:** add tests for `Segmenter` (the suite currently only covers `Aksara`).
- **TypeScript target:** modernise `tsconfig.json` from `es2016` to `es2020` or `es2022` to use `??`, `?.`, top-level await in user code.
- **Browser support:** the segmenter currently depends on `onnxruntime-node`. A `onnxruntime-web` build would let the same code run in the browser. Investigate WASM size and the model format compatibility.
- **CLI tool:** ship a `bin/aksara` binary so users can `npx aksara-ts 'lamun sira nginguk ucing'` without writing a TypeScript file.
- **More scripts:** `transliterate_corpus.js` exists in `training/` but is undocumented; move it to `scripts/` with a README entry.

---

## Out of scope

Items that look related but should be declined unless the user asks:

- Training a Javanese **language model from scratch**. The segmenter uses a character n-gram LM (cheap, good enough). A neural LM is a separate project.
- A full GUI / web app. The library is the deliverable.
- Supporting other Indonesian scripts (Sundanese `ᮃᮊ᮪ᮞᮛ`, Balinese). Different alphabets, different codepoint ranges, different morphology — that's a fork, not a roadmap item.
- A desktop OCR app. The pipeline works headless; a GUI is its own product.

---

## Tracking

Each phase maps to one milestone on the GitHub project. Issues should reference the phase number (e.g. `Phase 1.1: OCR TypeScript runtime`). The README's brief `## Roadmap` section will be replaced with a one-line pointer to this file once Phase 1 ships.
