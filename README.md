# aksara.ts

A TypeScript library for bidirectional transliteration between Latin-script Javanese and **Aksara Jawa** (Hanacaraka), paired with a neural word segmenter — built to give AI systems visibility into a script that is essentially absent from their training data.

```
ꦭꦩꦸꦤ꧀ꦱꦶꦫꦔꦶꦔꦸꦏꦸꦕꦶꦁ  →  lamun sira nginguk ucing
lamun sira nginguk ucing   →  ꦭꦩꦸꦤ꧀ꦱꦶꦫꦔꦶꦔꦸꦏꦸꦕꦶꦁ
```

## Why this exists

Aksara Jawa is essentially invisible to AI. Large language models were trained on internet-scale text — Aksara Jawa has almost none of it. When a model encounters `ꦲꦤꦕꦫꦏ`, it sees a sequence of rare Unicode codepoints with no semantic weight attached.

This is a preservation problem. Thousands of Javanese manuscripts exist only in physical form. Digitising them with OCR produces Unicode output that no downstream AI tool can use without a transliteration layer sitting in between.

aksara.ts is that layer. The intended pipeline:

```
manuscript image → OCR → Aksara Unicode → fromAksara() → Segmenter → readable Javanese → LLM
```

## Background

Aksara Jawa is an **abugida** — a script where each consonant glyph carries an inherent *a* vowel modified by diacritics. It has been used to write Javanese for centuries and remains culturally significant today, though it exists almost entirely outside the training distribution of modern AI models.

## Installation

```bash
bun add aksara-ts
# or
npm install aksara-ts
```

## Usage

### Forward: Latin → Aksara

```typescript
import { Aksara } from 'aksara-ts';

new Aksara('hanacaraka').getAksara();        // → 'ꦲꦤꦕꦫꦏ'
new Aksara('aji saka', true).getAksara();    // → 'ꦲꦗꦶ ꦱꦏ'
new Aksara('wong jawa').getAksara();         // → 'ꦮꦺꦴꦁꦗꦮ'
new Aksara('kra').getAksara();               // → 'ꦏꦿ'   (cakra for medial r)
new Aksara('1234').getAksara();              // → '꧑꧒꧓꧔'  (Javanese numerals)
new Aksara('aksara', false, true).getAksara(); // → 'ꦄꦏ꧀ꦱꦫ' (explicit vowel letters)
```

### Reverse: Aksara → Latin

```typescript
import { Aksara } from 'aksara-ts';

Aksara.fromAksara('ꦲꦤꦕꦫꦏ');      // → 'hanacaraka'
Aksara.fromAksara('ꦮꦺꦴꦁꦗꦮ');      // → 'wong jawa'
Aksara.fromAksara('ꦧꦸꦟ꧀ꦝꦼꦭ꧀');   // → 'bunḍel'   (murda Na + retroflex Dda)
```

`fromAksara` handles the full modern consonant set plus murda (prestige) letters, retroflex consonants, vocalic syllables (ꦉ ꦊ), and the pengkal subscript (ꦾ).

**Known limitation:** ꦲ is ambiguous — it is both the consonant `h` and the carrier for standalone vowels under the *h*+vowel convention. `fromAksara('ꦲꦗꦶ')` returns `'haji'`, not `'aji'`. This is irreducible without explicit vowel letters.

### Word segmentation

Aksara Jawa uses no spaces between words. After decoding a manuscript with `fromAksara`, the output is a continuous character stream. The `Segmenter` class restores word boundaries using a BiLSTM model trained on Javanese Wikipedia.

```typescript
import { Segmenter } from 'aksara-ts/segmenter';

const segmenter = await Segmenter.load('./model/segmenter.onnx', './model/vocab.json');

await segmenter.segment('lambungkiwatémbongputih');
// → 'lambung kiwate mbong putih'
```

The model is not bundled — see [Training](#training) to produce it.

### Syllable break marker

Use `_` to force an explicit syllable boundary when automatic syllabification gives the wrong result:

```typescript
new Aksara('angkra').getAksara();    // → 'ꦲꦁꦏꦿ'   (ang-kra: ng closes syllable)
new Aksara('a_ngkra').getAksara();   // → 'ꦲꦔ꧀ꦏꦿ'  (a-ngkra: ng starts cluster)
```

`_` produces no output — it only resets the syllable state machine.

## API

### `new Aksara(text, spaces?, explicitVowels?)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `string` | — | Latin-script Javanese text |
| `spaces` | `boolean` | `false` | Preserve spaces in output |
| `explicitVowels` | `boolean` | `false` | Use standalone vowel letters (ꦄ ꦆ ꦈ ꦌ ꦎ) for vowels without a preceding consonant |

### `.getAksara(): string`
Returns the Aksara Jawa string.

### `.getText(): string`
Returns the original input text.

### `Aksara.fromAksara(text: string): string`
Decodes an Aksara Jawa string to Latin-script Javanese. Handles the full consonant set including murda, retroflex, and vocalic syllables. Unknown codepoints pass through unchanged.

### `Segmenter.load(modelPath, vocabPath): Promise<Segmenter>`
Loads a trained ONNX segmentation model.

### `segmenter.segment(text, threshold?): Promise<string>`
Inserts word boundaries into unsegmented Latin-script Javanese. `threshold` (default `0.5`) controls sensitivity — lower values insert more spaces.

## Reference

**Vowels**

| Latin | Diacritic | Stand-alone | Explicit stand-alone |
|-------|-----------|-------------|----------------------|
| `a`   | (inherent) | `ꦲ` | `ꦄ` |
| `i`   | `ꦶ` | `ꦲꦶ` | `ꦆ` |
| `u`   | `ꦸ` | `ꦲꦸ` | `ꦈ` |
| `e`   | `ꦼ` | `ꦲꦼ` | `ꦌ` |
| `é`   | `ꦺ` | `ꦲꦺ` | `ꦲꦺ` (no standalone form) |
| `o`   | `ꦺꦴ` | `ꦲꦺꦴ` | `ꦎ` |

**Consonants** (forward direction)

`h n c r k d t s w l p j y m g b th dh ng ny`

**Additional consonants** (reverse direction only)

| Glyph | Name | Decodes as |
|-------|------|------------|
| `ꦟ` | Na Murda | `n` |
| `ꦑ` `ꦓ` `ꦦ` `ꦯ` `ꦬ` | Ka/Ga/Pa/Sa/Ra Murda | same as base form |
| `ꦛ` | Tta (retroflex t) | `ṭ` |
| `ꦝ` | Dda (retroflex d) | `ḍ` |
| `ꦉ` | Re (vocalic r) | `re` |
| `ꦊ` | Le (vocalic l) | `le` |
| `ꦾ` | Pengkal (subscript ya) | medial `y` |

**Punctuation**

| Latin | Javanese |
|-------|----------|
| `,` | `꧈` pada lingsa |
| `.` | `꧉` pada lungsi |

Digits `0`–`9` → `꧐`–`꧙`. Unknown characters pass through unchanged.

## Training

The word segmenter is trained separately in Python and exported to ONNX for use at runtime.

### Setup

```bat
training\setup.bat
training\venv\Scripts\activate
```

### Train

```bash
python training/train.py data/jv.txt
```

Trains a 2-layer bidirectional LSTM on character sequences. Saves the best checkpoint to `model/segmenter.pt` and `model/vocab.json`.

### Export

```bash
python training/export.py data/jv.txt
```

Exports the checkpoint to `model/segmenter.onnx` for inference from TypeScript via `onnxruntime-node`.

### Findings

Training on 15,309 lines (438,767 character positions) of Javanese localisation data from TranslateWiki via OPUS:

| Metric | Value |
|--------|-------|
| Vocabulary | 121 characters |
| Space density | 11.6% |
| Best val_acc | **99.11%** (epoch 30) |
| Parameters | 601,921 |

The training corpus is software UI translation strings (MediaWiki and related projects), not natural prose. This means the model is well-calibrated on short, formulaic modern Javanese sentences but has limited exposure to the vocabulary of classical or literary texts. Words common in manuscript sources — `lamun`, `yén`, `hutama` — appear rarely or not at all. Training on prose or manuscript-register data would substantially improve segmentation quality on historical material.

The segmenter currently operates on `fromAksara` output, which means it inherits the `ꦲ` ambiguity: `awak` (body) decodes as `hawak`, which the segmenter has not been trained to split correctly.

### Citation

Training data sourced from OPUS:

> J. Tiedemann, 2012, *Parallel Data, Tools and Interfaces in OPUS*. In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012).

## Development

```bash
bun test       # 77 tests
bun run build  # TypeScript compile
bun run demo   # end-to-end pipeline demo
```

## Roadmap

- [ ] **Retrain segmenter on broader data** — current model was trained on software localisation strings; needs exposure to natural prose and poetic/classical vocabulary to handle manuscript text reliably
- [ ] **Structured token output** — expose syllable boundaries, punctuation names, and verse markers as typed tokens for RAG and embedding pipelines
- [ ] **Unicode normalisation** — OCR output uses inconsistent codepoint sequences for the same glyph; a normalisation pass is a prerequisite for reliable decoding
- [ ] **Murda consonants in forward direction** — currently decode-only; forward support requires a notation convention for the input
- [ ] **OCR pipeline integration** — end-to-end example connecting an OCR engine to this library and a language model

## License

MIT © [Simon Harms](https://github.com/thesimonharms)
