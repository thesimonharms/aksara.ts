#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.46,<5.0",
#     "sentencepiece>=0.2.0",
#     "python-dotenv>=1.0",
#     "pillow>=10.0",
#     "torch",
# ]
# ///
"""Expand TrOCR's BPE tokenizer with whole Javanese Aksara characters.

Why: microsoft/trocr-*-handwritten tokenizes each Aksara letter as 3 UTF-8 byte
pieces. Free-run generation often emits those bytes out of order → CER≈1 even
when teacher-forced loss looks great. Adding each character as a single token
fixes the generation path.

Usage:
  python expand_javanese_tokenizer.py
  python expand_javanese_tokenizer.py --corpus ../javanese_corpus_ocr.txt --save_dir ./tokenizer_javanese
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import PreTrainedTokenizerBase, TrOCRProcessor, VisionEncoderDecoderModel


# Unicode Javanese block (letters, digits, punctuation used in Aksara text).
JAVANESE_BLOCK = [chr(c) for c in range(0xA980, 0xA9E0)]


def collect_javanese_chars(corpus_path: Path | None = None) -> list[str]:
    """Full Javanese block plus any extra chars found in the OCR corpus."""
    chars: set[str] = set(JAVANESE_BLOCK)
    if corpus_path is not None and corpus_path.is_file():
        text = corpus_path.read_text(encoding="utf-8")
        for ch in text:
            o = ord(ch)
            # Keep anything in/near the Javanese block that appears in data.
            if 0xA980 <= o <= 0xA9FF:
                chars.add(ch)
    # Stable order for reproducible vocab growth.
    return sorted(chars, key=ord)


def tokens_missing_from_vocab(tokenizer: PreTrainedTokenizerBase, chars: list[str]) -> list[str]:
    """Return chars that are not already single-token entries in the vocab."""
    vocab = tokenizer.get_vocab()
    missing: list[str] = []
    for ch in chars:
        if ch in vocab:
            continue
        # Also skip if encode already yields exactly one non-special id.
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        if len(ids) == 1 and tokenizer.decode(ids) == ch:
            continue
        missing.append(ch)
    return missing


def expand_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    chars: list[str] | None = None,
    corpus_path: Path | None = None,
) -> list[str]:
    """Add missing Javanese chars as atomic tokens. Returns the list that was added."""
    if chars is None:
        chars = collect_javanese_chars(corpus_path)
    to_add = tokens_missing_from_vocab(tokenizer, chars)
    if not to_add:
        print("[INFO] Tokenizer already covers all requested Javanese chars")
        return []
    n = tokenizer.add_tokens(to_add)
    print(f"[INFO] Added {n} Javanese character tokens (requested {len(to_add)})")
    return to_add


def resize_model_for_tokenizer(model: VisionEncoderDecoderModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Grow decoder input/output embeddings to match tokenizer length."""
    new_size = len(tokenizer)
    old = model.decoder.get_input_embeddings().weight.shape[0]
    # VisionEncoderDecoderModel has no top-level set_input_embeddings — resize decoder.
    model.decoder.resize_token_embeddings(new_size)
    model.config.vocab_size = new_size
    if hasattr(model.config, "decoder") and model.config.decoder is not None:
        model.config.decoder.vocab_size = new_size
    print(f"[INFO] Resized decoder embeddings {old} → {new_size}")


def expand_processor_and_model(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    corpus_path: Path | None = None,
) -> list[str]:
    """In-place expand processor.tokenizer and resize model embeddings."""
    added = expand_tokenizer(processor.tokenizer, corpus_path=corpus_path)
    if added:
        resize_model_for_tokenizer(model, processor.tokenizer)
    # Keep configs aligned for generate().
    model.config.vocab_size = len(processor.tokenizer)
    if hasattr(model.config, "decoder") and model.config.decoder is not None:
        model.config.decoder.vocab_size = len(processor.tokenizer)
    return added


def _self_check(tokenizer: PreTrainedTokenizerBase, sample: str = "ꦲꦤꦕꦫꦏ") -> None:
    ids = tokenizer(sample, add_special_tokens=False).input_ids
    dec = tokenizer.decode(ids, skip_special_tokens=True)
    per_char = [tokenizer(ch, add_special_tokens=False).input_ids for ch in sample]
    print(f"[CHECK] {sample!r} → {len(ids)} tokens (expect {len(sample)}) ids={ids}")
    print(f"[CHECK] decode={dec!r} match={dec == sample}")
    print(f"[CHECK] per-char lens={[len(x) for x in per_char]}")
    if len(ids) != len(sample) or dec != sample:
        raise SystemExit("[ERROR] Tokenizer expansion self-check failed")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Expand TrOCR tokenizer with Javanese chars")
    p.add_argument("--base_model", default="microsoft/trocr-base-handwritten")
    p.add_argument(
        "--corpus",
        type=Path,
        default=here.parent / "javanese_corpus_ocr.txt",
        help="Optional corpus to discover extra Javanese codepoints",
    )
    p.add_argument(
        "--save_dir",
        type=Path,
        default=here / "tokenizer_javanese",
        help="Where to save expanded processor (+ optionally resized base model)",
    )
    p.add_argument(
        "--save_model",
        action="store_true",
        help="Also save a resized (random-init new rows) base model copy",
    )
    args = p.parse_args()

    print(f"[INFO] Loading processor from {args.base_model}")
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    before = processor.tokenizer("ꦲꦤꦕ", add_special_tokens=False).input_ids
    print(f"[INFO] Before expansion 'ꦲꦤꦕ' → {len(before)} tokens: {before}")

    chars = collect_javanese_chars(args.corpus if args.corpus.is_file() else None)
    print(f"[INFO] Candidate Javanese chars: {len(chars)}")

    if args.save_model:
        model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
        added = expand_processor_and_model(processor, model, corpus_path=args.corpus)
    else:
        added = expand_tokenizer(processor.tokenizer, chars=chars)
        model = None

    _self_check(processor.tokenizer)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.save_dir)
    meta = {
        "base_model": args.base_model,
        "added_tokens": added,
        "vocab_size": len(processor.tokenizer),
        "corpus": str(args.corpus) if args.corpus else None,
    }
    (args.save_dir / "javanese_expand.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if model is not None:
        model.save_pretrained(args.save_dir)
        print(f"[OK] Saved expanded processor + resized model → {args.save_dir}")
    else:
        print(f"[OK] Saved expanded processor → {args.save_dir}")
        print("[INFO] Pass this dir as --tokenizer_dir to finetune, or use --expand_javanese_tokenizer")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
