#!/usr/bin/env python3
"""
clean_javanese_corpus.py — Strip wiki/HTML markup and keep OCR-friendly Aksara lines.

Reads training/javanese_corpus_clean.txt (or --input) and writes a cleaned corpus
suited for synthetic TrOCR line images:
  - Removes [[wiki]] / {{templates}} / HTML tags / entities
  - Keeps characters in the Javanese block plus basic punctuation/space
  - Drops lines that are too short, too long, or mostly non-Aksara
  - Splits overlong lines on spaces / ZWJ-ish boundaries when possible

Usage:
  python clean_javanese_corpus.py
  python clean_javanese_corpus.py --input ../javanese_corpus_clean.txt --output ../javanese_corpus_ocr.txt
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

# Javanese block + common digits that appear in manuscripts (Javanese digits A9D0–A9D9)
_JAV_RE = re.compile(r"[\uA980-\uA9DF]+")
_KEEP_RE = re.compile(r"[\uA980-\uA9DF\s\.,;:!\?\-\(\)\"'’‘]+")

_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_WIKI_FILE = re.compile(r"\[\[(?:File|Image|Berkas|Gambar)\s*:[^\]]*\]\]", re.I)
_WIKI_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_REF = re.compile(r"\[\d+\]")
_MULTI_SPACE = re.compile(r"\s+")


def strip_markup(text: str) -> str:
    text = html.unescape(text)
    text = _WIKI_FILE.sub(" ", text)
    # Unwrap [[label|display]] / [[page]] → keep visible text
    text = _WIKI_LINK.sub(r"\1", text)
    # Drop leftover brackets from broken markup
    text = text.replace("[[", " ").replace("]]", " ")
    # Remove templates (one pass; nested rare in this dump)
    for _ in range(3):
        nxt = _WIKI_TEMPLATE.sub(" ", text)
        if nxt == text:
            break
        text = nxt
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _REF.sub(" ", text)
    # Keep only Aksara + light punctuation
    text = "".join(ch if _KEEP_RE.match(ch) else " " for ch in text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def javanese_ratio(text: str) -> float:
    if not text:
        return 0.0
    jav = sum(1 for ch in text if "\uA980" <= ch <= "\uA9DF")
    return jav / len(text.replace(" ", "")) if text.replace(" ", "") else 0.0


def chunk_line(text: str, max_chars: int) -> list[str]:
    """Split long cleaned lines into OCR-sized chunks."""
    if len(text) <= max_chars:
        return [text] if text else []
    parts: list[str] = []
    # Prefer splitting on spaces; fall back to hard cuts
    tokens = text.split(" ")
    buf = ""
    for tok in tokens:
        if not tok:
            continue
        candidate = f"{buf} {tok}".strip() if buf else tok
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            parts.append(buf)
        if len(tok) <= max_chars:
            buf = tok
        else:
            for i in range(0, len(tok), max_chars):
                chunk = tok[i : i + max_chars]
                if len(chunk) >= 2:
                    parts.append(chunk)
            buf = ""
    if buf and len(buf) >= 2:
        parts.append(buf)
    return parts


def clean_file(inp: Path, out: Path, min_chars: int, max_chars: int, min_jav_ratio: float) -> None:
    raw = inp.read_text(encoding="utf-8").splitlines()
    cleaned: list[str] = []
    seen: set[str] = set()
    dropped = {"empty": 0, "short": 0, "ratio": 0, "dup": 0}

    for line in raw:
        line = strip_markup(line.strip())
        if not line:
            dropped["empty"] += 1
            continue
        for piece in chunk_line(line, max_chars):
            if len(piece) < min_chars:
                dropped["short"] += 1
                continue
            if javanese_ratio(piece) < min_jav_ratio:
                dropped["ratio"] += 1
                continue
            if piece in seen:
                dropped["dup"] += 1
                continue
            seen.add(piece)
            cleaned.append(piece)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(cleaned) + "\n", encoding="utf-8")
    print(f"[OK] {inp} -> {out}")
    print(f"     input lines:  {len(raw)}")
    print(f"     output lines: {len(cleaned)}")
    print(f"     dropped:      {dropped}")
    if cleaned:
        lens = sorted(len(x) for x in cleaned)
        print(
            f"     len p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} "
            f"p99={lens[int(len(lens)*0.99)]} max={lens[-1]}"
        )


def main() -> None:
    here = Path(__file__).resolve().parent
    training = here.parent
    p = argparse.ArgumentParser(description="Clean Javanese corpus for OCR dataset generation.")
    p.add_argument("--input", type=Path, default=training / "javanese_corpus_clean.txt")
    p.add_argument("--output", type=Path, default=training / "javanese_corpus_ocr.txt")
    p.add_argument("--min_chars", type=int, default=3)
    p.add_argument("--max_chars", type=int, default=48, help="Hard cap for OCR line length (chars).")
    p.add_argument("--min_jav_ratio", type=float, default=0.85)
    args = p.parse_args()
    if not args.input.exists():
        sys.exit(f"[ERROR] Input corpus not found: {args.input}")
    clean_file(args.input, args.output, args.min_chars, args.max_chars, args.min_jav_ratio)


if __name__ == "__main__":
    main()
