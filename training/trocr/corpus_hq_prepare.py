#!/usr/bin/env python3
"""Prepare held-out train/val text pools for the synthetic HQ OCR pipeline.

- Dedupe (Unicode NFC)
- Drop markup leftovers
- Tag length buckets + rare-aksara flags
- Hash-holdout: hash(line) % 100 < val_pct → val pool (default 5%)

Does NOT render images. Safe to run anytime.

Example:
  python corpus_hq_prepare.py \\
    --inputs ../javanese_corpus_ocr.txt \\
    --out-dir ../corpus_hq \\
    --val-pct 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

# Soft oversample only murda / mahaprana letters (genuinely rarer in wiki OCR).
# Sandhangan like cakra/pengkal appear too often to be useful "rare" tags.
RARE_FOCUS = frozenset("ꦟꦡꦣꦦꦨꦬꦯꦰ꧁꧂꧃꧄꧅")


def is_javanese_char(ch: str) -> bool:
    o = ord(ch)
    return 0xA980 <= o <= 0xA9DF


def clean_line(raw: str, min_chars: int, max_chars: int, min_jav_ratio: float) -> str | None:
    line = unicodedata.normalize("NFC", raw.strip())
    if not line or len(line) < min_chars:
        return None
    if "[[" in line or "]]" in line or "<" in line or ">" in line or "{{" in line:
        return None
    if len(line) > max_chars:
        line = line[:max_chars]
    jav = sum(1 for c in line if is_javanese_char(c))
    if jav / max(1, len(line)) < min_jav_ratio:
        return None
    return line


def length_bucket(n: int) -> str:
    if n <= 8:
        return "short"
    if n <= 24:
        return "mid"
    return "long"


def has_rare(line: str) -> bool:
    return any(c in RARE_FOCUS for c in line)


def line_hash_bucket(line: str) -> int:
    digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def prepare(
    inputs: list[Path],
    out_dir: Path,
    *,
    min_chars: int = 3,
    max_chars: int = 48,
    min_jav_ratio: float = 0.85,
    val_pct: int = 5,
) -> dict:
    seen: set[str] = set()
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    skipped = 0

    for path in inputs:
        if not path.is_file():
            print(f"[WARN] missing input {path}", flush=True)
            continue
        print(f"[INFO] reading {path}", flush=True)
        with path.open(encoding="utf-8") as f:
            for raw in f:
                line = clean_line(raw, min_chars, max_chars, min_jav_ratio)
                if line is None:
                    skipped += 1
                    continue
                if line in seen:
                    skipped += 1
                    continue
                seen.add(line)
                row = {
                    "text": line,
                    "text_id": hashlib.sha256(line.encode("utf-8")).hexdigest()[:16],
                    "bucket": length_bucket(len(line)),
                    "rare": has_rare(line),
                    "n_chars": len(line),
                }
                if line_hash_bucket(line) < val_pct:
                    val_rows.append(row)
                else:
                    train_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt = out_dir / "corpus_hq_train.txt"
    val_txt = out_dir / "corpus_hq_val.txt"
    train_meta = out_dir / "corpus_hq_train.jsonl"
    val_meta = out_dir / "corpus_hq_val.jsonl"
    summary_path = out_dir / "corpus_hq_summary.json"

    train_txt.write_text("\n".join(r["text"] for r in train_rows) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(r["text"] for r in val_rows) + "\n", encoding="utf-8")
    with train_meta.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_meta.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def bucket_counts(rows: list[dict]) -> dict[str, int]:
        c = Counter(r["bucket"] for r in rows)
        return {k: c[k] for k in ("short", "mid", "long")}

    summary = {
        "inputs": [str(p) for p in inputs],
        "train_lines": len(train_rows),
        "val_lines": len(val_rows),
        "skipped": skipped,
        "val_pct": val_pct,
        "train_buckets": bucket_counts(train_rows),
        "val_buckets": bucket_counts(val_rows),
        "train_rare": sum(1 for r in train_rows if r["rare"]),
        "val_rare": sum(1 for r in val_rows if r["rare"]),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "min_jav_ratio": min_jav_ratio,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[OK] wrote {train_txt} ({len(train_rows)})", flush=True)
    print(f"[OK] wrote {val_txt} ({len(val_rows)})", flush=True)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare HQ synthetic OCR text pools.")
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[Path(__file__).resolve().parent.parent / "javanese_corpus_ocr.txt"],
        help="One or more corpus text files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "corpus_hq",
        help="Output directory for train/val pools + metadata.",
    )
    p.add_argument("--min-chars", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=48)
    p.add_argument("--min-jav-ratio", type=float, default=0.85)
    p.add_argument("--val-pct", type=int, default=5, help="Hash holdout percent for val pool.")
    args = p.parse_args()
    prepare(
        list(args.inputs),
        args.out_dir,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_jav_ratio=args.min_jav_ratio,
        val_pct=max(1, min(20, args.val_pct)),
    )


if __name__ == "__main__":
    main()
