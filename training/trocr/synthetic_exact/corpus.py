"""Aksara-only short chunks with a hash holdout for val text."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

MARK_START = set("ꦀꦁꦂꦃꦴꦵꦶꦷꦸꦹꦺꦻꦼꦽꦾꦿ꧀")
PADA = set("꧈꧉꧊꧋꧌꧍")


def is_javanese(ch: str) -> bool:
    return 0xA980 <= ord(ch) <= 0xA9DF


def aksara_only(raw: str) -> str:
    line = unicodedata.normalize("NFC", raw)
    return "".join(ch for ch in line if is_javanese(ch))


def _ok_chunk(s: str, min_len: int, max_len: int) -> bool:
    if len(s) < min_len or len(s) > max_len:
        return False
    if s[0] in MARK_START:
        return False
    pangkon = s.count("꧀")
    if pangkon / len(s) > 0.45:
        return False
    letters = sum(1 for c in s if c not in MARK_START and c not in PADA)
    return letters >= 1


def chunk_aksara(s: str, *, min_len: int = 2, max_len: int = 12) -> list[str]:
    """Split aksara-only text into OCR-sized pieces that don't start on marks."""
    s = aksara_only(s)
    if not s:
        return []
    parts: list[str] = []
    buf: list[str] = []
    for ch in s:
        if ch in PADA:
            piece = "".join(buf)
            buf = []
            if _ok_chunk(piece, min_len, max_len):
                parts.append(piece)
            continue
        buf.append(ch)
        if len(buf) >= max_len:
            # Include trailing marks that belong to the last consonant.
            piece = "".join(buf)
            # If we ended mid-cluster, that's ok at max_len; next chunk skips marks.
            if _ok_chunk(piece, min_len, max_len):
                parts.append(piece)
            buf = []
    piece = "".join(buf)
    if _ok_chunk(piece, min_len, max_len):
        parts.append(piece)
    return parts


def line_hash_bucket(line: str) -> int:
    digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def load_clean_pools(
    paths: list[Path],
    *,
    min_len: int = 2,
    max_len: int = 12,
    val_pct: int = 5,
) -> tuple[list[str], list[str]]:
    """Return (train_texts, val_texts) unique aksara chunks."""
    train: list[str] = []
    val: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            print(f"[WARN] missing corpus {path}", flush=True)
            continue
        print(f"[INFO] reading {path}", flush=True)
        with path.open(encoding="utf-8") as f:
            for raw in f:
                for chunk in chunk_aksara(raw, min_len=min_len, max_len=max_len):
                    if chunk in seen:
                        continue
                    seen.add(chunk)
                    if line_hash_bucket(chunk) < val_pct:
                        val.append(chunk)
                    else:
                        train.append(chunk)
    return train, val
