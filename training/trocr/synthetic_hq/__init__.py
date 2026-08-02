"""Higher-quality synthetic Javanese Aksara OCR rendering (HQ pipeline).

Keep separate from generate_trocr_dataset.py so v1–v4 data stays reproducible.
"""

from __future__ import annotations

from .quality import passes_quality_gate
from .render import HqRenderer
from .sampler import StratifiedCorpus, SampleSpec, load_corpus_jsonl

__all__ = [
    "HqRenderer",
    "SampleSpec",
    "StratifiedCorpus",
    "load_corpus_jsonl",
    "passes_quality_gate",
]
