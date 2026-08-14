"""Stratified length + rare-glyph sampling for HQ synthetic OCR."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleSpec:
    idx: int
    text: str
    text_id: str
    bucket: str
    rare: bool
    aug_seed: int


def load_corpus_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class StratifiedCorpus:
    """Draw texts with short/mid/long quotas and soft rare oversampling."""

    def __init__(
        self,
        rows: list[dict],
        *,
        short_frac: float = 0.15,
        mid_frac: float = 0.55,
        long_frac: float = 0.30,
        rare_boost: float = 2.0,
        rng: random.Random | None = None,
    ):
        if abs(short_frac + mid_frac + long_frac - 1.0) > 1e-6:
            raise ValueError("bucket fractions must sum to 1.0")
        self.short_frac = short_frac
        self.mid_frac = mid_frac
        self.long_frac = long_frac
        self.rare_boost = max(1.0, rare_boost)
        self.rng = rng or random.Random()
        self.by_bucket: dict[str, list[dict]] = {"short": [], "mid": [], "long": []}
        for r in rows:
            b = r.get("bucket") or "mid"
            if b not in self.by_bucket:
                b = "mid"
            self.by_bucket[b].append(r)
        # Fallback: if a bucket is empty, borrow from mid/all.
        all_rows = rows[:]
        for b in ("short", "mid", "long"):
            if not self.by_bucket[b]:
                self.by_bucket[b] = all_rows or [{"text": "ꦲ", "text_id": "fallback", "bucket": b, "rare": False}]

    def _pick_from_bucket(self, bucket: str) -> dict:
        pool = self.by_bucket[bucket]
        weights = [self.rare_boost if r.get("rare") else 1.0 for r in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def plan(self, count: int, *, seed: int) -> list[SampleSpec]:
        """Deterministic sample plan for resume-safe parallel render.

        O(count) — builds a shuffled multi-pass deck per bucket (unique-first,
        then controlled repeats) instead of scanning the pool on every draw.
        """
        rng = random.Random(seed)
        n_short = int(round(count * self.short_frac))
        n_long = int(round(count * self.long_frac))
        n_mid = count - n_short - n_long
        quotas = (["short"] * n_short) + (["mid"] * n_mid) + (["long"] * n_long)
        rng.shuffle(quotas)

        # How many full passes over each bucket pool we need for this count.
        total_pool = max(1, sum(len(p) for p in self.by_bucket.values()))
        max_passes = max(1, math.ceil(count / total_pool) + 1)

        decks: dict[str, list[dict]] = {}
        cursors = {b: 0 for b in ("short", "mid", "long")}
        for b in ("short", "mid", "long"):
            base = list(self.by_bucket[b])
            # Soft rare boost: duplicate rare rows once in the base deck.
            boosted = list(base)
            for r in base:
                if r.get("rare"):
                    boosted.append(r)
            deck: list[dict] = []
            for _ in range(max_passes):
                pass_rows = list(boosted)
                rng.shuffle(pass_rows)
                deck.extend(pass_rows)
            decks[b] = deck or [{"text": "ꦲ", "text_id": "fallback", "bucket": b, "rare": False}]

        specs: list[SampleSpec] = []
        for idx, bucket in enumerate(quotas):
            deck = decks[bucket]
            chosen = deck[cursors[bucket] % len(deck)]
            cursors[bucket] += 1
            specs.append(
                SampleSpec(
                    idx=idx,
                    text=chosen["text"],
                    text_id=str(chosen.get("text_id") or ""),
                    bucket=bucket,
                    rare=bool(chosen.get("rare")),
                    aug_seed=rng.randint(0, 2**31 - 1),
                )
            )
        return specs
