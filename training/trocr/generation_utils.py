#!/usr/bin/env python3
"""Free-gen helpers for Javanese TrOCR — anti-runaway sandhangan / cecak loops.

History: no_repeat_ngram_size was forced to 0 because byte-level BPE made ngram=3
illegal for valid Aksara. Vocab is now atomic (~50361), so we keep ngram=0 and
instead apply a targeted LogitsProcessor that:
  1) bans consecutive repeats of sandhangan/mark tokens (ꦁꦁ etc. is invalid),
  2) bans any token repeating >= 3 times in a row (allows rare XX, kills XXXX…).

Processors are applied at score/inference time (not serializable in Hub
generation_config). Set ANTI_LOOP=0 to reproduce the old unguarded generate().
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor

# Marks that must not repeat consecutively (orthography).
SANDHANGAN_CHARS: frozenset[str] = frozenset(
    {
        "ꦁ",  # cecak -ng
        "ꦂ",  # layar -r
        "ꦃ",  # wignyan -h
        "꧀",  # pangkon / virama
        "ꦼ",  # pepet
        "꧈",  # pada lingsa
        "꧉",  # pada lungsi
    }
)

DEFAULT_MAX_NEW_TOKENS = 64
MAX_SAME_TOKEN_RUN = 3  # ban when run length would reach this


def anti_loop_enabled() -> bool:
    """Env escape hatch: ANTI_LOOP=0 disables the processor."""
    return os.environ.get("ANTI_LOOP", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def estimate_char_budget(image: Image.Image) -> int:
    """Aksara count from ink bbox (square-padded 384 canvases look ~1:1 overall)."""
    try:
        from image_prep import estimate_char_budget as _ink_budget

        return _ink_budget(image)
    except Exception:
        w, h = image.size
        return max(1, int(round(w / max(h * 0.6, 1.0))))


def width_adaptive_max_new_tokens(
    image: Image.Image | None,
    *,
    tokens_per_char: float = 1.15,
    hard_cap: int = DEFAULT_MAX_NEW_TOKENS,
) -> int:
    if image is None:
        return hard_cap
    w, h = image.size
    # Square 384×384 canvases make ink aspect look "tall" (sandhangan), so the
    # line-width heuristic under-counts and truncates free-gen (v6 exact-match).
    if h > 0 and min(w, h) / max(w, h) >= 0.9:
        return int(hard_cap)
    chars = estimate_char_budget(image)
    return int(min(hard_cap, max(4, chars * tokens_per_char + 2)))


def resolve_mark_token_ids(tokenizer) -> set[int]:
    """Map sandhangan chars to atomic tokenizer ids (skip multi-id / unk)."""
    ids: set[int] = set()
    unk = getattr(tokenizer, "unk_token_id", None)
    for ch in SANDHANGAN_CHARS:
        tid = tokenizer.convert_tokens_to_ids(ch)
        if tid is None or tid < 0:
            continue
        if unk is not None and tid == unk:
            # Fall back to encode if convert_tokens_to_ids missed an added token.
            enc = tokenizer(ch, add_special_tokens=False).input_ids
            if len(enc) == 1 and enc[0] != unk:
                ids.add(int(enc[0]))
            continue
        enc = tokenizer(ch, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return ids


class NoRunawayMarksLogitsProcessor(LogitsProcessor):
    """Block sandhangan double-taps and long identical-token runs."""

    def __init__(
        self,
        mark_token_ids: set[int],
        *,
        max_same_run: int = MAX_SAME_TOKEN_RUN,
    ):
        self.mark_token_ids = set(mark_token_ids)
        self.max_same_run = max(2, int(max_same_run))

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # input_ids: (batch, seq); scores: (batch, vocab)
        batch, seq_len = input_ids.shape
        if seq_len < 1:
            return scores
        for b in range(batch):
            last = int(input_ids[b, -1].item())
            # Consecutive sandhangan/mark: never allow a second copy in a row.
            if last in self.mark_token_ids:
                scores[b, last] = torch.finfo(scores.dtype).min
                continue
            # Any token: ban extending a run to max_same_run (e.g. third copy).
            run = 1
            for t in range(seq_len - 2, -1, -1):
                if int(input_ids[b, t].item()) == last:
                    run += 1
                else:
                    break
            if run + 1 >= self.max_same_run:
                scores[b, last] = torch.finfo(scores.dtype).min
        return scores


def build_anti_loop_processors(processor: TrOCRProcessor) -> LogitsProcessorList:
    marks = resolve_mark_token_ids(processor.tokenizer)
    return LogitsProcessorList(
        [NoRunawayMarksLogitsProcessor(marks, max_same_run=MAX_SAME_TOKEN_RUN)]
    )


def trocr_generate(
    model,
    processor: TrOCRProcessor,
    pixel_values: torch.Tensor,
    *,
    image: Image.Image | None = None,
    anti_loop: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    num_beams: int = 1,
    **overrides,
):
    """Greedy (default) free-gen with optional anti-runaway processors.

    Returns generate() token id tensor.
    """
    use_anti = anti_loop_enabled() if anti_loop is None else bool(anti_loop)
    cls_id = processor.tokenizer.cls_token_id
    eos_id = processor.tokenizer.sep_token_id
    if eos_id is None:
        eos_id = processor.tokenizer.eos_token_id
    if max_new_tokens is None:
        max_new_tokens = width_adaptive_max_new_tokens(image)

    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "num_beams": int(num_beams),
        "do_sample": False,
        "decoder_start_token_id": cls_id,
        "eos_token_id": eos_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "no_repeat_ngram_size": 0,
        "use_cache": True,
    }
    if use_anti:
        kwargs["logits_processor"] = build_anti_loop_processors(processor)
    kwargs.update(overrides)
    # Allow caller to force-disable via override
    if anti_loop is False:
        kwargs.pop("logits_processor", None)
    return model.generate(pixel_values, **kwargs)


def _smoke_processor() -> None:
    """Minimal self-check without a full model (token id resolution + mask)."""
    # Synthetic scores: vocab=8, last token id=3 is a "mark".
    proc = NoRunawayMarksLogitsProcessor(mark_token_ids={3}, max_same_run=3)
    scores = torch.zeros(1, 8)
    # After a mark, that mark logit must be crushed.
    ids = torch.tensor([[1, 3]], dtype=torch.long)
    out = proc(ids, scores.clone())
    assert out[0, 3] == torch.finfo(out.dtype).min, "mark repeat not masked"
    # After two identical non-marks, third must be crushed (run+1 >= 3).
    scores2 = torch.zeros(1, 8)
    ids2 = torch.tensor([[5, 5]], dtype=torch.long)
    out2 = proc(ids2, scores2.clone())
    assert out2[0, 5] == torch.finfo(out2.dtype).min, "triple run not masked"
    # Single non-mark may repeat once (XX allowed).
    scores3 = torch.zeros(1, 8)
    ids3 = torch.tensor([[5]], dtype=torch.long)
    out3 = proc(ids3, scores3.clone())
    assert out3[0, 5] == 0, "single non-mark should still be allowed once more"
    print("[OK] NoRunawayMarksLogitsProcessor smoke passed", flush=True)


if __name__ == "__main__":
    _smoke_processor()
