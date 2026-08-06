#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.46,<5.0",
#     "datasets>=2.20",
#     "sentencepiece>=0.2.0",
#     "python-dotenv>=1.0",
#     "pillow>=10.0",
#     "torch",
# ]
# ///
"""Quick sanity check: load fine-tuned TrOCR and score a few Hub val samples.

Compares baseline generate() vs width-adaptive inference tweaks (max_new_tokens,
optional min_new_tokens, length_penalty + light beam for longer lines).

HF Jobs example:
  hf jobs uv run --flavor t4-small --timeout 20m --secrets HF_TOKEN \\
    training/trocr/verify_trocr.py
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    # Local: repo_root/.env via training/trocr/verify_trocr.py → parents[2]
    # HF Jobs flat /app layout: no parents[2] — secrets already in env.
    _file = Path(__file__).resolve()
    if len(_file.parents) > 2:
        _env = _file.parents[2] / ".env"
        if _env.is_file():
            load_dotenv(_env)

import torch
from datasets import Image as HFImage, load_dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from device_utils import attn_implementation, pick_device
from generation_utils import anti_loop_enabled, trocr_generate

MODEL_ID = os.environ.get("HUB_MODEL_ID", "thesimonharms/trocr-javanese-synthetic")
# Repo root holds the expanded-tokenizer checkpoint (vocab ~50361).
# Trainer also pushes under final/ — that copy is still the old 50265 vocab.
# Set HUB_MODEL_SUBFOLDER=final only when you intentionally want that older tree.
MODEL_SUBFOLDER = os.environ.get("HUB_MODEL_SUBFOLDER", "").strip()
DATASET_ID = os.environ.get("DATASET_NAME", "thesimonharms/javanese-dataset")
N = int(os.environ.get("N_SAMPLES", "24"))
# "both" | "baseline" | "tweak"
MODE = os.environ.get("VERIFY_MODE", "both").strip().lower()


def cer(pred: str, ref: str) -> float:
    # Simple char-level Levenshtein / len(ref)
    if not ref:
        return 0.0 if not pred else 1.0
    a, b = pred, ref
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            if ca == cb:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / max(1, len(b))


def to_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        raw = img.get("bytes")
        if raw:
            return Image.open(BytesIO(raw)).convert("RGB")
        path = img.get("path")
        if path:
            return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)!r}")


def safe(s: str) -> str:
    return s.encode("ascii", "backslashreplace").decode("ascii")


def estimate_char_budget(image: Image.Image) -> int:
    """Rough aksara count from line-image aspect ratio (synthetic renders)."""
    w, h = image.size
    # Glyph width is typically ~0.5–0.7× line height for these fonts.
    return max(1, int(round(w / max(h * 0.6, 1.0))))


def gen_kwargs_baseline(cls_id: int) -> dict:
    return {
        "max_new_tokens": 64,
        "num_beams": 1,
        "do_sample": False,
        "decoder_start_token_id": cls_id,
        "no_repeat_ngram_size": 0,
        "length_penalty": 1.0,
        "use_cache": True,
    }


def gen_kwargs_tweak(image: Image.Image, cls_id: int, tokens_per_char: float) -> dict:
    """Width-adaptive caps + mild length encouragement on longer lines."""
    chars = estimate_char_budget(image)
    # Cap tokens from geometry so short crops cannot wander into phrase-length output.
    max_tok = int(min(96, max(4, chars * tokens_per_char + 2)))
    # Soft floor against early EOS on medium/long lines (preds were truncating).
    min_tok = int(max(1, min(max_tok - 1, chars * tokens_per_char * 0.45)))

    # Short lines: greedy + tight cap. Longer: light beam + length_penalty > 1.
    if chars <= 4:
        return {
            "max_new_tokens": max_tok,
            "min_new_tokens": 1,
            "num_beams": 1,
            "do_sample": False,
            "decoder_start_token_id": cls_id,
            "no_repeat_ngram_size": 0,
            "length_penalty": 1.0,
            "early_stopping": True,
            "use_cache": True,
            "_meta": {"chars_est": chars, "max_tok": max_tok, "min_tok": 1, "beams": 1},
        }

    return {
        "max_new_tokens": max_tok,
        "min_new_tokens": min_tok,
        "num_beams": 4,
        "do_sample": False,
        "decoder_start_token_id": cls_id,
        "no_repeat_ngram_size": 0,
        # >1 encourages longer sequences under beam search (preds were too short).
        "length_penalty": 1.15,
        "early_stopping": True,
        "use_cache": True,
        "_meta": {
            "chars_est": chars,
            "max_tok": max_tok,
            "min_tok": min_tok,
            "beams": 4,
        },
    }


def summarize(label: str, rows: list[dict]) -> dict:
    n = max(1, len(rows))
    mean_cer = sum(r["cer"] for r in rows) / n
    exact = sum(1 for r in rows if r["ref"] == r["pred"]) / n
    mean_len_ratio = sum(r["len_ratio"] for r in rows) / n
    mean_abs_len_err = sum(abs(len(r["pred"]) - len(r["ref"])) for r in rows) / n
    print(f"\n=== {label} ({len(rows)} samples) ===", flush=True)
    print(f"[OK] mean CER: {mean_cer:.4f}", flush=True)
    print(f"[OK] exact-match rate: {exact:.2%}", flush=True)
    print(f"[OK] mean pred_len/ref_len: {mean_len_ratio:.3f}", flush=True)
    print(f"[OK] mean |pred_len-ref_len|: {mean_abs_len_err:.2f}", flush=True)
    return {
        "mean_cer": mean_cer,
        "exact": exact,
        "mean_len_ratio": mean_len_ratio,
        "mean_abs_len_err": mean_abs_len_err,
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = pick_device()
    print(f"[INFO] device={device} model={MODEL_ID} subfolder={MODEL_SUBFOLDER!r}", flush=True)
    print(f"[INFO] VERIFY_MODE={MODE} N_SAMPLES={N}", flush=True)

    load_kw: dict = {"token": token}
    attn = attn_implementation()
    if attn:
        load_kw["attn_implementation"] = attn
    if MODEL_SUBFOLDER:
        load_kw["subfolder"] = MODEL_SUBFOLDER
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_ID, token=token)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, **load_kw)
        where = f"{MODEL_ID}/{MODEL_SUBFOLDER}" if MODEL_SUBFOLDER else MODEL_ID
        print(f"[INFO] Loaded from {where}", flush=True)
    except Exception as exc:
        if MODEL_SUBFOLDER:
            print(f"[WARN] subfolder load failed ({exc}); trying repo root", flush=True)
            processor = TrOCRProcessor.from_pretrained(MODEL_ID, token=token)
            model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, token=token)
        else:
            raise

    model.to(device)
    model.eval()
    # Force correct TrOCR start token (CLS=0). A stale generation_config with
    # decoder_start=EOS(2) produces near-random unicode while train loss looks fine.
    cls_id = processor.tokenizer.cls_token_id
    model.config.decoder_start_token_id = cls_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = cls_id
        model.generation_config.no_repeat_ngram_size = 0

    vocab_size = len(processor.tokenizer)
    # Expanded Javanese vocab is ~50361; stock TrOCR is 50265.
    tokens_per_char = 1.15 if vocab_size > 50300 else 3.0
    print(
        f"[INFO] decoder_start_token_id={model.generation_config.decoder_start_token_id} "
        f"vocab_size={vocab_size} tokens_per_char≈{tokens_per_char}",
        flush=True,
    )

    print(f"[INFO] Loading validation split from {DATASET_ID} …", flush=True)
    ds = load_dataset(DATASET_ID, split="validation", token=token)
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage())
    n = min(N, len(ds))
    print(f"[INFO] Scoring {n} / {len(ds)} validation examples", flush=True)

    modes = []
    if MODE in ("both", "baseline"):
        modes.append("baseline")
    if MODE in ("both", "tweak"):
        modes.append("tweak")
    if not modes:
        print(f"[ERR] unknown VERIFY_MODE={MODE!r}", flush=True)
        sys.exit(1)

    results: dict[str, list[dict]] = {m: [] for m in modes}

    with torch.inference_mode():
        for i in range(n):
            ex = ds[i]
            image = to_rgb(ex["image"])
            ref = (ex.get("text") or ex.get("label") or "").strip()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
                device
            )
            gold = processor.tokenizer(
                ref, return_tensors="pt", add_special_tokens=True
            ).input_ids.to(device)

            tf_out = model(pixel_values=pixel_values, labels=gold)
            pred_tf = tf_out.logits.argmax(-1)
            tf_acc = (pred_tf[:, :-1] == gold[:, 1:]).float().mean().item()
            if i < 3:
                print(
                    f"[DBG] sample {i} tf_loss={tf_out.loss.item():.4f} "
                    f"tf_next_token_acc={tf_acc:.3f} "
                    f"img={image.size[0]}x{image.size[1]} "
                    f"chars_est={estimate_char_budget(image)} ref_len={len(ref)}",
                    flush=True,
                )

            for mode in modes:
                if mode == "baseline":
                    ids = trocr_generate(
                        model,
                        processor,
                        pixel_values,
                        image=image,
                        num_beams=1,
                    )
                    meta = {
                        "chars_est": estimate_char_budget(image),
                        "max_tok": 64,
                        "anti_loop": anti_loop_enabled(),
                    }
                else:
                    kw = gen_kwargs_tweak(image, cls_id, tokens_per_char)
                    meta = kw.pop("_meta")
                    meta["anti_loop"] = anti_loop_enabled()
                    # Keep tweak beams/length_penalty; still attach anti-loop.
                    ids = trocr_generate(
                        model,
                        processor,
                        pixel_values,
                        image=image,
                        max_new_tokens=kw.get("max_new_tokens"),
                        num_beams=kw.get("num_beams", 4),
                        min_new_tokens=kw.get("min_new_tokens"),
                        length_penalty=kw.get("length_penalty", 1.0),
                        early_stopping=kw.get("early_stopping", True),
                    )

                pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
                score = cer(pred, ref)
                len_ratio = (len(pred) / len(ref)) if ref else (0.0 if not pred else 1.0)
                results[mode].append(
                    {
                        "cer": score,
                        "ref": ref,
                        "pred": pred,
                        "len_ratio": len_ratio,
                        "tf_loss": tf_out.loss.item(),
                        "tf_acc": tf_acc,
                        "meta": meta,
                    }
                )

                if i < 8 or score < 0.35:
                    print(
                        f"--- sample {i} [{mode}] CER={score:.3f} "
                        f"len={len(pred)}/{len(ref)} meta={meta}",
                        flush=True,
                    )
                    print(f"  REF : {safe(ref)}", flush=True)
                    print(f"  PRED: {safe(pred)}", flush=True)

    summaries = {m: summarize(m, results[m]) for m in modes}
    if "baseline" in summaries and "tweak" in summaries:
        d = summaries["baseline"]["mean_cer"] - summaries["tweak"]["mean_cer"]
        print(
            f"\n[COMPARE] tweak CER delta vs baseline: {d:+.4f} "
            f"(positive = tweak better)",
            flush=True,
        )
        print(
            f"[COMPARE] length ratio baseline={summaries['baseline']['mean_len_ratio']:.3f} "
            f"tweak={summaries['tweak']['mean_len_ratio']:.3f}",
            flush=True,
        )

    best = min(summaries.values(), key=lambda s: s["mean_cer"])
    if best["mean_cer"] > 0.5:
        print(
            "[WARN] High CER — model may not have learned script well, or wrong checkpoint/path.",
            flush=True,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
