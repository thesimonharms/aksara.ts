#!/usr/bin/env python3
"""Large local CER verify on Hub validation samples."""

from __future__ import annotations

import os
import sys
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import torch
from datasets import Image as HFImage, load_dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from device_utils import attn_implementation, pick_device

MODEL_ID = os.environ.get("HUB_MODEL_ID", "thesimonharms/trocr-javanese-synthetic-v2")
DATASET_ID = os.environ.get("DATASET_NAME", "thesimonharms/javanese-dataset")
SPLIT = os.environ.get("VERIFY_SPLIT", "validation")
N = int(os.environ.get("N_SAMPLES", "1500"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))


def cer(pred: str, ref: str) -> float:
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


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = pick_device()
    print(f"[INFO] device={device} model={MODEL_ID} split={SPLIT} N={N}", flush=True)

    processor = TrOCRProcessor.from_pretrained(MODEL_ID, token=token)
    load_kw: dict = {"token": token}
    attn = attn_implementation()
    if attn:
        load_kw["attn_implementation"] = attn
        print(f"[INFO] attn_implementation={attn}", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, **load_kw)
    model.to(device)
    model.eval()
    cls_id = processor.tokenizer.cls_token_id
    model.config.decoder_start_token_id = cls_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = cls_id
        model.generation_config.no_repeat_ngram_size = 0
    print(
        f"[INFO] vocab={len(processor.tokenizer)} decoder_start={cls_id}",
        flush=True,
    )

    ds = load_dataset(DATASET_ID, split=SPLIT, token=token)
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage())
    n = min(N, len(ds))
    print(f"[INFO] Scoring {n} / {len(ds)} {SPLIT} examples", flush=True)

    scores: list[float] = []
    exact = 0
    len_ratios: list[float] = []
    short_scores: list[float] = []  # ref len <= 8
    long_scores: list[float] = []
    t0 = time.time()

    with torch.inference_mode():
        for i in range(n):
            ex = ds[i]
            image = to_rgb(ex["image"])
            ref = (ex.get("text") or ex.get("label") or "").strip()
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(
                pv,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
                decoder_start_token_id=cls_id,
                no_repeat_ngram_size=0,
                use_cache=True,
            )
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            score = cer(pred, ref)
            scores.append(score)
            if pred == ref:
                exact += 1
            if ref:
                len_ratios.append(len(pred) / len(ref))
            if len(ref) <= 8:
                short_scores.append(score)
            else:
                long_scores.append(score)

            if (i + 1) % LOG_EVERY == 0 or i + 1 == n:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n - i - 1) / max(rate, 1e-6)
                mean = sum(scores) / len(scores)
                print(
                    f"[PROG] {i+1}/{n} mean_CER={mean:.4f} "
                    f"exact={exact/(i+1):.2%} "
                    f"{rate:.2f} samp/s ETA={eta/60:.1f}m",
                    flush=True,
                )

    mean = sum(scores) / max(1, len(scores))
    sorted_s = sorted(scores)
    p50 = sorted_s[len(sorted_s) // 2]
    p90 = sorted_s[int(len(sorted_s) * 0.9)]
    print("\n=== Summary ===", flush=True)
    print(f"[OK] n={n} mean CER={mean:.4f}", flush=True)
    print(f"[OK] exact-match={exact/n:.2%}", flush=True)
    print(f"[OK] CER p50={p50:.4f} p90={p90:.4f}", flush=True)
    if len_ratios:
        print(
            f"[OK] mean pred_len/ref_len={sum(len_ratios)/len(len_ratios):.3f}",
            flush=True,
        )
    if short_scores:
        print(
            f"[OK] short(≤8) n={len(short_scores)} mean CER={sum(short_scores)/len(short_scores):.4f}",
            flush=True,
        )
    if long_scores:
        print(
            f"[OK] long(>8) n={len(long_scores)} mean CER={sum(long_scores)/len(long_scores):.4f}",
            flush=True,
        )
    print(f"[OK] wall={time.time()-t0:.1f}s", flush=True)
    if mean > 0.5:
        sys.exit(2)


if __name__ == "__main__":
    main()
