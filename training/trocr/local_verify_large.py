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

from device_utils import attn_implementation, pick_device, resolve_torch_device
from generation_utils import anti_loop_enabled, trocr_generate

MODEL_ID = os.environ.get("HUB_MODEL_ID", "thesimonharms/trocr-javanese-synthetic-v2")
MODEL_REVISION = os.environ.get("HUB_REVISION") or None
DATASET_ID = os.environ.get("DATASET_NAME", "thesimonharms/javanese-dataset")
SPLIT = os.environ.get("VERIFY_SPLIT", "validation")
N = int(os.environ.get("N_SAMPLES", "1500"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))


# Near-match = Levenshtein edit distance <= CLOSE_MAX (default 2).
CLOSE_MAX = int(os.environ.get("CLOSE_MAX", "2"))


def edit_distance(pred: str, ref: str) -> int:
    """Unicode-codepoint Levenshtein distance."""
    a, b = pred, ref
    if not a:
        return len(b)
    if not b:
        return len(a)
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
    return dp[-1]


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, ref) / max(1, len(ref))



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
    device_kind = pick_device()
    device = resolve_torch_device(device_kind)
    rev = f"@{MODEL_REVISION}" if MODEL_REVISION else ""
    print(
        f"[INFO] device={device_kind} ({device}) model={MODEL_ID}{rev} "
        f"split={SPLIT} N={N} anti_loop={anti_loop_enabled()}",
        flush=True,
    )

    if device_kind == "dml":
        # transformers.generate() is decorated with torch.inference_mode(),
        # which breaks DirectML ("Cannot set version_counter for inference tensor").
        class _NoInferenceMode:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __call__(self, fn=None):
                return self if fn is None else fn

        torch.inference_mode = _NoInferenceMode  # type: ignore[misc]

    pre_kw: dict = {"token": token}
    if MODEL_REVISION:
        pre_kw["revision"] = MODEL_REVISION
    processor = TrOCRProcessor.from_pretrained(MODEL_ID, **pre_kw)
    load_kw: dict = {"token": token}
    if MODEL_REVISION:
        load_kw["revision"] = MODEL_REVISION
    attn = attn_implementation()
    if attn and device_kind != "dml":
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
    exact = 0  # edit distance 0
    close = 0  # edit distance 1..CLOSE_MAX (near-match, not exact)
    len_ratios: list[float] = []
    short_scores: list[float] = []  # ref len <= 8
    long_scores: list[float] = []
    t0 = time.time()

    # DirectML breaks under inference_mode ("version_counter for inference tensor").
    _ctx = torch.no_grad if device_kind == "dml" else torch.inference_mode
    with _ctx():
        for i in range(n):
            ex = ds[i]
            image = to_rgb(ex["image"])
            ref = (ex.get("text") or ex.get("label") or "").strip()
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            if device_kind == "dml":
                # Avoid inference-tensor version_counter errors on DirectML.
                pv = pv.clone()
            ids = trocr_generate(
                model,
                processor,
                pv,
                image=image,
                num_beams=1,
            )
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            dist = edit_distance(pred, ref)
            score = 0.0 if not ref and not pred else (
                1.0 if not ref else dist / max(1, len(ref))
            )
            scores.append(score)
            if dist == 0:
                exact += 1
            elif dist <= CLOSE_MAX:
                close += 1
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
                near = exact + close
                print(
                    f"[PROG] {i+1}/{n} mean_CER={mean:.4f} "
                    f"exact={exact/(i+1):.2%} "
                    f"close(<={CLOSE_MAX})={close/(i+1):.2%} "
                    f"total_near={near/(i+1):.2%} "
                    f"{rate:.2f} samp/s ETA={eta/60:.1f}m",
                    flush=True,
                )

    mean = sum(scores) / max(1, len(scores))
    sorted_s = sorted(scores)
    p50 = sorted_s[len(sorted_s) // 2]
    p90 = sorted_s[int(len(sorted_s) * 0.9)]
    near = exact + close
    print("\n=== Summary ===", flush=True)
    print(f"[OK] n={n} mean CER={mean:.4f}", flush=True)
    print(f"[OK] exact-match (dist=0)={exact/n:.2%} ({exact}/{n})", flush=True)
    print(
        f"[OK] close-match (dist=1..{CLOSE_MAX})={close/n:.2%} ({close}/{n})",
        flush=True,
    )
    print(
        f"[OK] total near-match (exact+close, dist<={CLOSE_MAX})="
        f"{near/n:.2%} ({near}/{n})",
        flush=True,
    )
    print(f"[OK] CER p50={p50:.4f} p90={p90:.4f}", flush=True)
    if len_ratios:
        print(
            f"[OK] mean pred_len/ref_len={sum(len_ratios)/len(len_ratios):.3f}",
            flush=True,
        )
    if short_scores:
        print(
            f"[OK] short(<=8) n={len(short_scores)} mean CER={sum(short_scores)/len(short_scores):.4f}",
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
