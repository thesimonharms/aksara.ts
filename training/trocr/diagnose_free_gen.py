#!/usr/bin/env python3
"""Diagnose free-gen failure modes for a Hub TrOCR checkpoint."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import torch
from datasets import Image as HFImage, load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from device_utils import pick_device, resolve_torch_device
from generation_utils import anti_loop_enabled, trocr_generate
from local_verify_large import cer, edit_distance, to_rgb

MODEL = os.environ.get("HUB_MODEL_ID", "thesimonharms/trocr-javanese-synthetic-v6")
REV = os.environ.get("HUB_REVISION") or "main"
N = int(os.environ.get("N_SAMPLES", "300"))
OUT = Path(__file__).resolve().parent / f"local_verify_diagnose_{N}.json"


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device_kind = pick_device()
    device = resolve_torch_device(device_kind)
    print(
        f"[INFO] device={device_kind} model={MODEL}@{REV} n={N} "
        f"anti_loop={anti_loop_enabled()}",
        flush=True,
    )

    if device_kind == "dml":

        class _NoIM:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __call__(self, fn=None):
                return self if fn is None else fn

        torch.inference_mode = _NoIM  # type: ignore[misc]

    proc = TrOCRProcessor.from_pretrained(MODEL, token=token, revision=REV)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL, token=token, revision=REV)
    model.to(device)
    model.eval()
    cls = proc.tokenizer.cls_token_id
    model.config.decoder_start_token_id = cls
    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = cls
        model.generation_config.no_repeat_ngram_size = 0

    ds = load_dataset("thesimonharms/javanese-dataset", split="validation", token=token)
    ds = ds.cast_column("image", HFImage())

    rows: list[dict] = []
    _ctx = torch.no_grad if device_kind == "dml" else torch.inference_mode
    with _ctx():
        for i in range(min(N, len(ds))):
            ex = ds[i]
            ref = (ex.get("text") or "").strip()
            img = to_rgb(ex["image"])
            pv = proc(images=img, return_tensors="pt").pixel_values.to(device)
            if device_kind == "dml":
                pv = pv.clone()
            ids = trocr_generate(model, proc, pv, image=img, num_beams=1)
            pred = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            dist = edit_distance(pred, ref)
            score = cer(pred, ref)
            tags: list[str] = []
            if not pred:
                tags.append("empty")
            if ref and len(pred) > int(1.5 * len(ref)) + 2:
                tags.append("too_long")
            if ref and len(pred) < max(1, int(0.5 * len(ref))):
                tags.append("too_short")
            if score >= 1.0:
                tags.append("cer_ge_1")
            elif score >= 0.5:
                tags.append("cer_0.5_1")
            else:
                tags.append("cer_lt_0.5")
            if dist <= 2:
                tags.append("near")
            non_jv = [
                ch
                for ch in pred
                if not (0xA980 <= ord(ch) <= 0xA9DF or ch in " \t")
            ]
            if non_jv:
                tags.append("non_javanese")
            for k in (2, 3, 4):
                if len(pred) >= k * 3:
                    g = pred[:k]
                    if pred.count(g) >= 3:
                        tags.append(f"repeat_prefix_{k}")
                        break
            # Explicit cecak-wall tag
            if "ꦁꦁꦁ" in pred:
                tags.append("cecak_wall")
            rows.append(
                {
                    "i": i,
                    "ref": ref,
                    "pred": pred,
                    "cer": score,
                    "dist": dist,
                    "len_ratio": (len(pred) / len(ref) if ref else 0.0),
                    "tags": tags,
                    "ref_len": len(ref),
                }
            )
            if (i + 1) % 50 == 0:
                mean = sum(r["cer"] for r in rows) / len(rows)
                print(f"[PROG] {i+1}/{N} mean_CER={mean:.3f}", flush=True)

    n = len(rows)
    tag_c = Counter(t for r in rows for t in r["tags"])
    print(f"\n=== Failure tags (n={n}) ===", flush=True)
    for t, c in tag_c.most_common():
        print(f"  {t}: {c} ({c/n:.1%})", flush=True)

    print("\n=== CER buckets ===", flush=True)
    for a, b in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 2.0), (2.0, 99)]:
        c = sum(1 for r in rows if a <= r["cer"] < b)
        print(f"  [{a},{b}): {c} ({c/n:.1%})", flush=True)

    print("\n=== Worst 12 by CER ===", flush=True)
    for r in sorted(rows, key=lambda x: -x["cer"])[:12]:
        print(
            f"i={r['i']} CER={r['cer']:.2f} lenr={r['len_ratio']:.2f} tags={r['tags']}",
            flush=True,
        )
        print(f"  REF : {r['ref'][:100]}", flush=True)
        print(f"  PRED: {r['pred'][:100]}", flush=True)

    print("\n=== Best 8 ===", flush=True)
    for r in sorted(rows, key=lambda x: x["cer"])[:8]:
        print(f"i={r['i']} CER={r['cer']:.2f} dist={r['dist']}", flush=True)
        print(f"  REF : {r['ref'][:100]}", flush=True)
        print(f"  PRED: {r['pred'][:100]}", flush=True)

    OUT.write_text(
        json.dumps({"n": n, "tag_counts": dict(tag_c), "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
