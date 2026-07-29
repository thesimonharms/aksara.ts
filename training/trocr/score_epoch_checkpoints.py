#!/usr/bin/env python3
"""Score every local Trainer checkpoint on original-val (CER + exact + near-match).

Runs on whatever device pick_device() finds (NAS iGPU). Writes CSV locally and
optionally uploads to the Hub model repo under evals/scores.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
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
from local_verify_large import CLOSE_MAX, edit_distance, to_rgb


def score_checkpoint(
    model_path: str,
    ds,
    n: int,
    device: str,
    token: str | None,
) -> dict:
    processor = TrOCRProcessor.from_pretrained(model_path, token=token)
    load_kw: dict = {"token": token} if token else {}
    attn = attn_implementation()
    if attn:
        load_kw["attn_implementation"] = attn
    model = VisionEncoderDecoderModel.from_pretrained(model_path, **load_kw)
    model.to(device)
    model.eval()
    cls_id = processor.tokenizer.cls_token_id
    model.config.decoder_start_token_id = cls_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = cls_id
        model.generation_config.no_repeat_ngram_size = 0

    scores: list[float] = []
    exact = close = 0
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
            dist = edit_distance(pred, ref)
            score = 0.0 if not ref and not pred else (
                1.0 if not ref else dist / max(1, len(ref))
            )
            scores.append(score)
            if dist == 0:
                exact += 1
            elif dist <= CLOSE_MAX:
                close += 1
            if (i + 1) % 100 == 0 or i + 1 == n:
                mean = sum(scores) / len(scores)
                print(
                    f"  [{Path(model_path).name}] {i+1}/{n} "
                    f"CER={mean:.4f} exact={exact/(i+1):.2%} "
                    f"near={(exact+close)/(i+1):.2%}",
                    flush=True,
                )

    mean = sum(scores) / max(1, len(scores))
    near = exact + close
    del model
    if device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return {
        "checkpoint": Path(model_path).name,
        "path": str(model_path),
        "n": n,
        "mean_cer": round(mean, 6),
        "exact": exact,
        "exact_rate": round(exact / n, 6),
        "close": close,
        "close_rate": round(close / n, 6),
        "near": near,
        "near_rate": round(near / n, 6),
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root", type=Path, required=True,
                   help="Dir containing checkpoint-* folders (and optional final/).")
    p.add_argument("--dataset_name", default="thesimonharms/javanese-dataset")
    p.add_argument("--split", default="validation")
    p.add_argument("--n_samples", type=int, default=1500)
    p.add_argument("--out_csv", type=Path, required=True)
    p.add_argument("--hub_model_id", default=None,
                   help="If set, upload CSV to Hub at evals/scores.csv")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = pick_device()
    print(f"[INFO] device={device} ckpt_root={args.ckpt_root}", flush=True)

    ckpts = sorted(args.ckpt_root.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    final = args.ckpt_root / "final"
    if final.is_dir():
        ckpts.append(final)
    if not ckpts:
        raise SystemExit(f"No checkpoints under {args.ckpt_root}")

    ds = load_dataset(args.dataset_name, split=args.split, token=token)
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage())
    n = min(args.n_samples, len(ds))
    print(f"[INFO] Scoring {len(ckpts)} checkpoints × {n} samples", flush=True)

    rows = []
    for ck in ckpts:
        print(f"[INFO] === {ck.name} ===", flush=True)
        try:
            rows.append(score_checkpoint(str(ck), ds, n, device, token))
        except Exception as exc:
            print(f"[WARN] failed {ck}: {exc}", flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {args.out_csv}", flush=True)

    if args.hub_model_id and token and rows:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(args.out_csv),
            path_in_repo="evals/scores.csv",
            repo_id=args.hub_model_id,
            repo_type="model",
            commit_message="Add per-epoch CER / near-match scoreboard",
        )
        print(f"[OK] uploaded evals/scores.csv → {args.hub_model_id}", flush=True)


if __name__ == "__main__":
    main()
