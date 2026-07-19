#!/usr/bin/env python3
"""Minimal TrOCR train-step probe for Intel Arc XPU workarounds."""

from __future__ import annotations

import argparse
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


def to_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return Image.open(img["path"]).convert("RGB")
    raise TypeError(type(img))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa", "default"])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off")
    p.add_argument("--device", default="xpu")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    print(f"[probe] torch={torch.__version__} device={args.device} attn={args.attn} amp={args.amp}", flush=True)
    if args.device == "xpu":
        print(f"[probe] xpu={torch.xpu.is_available()} name={torch.xpu.get_device_name(0)}", flush=True)

    load_kw = {"token": token}
    if args.attn != "default":
        load_kw["attn_implementation"] = args.attn

    proc = TrOCRProcessor.from_pretrained("thesimonharms/trocr-javanese-synthetic-v2", token=token)
    model = VisionEncoderDecoderModel.from_pretrained(
        "thesimonharms/trocr-javanese-synthetic-v2", **load_kw
    )
    model.to(args.device)
    model.train()
    print(f"[probe] param_device={next(model.parameters()).device}", flush=True)

    ds = load_dataset(
        "thesimonharms/javanese-dataset-180k",
        split=f"train[:{args.batch}]",
        token=token,
    ).cast_column("image", HFImage())
    imgs = [to_rgb(ex["image"]) for ex in ds]
    texts = [(ex.get("text") or "").strip() for ex in ds]
    pv = proc(images=imgs, return_tensors="pt").pixel_values.to(args.device)
    labels = proc.tokenizer(texts, padding=True, return_tensors="pt").input_ids.to(args.device)
    labels[labels == proc.tokenizer.pad_token_id] = -100

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    dtype = {"off": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]

    for step in range(args.steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        if dtype is None:
            out = model(pixel_values=pv, labels=labels)
            loss = out.loss
        else:
            with torch.amp.autocast(args.device, dtype=dtype):
                out = model(pixel_values=pv, labels=labels)
                loss = out.loss
        print(f"[probe] step={step} forward loss={float(loss.detach()):.4f} t={time.time()-t0:.2f}s", flush=True)
        t1 = time.time()
        loss.backward()
        print(f"[probe] step={step} backward ok t={time.time()-t1:.2f}s", flush=True)
        opt.step()
        if args.device == "xpu":
            torch.xpu.synchronize()
        print(f"[probe] step={step} FULL ok total={time.time()-t0:.2f}s", flush=True)

    print("[probe] SUCCESS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[probe] FAIL {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
