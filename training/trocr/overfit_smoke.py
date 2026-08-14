#!/usr/bin/env python3
"""Tiny overfit: if this cannot drive loss << 1 on 8 lines, the train stack is broken."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from datasets import Image as HFImage, load_dataset
from torch.optim import AdamW
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from device_utils import attn_implementation, pick_device
from finetune_trocr import TrocrDataCollator, _expand_javanese_tokenizer, _patch_ved_unshifted_loss
from generation_utils import trocr_generate
from image_prep import ink_bbox, pad_to_square
from local_verify_large import to_rgb

N = int(os.environ.get("N", "8"))
STEPS = int(os.environ.get("STEPS", "200"))
LR = float(os.environ.get("LR", "1e-4"))
BASE = os.environ.get("BASE_MODEL", "microsoft/trocr-small-printed")
DATA = os.environ.get("DATASET_NAME", "thesimonharms/javanese-synthetic-exact")


def main() -> None:
    device = pick_device()
    print(f"[smoke] device={device} attn={attn_implementation()} n={N} steps={STEPS} lr={LR}", flush=True)
    print(f"[smoke] TROCR_FORCE_FP32={os.environ.get('TROCR_FORCE_FP32')} TROCR_ATTN={os.environ.get('TROCR_ATTN')}", flush=True)

    kw = {}
    attn = attn_implementation()
    if attn:
        kw["attn_implementation"] = attn
    processor = TrOCRProcessor.from_pretrained(BASE)
    model = VisionEncoderDecoderModel.from_pretrained(BASE, **kw)
    print(
        f"[smoke] pretrained decoder_start={model.config.decoder_start_token_id} "
        f"bos={model.config.bos_token_id} eos={model.config.eos_token_id} "
        f"pad={model.config.pad_token_id}",
        flush=True,
    )
    _expand_javanese_tokenizer(processor, model)
    tok = processor.tokenizer
    model.config.decoder_start_token_id = tok.cls_token_id
    model.config.bos_token_id = tok.cls_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id
    _patch_ved_unshifted_loss(model)
    print(
        f"[smoke] img_size={getattr(processor.image_processor, 'size', None)} "
        f"decoder_start={model.config.decoder_start_token_id} vocab={len(tok)}",
        flush=True,
    )

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(DATA, split="train", token=token)
    ds = ds.cast_column("image", HFImage()).shuffle(seed=42).select(range(N))
    rows = []
    for i in range(N):
        ex = ds[i]
        img = pad_to_square(to_rgb(ex["image"]))
        box = ink_bbox(img)
        print(f"[smoke] {i} size={img.size} ink={box} text={ex['text']!r}", flush=True)
        rows.append({"image": img, "text": ex["text"]})

    collator = TrocrDataCollator(processor, 24, pad_square=False)
    batch = collator(rows)
    pv = batch["pixel_values"]
    labels = batch["labels"]
    print(
        f"[smoke] pixel_values shape={tuple(pv.shape)} mean={pv.mean():.4f} std={pv.std():.4f} "
        f"min={pv.min():.3f} max={pv.max():.3f}",
        flush=True,
    )
    print(f"[smoke] labels[0]={[x for x in labels[0].tolist() if x != -100]}", flush=True)
    # Distinctness: blank images would have near-identical pixel means.
    means = [float(pv[i].mean()) for i in range(pv.size(0))]
    print(f"[smoke] per-image pixel means={['%.4f' % m for m in means]}", flush=True)

    model.to(device)
    model.train()
    pv = pv.to(device)
    labels = labels.to(device)
    opt = AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)

    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        out = model(pixel_values=pv, labels=labels)
        loss = out.loss
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 20 == 0 or step == STEPS:
            print(f"[smoke] step={step:04d} loss={loss.item():.4f} grad_norm={float(gn):.3f}", flush=True)

    model.eval()
    exact = 0
    with torch.inference_mode():
        for i, row in enumerate(rows):
            image = row["image"]
            ref = row["text"]
            pix = processor(images=image, return_tensors="pt").pixel_values.to(device)
            gold = labels[i : i + 1]
            tf = model(pixel_values=pix, labels=gold)
            ids = trocr_generate(
                model,
                processor,
                pix,
                image=image,
                num_beams=1,
                anti_loop=False,
                max_new_tokens=24,
            )
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            hit = pred == ref
            exact += int(hit)
            gold_row = gold[0]
            pred_tf = tf.logits.argmax(-1)[0]
            mask = gold_row != -100
            tf_txt = processor.tokenizer.decode(
                pred_tf[mask].tolist(), skip_special_tokens=True
            )
            print(
                f"[smoke] tf_loss={tf.loss.item():.4f} tf_arg={tf_txt!r} "
                f"match={hit} ref={ref!r} pred={pred!r}",
                flush=True,
            )
    print(f"[smoke] exact={exact}/{N}", flush=True)
    if exact < N:
        sys.exit(3)


if __name__ == "__main__":
    main()
