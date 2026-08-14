#!/usr/bin/env python3
"""Diagnose failed v6 Phase 0 overfit (run inside NAS train container or with local ckpt)."""
from __future__ import annotations

import json
import os
from pathlib import Path

CKPT = Path(os.environ.get("CKPT", "/workspace/output/trocr_v6_stage_0/final"))
N = int(os.environ.get("N", "8"))


def main() -> None:
    ts = CKPT / "trainer_state.json"
    if not ts.exists():
        parent = CKPT.parent
        cands = sorted(parent.glob("checkpoint-*/trainer_state.json"))
        print("no trainer_state in final; checkpoints:", [str(p) for p in cands])
        if cands:
            ts = cands[-1]
    if ts.exists():
        d = json.loads(ts.read_text(encoding="utf-8"))
        print("=== loss history ===")
        for e in d.get("log_history", []):
            row = {k: e[k] for k in ("epoch", "step", "loss", "eval_loss", "learning_rate") if k in e}
            if row:
                print(row)

    cfg = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
    gen = {}
    gp = CKPT / "generation_config.json"
    if gp.exists():
        gen = json.loads(gp.read_text(encoding="utf-8"))
    print("=== model ===")
    print("architectures", cfg.get("architectures"))
    print("encoder", (cfg.get("encoder") or {}).get("model_type"), "decoder", (cfg.get("decoder") or {}).get("model_type"))
    print("vocab", cfg.get("vocab_size"), "dec_vocab", (cfg.get("decoder") or {}).get("vocab_size"))
    print(
        "ids cfg",
        {k: cfg.get(k) for k in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id")},
    )
    print(
        "ids gen",
        {k: gen.get(k) for k in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id", "max_length")},
    )

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(str(CKPT))
    tok = processor.tokenizer
    print("=== tokenizer ===")
    print("class", type(tok).__name__, "len", len(tok))
    print(
        "cls", tok.cls_token, tok.cls_token_id,
        "bos", tok.bos_token, tok.bos_token_id,
        "eos", tok.eos_token, tok.eos_token_id,
        "pad", tok.pad_token, tok.pad_token_id,
        "sep", tok.sep_token, tok.sep_token_id,
    )
    sample = "ꦲꦤꦕꦫꦏ"
    ids = tok(sample, add_special_tokens=False).input_ids
    ids_sp = tok(sample, add_special_tokens=True).input_ids
    print("no_special", ids, "decode", tok.decode(ids))
    print("with_special", ids_sp, "decode", tok.decode(ids_sp, skip_special_tokens=True))
    proc_lab = processor(text=[sample], padding="max_length", truncation=True, max_length=24, return_tensors="pt")
    lab = proc_lab["labels"][0].tolist()
    print("processor labels", lab)
    print("processor decode", tok.decode([i for i in lab if i != tok.pad_token_id], skip_special_tokens=True))

    # How many Javanese chars are truly 1-token?
    jav = [chr(c) for c in range(0xA980, 0xA9E0)]
    bad = []
    for ch in jav:
        i = tok(ch, add_special_tokens=False).input_ids
        if len(i) != 1 or tok.decode(i) != ch:
            bad.append((ch, i, tok.decode(i)))
    print(f"javanese chars not atomic: {len(bad)}/{len(jav)}")
    print("examples", bad[:8])

    import torch
    from datasets import Image as HFImage, load_dataset
    from image_prep import pad_to_square
    from local_verify_large import to_rgb
    from generation_utils import trocr_generate

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(os.environ.get("DATASET_NAME", "thesimonharms/javanese-synthetic-exact"), split="train", token=token)
    if "image" in ds.column_names:
        ds = ds.cast_column("image", HFImage())
    ds = ds.shuffle(seed=42).select(range(256))

    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    model = VisionEncoderDecoderModel.from_pretrained(str(CKPT))
    model.to(device)
    model.eval()
    print("=== samples ===")
    with torch.inference_mode():
        for i in range(N):
            ex = ds[i]
            image = pad_to_square(to_rgb(ex["image"]))
            ref = (ex.get("text") or "").strip()
            pv = processor(images=image, return_tensors="pt").pixel_values.to(device)
            ids_g = trocr_generate(model, processor, pv, image=image, num_beams=1)
            pred = processor.batch_decode(ids_g, skip_special_tokens=True)[0].strip()
            raw_ids = ids_g[0].tolist()
            print(f"-- {i} --")
            print("ref ", ref)
            print("pred", pred)
            print("gen_ids", raw_ids[:40], "len", len(raw_ids))
            print("match", pred == ref)


if __name__ == "__main__":
    main()
