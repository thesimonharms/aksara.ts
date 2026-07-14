#!/usr/bin/env python3
"""
finetune_trocr.py — Fine-tune microsoft/trocr-base-handwritten on Javanese Aksara
line images via transformers' Seq2SeqTrainer.

Why Seq2SeqTrainer, not AutoTrain:
  AutoTrain Advanced is deprecated, and it never had a first-class task for
  VisionEncoderDecoder (TrOCR) training. The correct, supported path is the
  Seq2SeqTrainer from `transformers`, which is what this script uses.

RBAC / secrets:
  - Locally: load environment variables from ../../.env (gitignored).
  - In a HF Space: HF_TOKEN / HF_USERNAME come from Space Secrets -> env vars.
  Both paths use the same `os.environ` lookups; load_dotenv() is a no-op when
  no .env is present, so Space secrets take precedence there.

Push behavior:
  - The fine-tuned model is pushed to {HF_USERNAME}/javanese-trocr-handwritten
    by default (toggle off with --no_push).
  - Use --push_dataset to also publish the generated imagefolder dataset to
    {HF_USERNAME}/javanese-dataset so future runs (or other Spaces) can load
    it directly via --dataset_name.

CLI examples:
  # Local CPU/DirectML (slow): load local trocr_dataset, no hub push
  python finetune_trocr.py --no_push
  # Production: train on HF Hub dataset, publish model
  python finetune_trocr.py --dataset_name <user>/javanese-dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 1. Load **local** .env before importing HF libs that read HF_TOKEN.
#    In a HF Space this file is absent (gitignored), so the no-op is fine:
#    Space Secrets populate the environment for us.
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(_ENV_PATH)
except ImportError:
    pass

import torch
from PIL import Image
from datasets import DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

import evaluate


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"[ERROR] Required env var '{name}' is unset. "
            f"Set it in ../../.env locally, or as a HF Space Secret."
        )
    return value


@dataclass
class TrainConfig:
    base_model: str = "microsoft/trocr-base-handwritten"
    dataset_dir: Optional[Path] = None        # local imagefolder root (train/ + validation/ subdirs)
    dataset_name: Optional[str] = None       # HF Hub dataset id — overrides dataset_dir
    output_dir: Path = Path("./trocr_ckpt")
    epochs: int = 5
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 4e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_target_length: int = 64
    hub_model_id: Optional[str] = None
    push_to_hub: bool = True
    push_dataset: bool = False
    seed: int = 42
    num_beams: int = 4
    pdf_labeled_dir: Optional[Path] = None  # real-handwriting labeled pairs from label_pdfs.py
    upsample_labeled: float = 1.0           # multiply labeled pairs so they're not drowned out


def _load_pdf_labeled(pdf_dir: Path, upsample: float = 1.0):
    """Load real-handwriting (image, text) pairs produced by label_pdfs.py.

    Expects label_XXXXXX.{png,jpg,jpeg} + label_XXXXXX.txt pairs. Returns a datasets.Dataset
    with `image` (PIL.Image) and `text` (str) columns, or None if the dir is
    empty / missing.
    """
    from datasets import Dataset, Image as HFImage
    if not pdf_dir or not pdf_dir.exists():
        return None
    exts = ("label_*.png", "label_*.PNG", "label_*.jpg", "label_*.JPG", "label_*.jpeg", "label_*.JPEG")
    image_files = sorted({p.resolve() for ext in exts for p in pdf_dir.glob(ext)})
    if not image_files:
        print(f"[INFO] No label_*.png/jpg pairs found in {pdf_dir}")
        return None
    images, texts = [], []
    missing = 0
    for img_path in image_files:
        txt = img_path.with_suffix(".txt")
        if not txt.exists():
            missing += 1
            continue
        try:
            label = txt.read_text(encoding="utf-8").strip()
            if not label:
                continue
            img = Image.open(img_path).convert("RGB")  # noqa: F821 — PIL imported at module top
            images.append(img)
            texts.append(label)
        except Exception as exc:
            print(f"[WARN] couldn't read {img_path}: {exc}")
    if missing:
        print(f"[WARN] {missing} image files had no matching .txt — skipped")
    if not texts:
        return None
    # Upsample so the ~hundreds of real-handwriting pairs aren't drowned out
    # by the ~thousands of synthetic samples (>=1.0 means duplicate by that factor).
    n = int(round(upsample))
    images = images * max(1, n)
    texts  = texts  * max(1, n)
    print(f"[INFO] Loaded {len(images)} labeled real-handwriting pairs from {pdf_dir}"
          f" (upsample ×{n})")
    return Dataset.from_dict({"image": images, "text": texts}).cast_column("image", HFImage())


_load_labeled_pairs = _load_pdf_labeled

def build_dataset(config: TrainConfig, processor: TrOCRProcessor, hf_token: Optional[str]) -> tuple:
    """Return (raw_loadable, model_ready) — raw is push-able, model_ready is processed."""

    if config.dataset_name:
        print(f"[INFO] Loading dataset from HF Hub: {config.dataset_name}")
        raw = load_dataset(config.dataset_name, token=hf_token)
        if "validation" not in raw:
            raw = raw["train"].train_test_split(test_size=0.1, seed=config.seed)
    else:
        ddir = config.dataset_dir or Path(__file__).resolve().parent / "trocr_dataset"
        if not ddir.exists():
            sys.exit(
                f"[ERROR] Local dataset directory not found: {ddir}\n"
                f"        Run:  python generate_trocr_dataset.py --fonts_dir ../fonts --pdfs_dir ../pdfs\n"
                f"        Or use --dataset_name <hf-hub-id> to load from the Hub."
            )
        print(f"[INFO] Loading local imagefolder dataset from: {ddir}")
        train = load_dataset("imagefolder", data_dir=str(ddir / "train"), split="train")
        val = load_dataset("imagefolder", data_dir=str(ddir / "validation"), split="train")
        raw = DatasetDict({"train": train, "validation": val})

    # Optionally add real-handwriting pairs from label_pdfs.py. These are the
    # highest-signal samples in the mix (real ink, real paper, real glyph
    # shapes) and dramatically curb the synthetic→real domain gap.
    if config.pdf_labeled_dir:
        labeled = _load_pdf_labeled(config.pdf_labeled_dir, config.upsample_labeled)
        if labeled is not None:
            from datasets import concatenate_datasets
            raw["train"] = concatenate_datasets([raw["train"], labeled])
            print(f"[INFO] Combined training set: {len(raw['train'])} examples "
                  f"(synthetic + labeled real-handwriting)")


    # Pre-tokenize once. The processor handles image -> pixel_values AND text -> labels,
    # padding both to max_length so default_data_collator can stack batches cleanly.
    max_len = config.max_target_length
    pad_id = processor.tokenizer.pad_token_id

    def preprocess(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        enc = processor(
            images=images,
            text=batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        # Trainer convention: mask pad tokens in labels with -100 so CTC/CE ignores them.
        labels = enc["labels"].clone()
        labels[labels == pad_id] = -100
        return {"pixel_values": enc["pixel_values"], "labels": labels}

    tokenized = raw.map(
        preprocess,
        remove_columns=raw["train"].column_names,
        batched=True,
        batch_size=config.batch_size,
        desc="Preprocessing",
    )
    return raw, tokenized


def push_dataset_to_hub(raw, config: TrainConfig, hf_token: str) -> str:
    target = config.dataset_name or f"{_require_env('HF_USERNAME')}/javanese-dataset"
    print(f"[INFO] Pushing dataset to HF Hub: {target}")
    raw.push_to_hub(target, token=hf_token, private=False)
    return target


def train(config: TrainConfig) -> str:
    hf_token = os.environ.get("HF_TOKEN")
    hf_user = os.environ.get("HF_USERNAME")

    if config.push_to_hub and not (hf_token and hf_user):
        print("[WARN] HF_TOKEN / HF_USERNAME not set — will train locally without Hub push.")
        config.push_to_hub = False

    print(f"[INFO] Base model: {config.base_model}")
    processor = TrOCRProcessor.from_pretrained(config.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(config.base_model)

    # Decoder config — the base is pre-trained on English, these are already set
    # correctly on the checkpoint, but we enforce defensively so a fresh load
    # can't silently regress (e.g. if decoder_start_token_id unset -> bogus loss).
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = config.max_target_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = config.num_beams

    raw, tokenized = build_dataset(config, processor, hf_token)

    if config.push_dataset and hf_token:
        push_dataset_to_hub(raw, config, hf_token)

    cer_metric = evaluate.load("cer")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = labels.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        return {"cer": cer_metric.compute(predictions=pred_str, references=label_str)}

    # fp16 only on CUDA; on CPU / DirectML we stay in fp32 (Triton-less autocast).
    use_fp16 = torch.cuda.is_available()

    hub_kwargs = {}
    if config.push_to_hub and hf_token:
        hub_id = config.hub_model_id or f"{hf_user}/javanese-trocr-handwritten"
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": hub_id,
            "hub_token": hf_token,
        }
        print(f"[INFO] Trained model will be pushed to: {hub_id}")

    args = Seq2SeqTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        seed=config.seed,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.num_beams,
        fp16=use_fp16,
        report_to="none",
        **hub_kwargs,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    print("[INFO] Starting fine-tuning...")
    trainer.train()

    final_dir = config.output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[OK] Model + processor saved to {final_dir.resolve()}")

    if config.push_to_hub and hf_token:
        trainer.push_to_hub()
        print("[OK] Model pushed to HF Hub")

    return str(final_dir.resolve())


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Fine-tune TrOCR for Javanese Aksara OCR.")
    p.add_argument("--base_model", default="microsoft/trocr-base-handwritten")
    p.add_argument("--dataset_dir", type=Path, default=None,
                   help="Local imagefolder root (must contain train/ and validation/).")
    p.add_argument("--dataset_name", default=None,
                   help="HF Hub dataset id; overrides --dataset_dir.")
    p.add_argument("--output_dir", type=Path, default=Path("./trocr_ckpt"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--max_target_length", type=int, default=64)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--hub_model_id", default=None)
    p.add_argument("--no_push", action="store_true", help="Don't push model to HF Hub.")
    p.add_argument("--push_dataset", action="store_true", help="Push dataset to HF Hub before training.")
    p.add_argument("--pdf_labeled_dir", "--labeled_dir", dest="pdf_labeled_dir",
                   type=Path, default=Path("../pdf_labeled"),
                   help="Dir of label_XXXX.{png,jpg} + label_XXXX.txt real-handwriting pairs "
                        "from label_pdfs.py. Set to '' to skip.")
    p.add_argument("--upsample_labeled", type=int, default=1,
                   help="Duplicate labeled pairs by this factor (e.g. 5 so ~hundreds "
                        "aren't drowned by thousands of synthetic samples).")
    a = p.parse_args()

    if a.dataset_dir is None and a.dataset_name is None:
        a.dataset_dir = Path(__file__).resolve().parent / "trocr_dataset"

    return TrainConfig(
        base_model=a.base_model,
        dataset_dir=a.dataset_dir,
        dataset_name=a.dataset_name,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        eval_batch_size=a.eval_batch_size,
        learning_rate=a.lr,
        max_target_length=a.max_target_length,
        hub_model_id=a.hub_model_id,
        push_to_hub=not a.no_push,
        push_dataset=a.push_dataset,
        num_beams=a.num_beams,
        pdf_labeled_dir=a.pdf_labeled_dir if a.pdf_labeled_dir and str(a.pdf_labeled_dir) else None,
        upsample_labeled=float(max(1, int(a.upsample_labeled))),
    )


def run_pipeline(overrides: Optional[dict] = None) -> str:
    """Entry point for the HF Space app — accepts a dict of TrainConfig overrides."""
    cfg = TrainConfig()
    if overrides:
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
    return train(cfg)


if __name__ == "__main__":
    train(parse_args())