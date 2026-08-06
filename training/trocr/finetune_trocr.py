#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=4.46,<5.0",
#     "datasets>=2.20",
#     "accelerate>=0.34",
#     "evaluate>=0.4",
#     "jiwer>=3.0",
#     "sentencepiece>=0.2.0",
#     "python-dotenv>=1.0",
#     "pillow>=10.0",
#     "torch",
# ]
# ///
"""
finetune_trocr.py — Fine-tune microsoft/trocr-base-handwritten on Javanese Aksara
line images via transformers' Seq2SeqTrainer.

Why Seq2SeqTrainer, not AutoTrain:
  AutoTrain Advanced is deprecated, and it never had a first-class task for
  VisionEncoderDecoder (TrOCR) training. The correct, supported path is the
  Seq2SeqTrainer from `transformers`, which is what this script uses.

RBAC / secrets:
  - Locally: load environment variables from ../../.env (gitignored).
  - HF Jobs / Space: HF_TOKEN from --secrets / Space Secrets → os.environ.
  - Username: HF_USERNAME, else SPACE_AUTHOR_NAME, else huggingface_hub.whoami().

Push behavior:
  - Checkpoints push to the Hub on every save (epoch) when push is enabled, so a
    timed-out Job still leaves a usable revision.
  - Final model is also pushed explicitly after training.
  - Use --push_dataset to publish the imagefolder dataset to the Hub.

CLI examples:
  # Local smoke (no Hub):
  python finetune_trocr.py --no_push --epochs 1 --max_train_samples 200
  # HF Jobs / production:
  python finetune_trocr.py --dataset_name <user>/javanese-dataset \\
      --hub_model_id <user>/trocr-javanese-synthetic --epochs 5 --batch_size 24
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 1. Load **local** .env before importing HF libs that read HF_TOKEN.
#    Monorepo layout: training/trocr/finetune_trocr.py → ../../.env (repo root).
#    HF Jobs / Space flat layout: no parents[2] .env — secrets already in env.
try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    _file = Path(__file__).resolve()
    if len(_file.parents) > 2:
        _env_path = _file.parents[2] / ".env"
        if _env_path.is_file():
            load_dotenv(_env_path)

import torch
from PIL import Image
from datasets import DatasetDict, Image as HFImage, load_dataset
from transformers import (
    TrainerCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# Prefer sibling device_utils locally; keep inline fallback for HF Jobs (single-file stage).
try:
    from device_utils import (
        attn_implementation,
        dataloader_kwargs,
        log_device,
        pick_device,
        use_amp,
    )
except ImportError:  # pragma: no cover — Jobs flat layout

    def log_device() -> str:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] device={device}")
        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        return device

    def use_amp(device: str | None = None) -> tuple[bool, bool]:
        return (torch.cuda.is_available(), False)

    def attn_implementation(device: str | None = None) -> str | None:
        return None

    def dataloader_kwargs(device: str | None = None) -> dict:
        cuda = torch.cuda.is_available()
        return {
            "dataloader_num_workers": 2 if cuda else 0,
            "dataloader_pin_memory": bool(cuda),
        }

    def pick_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

try:
    from transformers_xpu_helper import ultra_255h_config
    from transformers_xpu_helper.trainer import build_seq2seq_training_arguments

    _HAS_XPU_HELPER = True
except ImportError:  # pragma: no cover
    _HAS_XPU_HELPER = False
    ultra_255h_config = None  # type: ignore
    build_seq2seq_training_arguments = None  # type: ignore

# Javanese block — whole characters added to the BPE vocab so free-run generation
# does not scramble UTF-8 byte tokens (see expand_javanese_tokenizer.py).
_JAVANESE_BLOCK = [chr(c) for c in range(0xA980, 0xA9E0)]


def _collect_javanese_chars(corpus_path: Optional[Path] = None) -> list[str]:
    chars: set[str] = set(_JAVANESE_BLOCK)
    if corpus_path is not None and corpus_path.is_file():
        for ch in corpus_path.read_text(encoding="utf-8"):
            if 0xA980 <= ord(ch) <= 0xA9FF:
                chars.add(ch)
    return sorted(chars, key=ord)


def _expand_javanese_tokenizer(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    corpus_path: Optional[Path] = None,
) -> int:
    """Add atomic Javanese chars; resize decoder embeddings. Returns #tokens added."""
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    to_add: list[str] = []
    for ch in _collect_javanese_chars(corpus_path):
        if ch in vocab:
            continue
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        if len(ids) == 1 and tokenizer.decode(ids) == ch:
            continue
        to_add.append(ch)
    if not to_add:
        print("[INFO] Tokenizer already covers Javanese characters")
        return 0
    n = tokenizer.add_tokens(to_add)
    new_size = len(tokenizer)
    old = model.decoder.get_input_embeddings().weight.shape[0]
    # VisionEncoderDecoderModel does not implement set_input_embeddings; resize decoder.
    model.decoder.resize_token_embeddings(new_size)
    model.config.vocab_size = new_size
    if getattr(model.config, "decoder", None) is not None:
        model.config.decoder.vocab_size = new_size
    print(f"[INFO] Added {n} Javanese tokens; decoder embeddings {old} → {new_size}")
    sample = "ꦲꦤꦕꦫꦏ"
    ids = tokenizer(sample, add_special_tokens=False).input_ids
    print(f"[INFO] Token check {sample!r} → {len(ids)} tokens (want {len(sample)})")
    if len(ids) != len(sample):
        print("[WARN] Sample still multi-token — encoding may not use added chars yet")
    return int(n)

import evaluate


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"[ERROR] Required env var '{name}' is unset. "
            f"Set it in ../../.env locally, pass --secrets HF_TOKEN on HF Jobs, "
            f"or set a Space Secret."
        )
    return value


def _hf_token() -> Optional[str]:
    """Write token for Hub push. Prefer HF_TOKEN; accept HUGGING_FACE_HUB_TOKEN."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _hf_username() -> Optional[str]:
    """Hub user/org: HF_USERNAME → SPACE_AUTHOR_NAME → whoami(token)."""
    explicit = os.environ.get("HF_USERNAME") or os.environ.get("SPACE_AUTHOR_NAME")
    if explicit:
        return explicit
    token = _hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import whoami

        info = whoami(token=token)
        return info.get("name") or info.get("fullname")
    except Exception as exc:
        print(f"[WARN] whoami() failed while resolving username: {exc}")
        return None


@dataclass
class TrainConfig:
    base_model: str = "microsoft/trocr-base-handwritten"
    dataset_dir: Optional[Path] = None        # local imagefolder root (train/ + validation/)
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
    pdf_labeled_dir: Optional[Path] = None
    upsample_labeled: float = 1.0
    # Jobs-friendly reliability / wall-clock knobs
    eval_every_epochs: int = 2          # run eval every N epochs (1 = every epoch)
    predict_with_generate: bool = False # CER generation is expensive; default off mid-train
    max_steps: Optional[int] = None     # smoke Jobs: cap optimizer steps
    max_train_samples: Optional[int] = None
    # True: save VRAM (slower). False: faster when batch fits in GPU memory.
    gradient_checkpointing: bool = True
    # Add whole Javanese chars to the BPE vocab (needed for usable free-run OCR).
    expand_javanese_tokenizer: bool = True
    tokenizer_dir: Optional[Path] = None  # optional pre-expanded processor dir
    javanese_corpus: Optional[Path] = None
    # Length curriculum: oversample short lines so free-run stops "always emitting a phrase".
    short_line_max_chars: int = 8
    short_line_fraction: float = 0.0  # 0 = off; 0.25 ≈ 25% of train ≤ short_line_max_chars
    # Extra Hub dataset(s) mixed into train (+ optional val).
    # Comma-separated names/upsamples supported, e.g. name="a,b" upsample="1,8".
    extra_dataset_name: Optional[str] = None
    extra_dataset_upsample: int | str = 1
    skip_final_cer: bool = False
    resume_from_checkpoint: bool | str = False
    early_stopping_patience: int = 0  # 0 = disabled; NAS hands-off uses 3+
    load_best_model_at_end: bool = False
    freeze_encoder: bool = False
    # None = keep every epoch checkpoint on disk (needed for post-train scoring).
    save_total_limit: Optional[int] = 2
    hub_tag_epochs: bool = True  # tag Hub tip as epoch-N after each save


class HubEpochTagCallback(TrainerCallback):
    """After each epoch save/push, tag the Hub tip as epoch-N (root stays latest only)."""

    def __init__(self, repo_id: str, token: str | None):
        self.repo_id = repo_id
        self.token = token

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.repo_id or not self.token:
            return
        if state.epoch is None:
            return
        epoch_i = int(round(float(state.epoch)))
        if epoch_i < 1:
            return
        tag = f"epoch-{epoch_i}"
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            # Move tag if this epoch is re-saved (resume).
            try:
                api.delete_tag(self.repo_id, tag=tag, repo_type="model")
            except Exception:
                pass
            api.create_tag(
                self.repo_id,
                tag=tag,
                repo_type="model",
                tag_message=f"End of training epoch {epoch_i}",
            )
            print(f"[OK] Hub tag {tag} → {self.repo_id}", flush=True)
        except Exception as exc:
            print(f"[WARN] Hub tag {tag} failed: {exc}", flush=True)


def _freeze_vision_encoder(model: VisionEncoderDecoderModel) -> None:
    enc = getattr(model, "encoder", None)
    if enc is None:
        print("[WARN] model has no .encoder — freeze skipped", flush=True)
        return
    n = 0
    for p in enc.parameters():
        p.requires_grad = False
        n += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[INFO] Froze vision encoder ({n} tensors). "
        f"Trainable params: {trainable:,} / {total:,}",
        flush=True,
    )


def _load_pdf_labeled(pdf_dir: Path, upsample: float = 1.0):
    """Load real-handwriting (image, text) pairs produced by label_pdfs.py."""
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
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            texts.append(label)
        except Exception as exc:
            print(f"[WARN] couldn't read {img_path}: {exc}")
    if missing:
        print(f"[WARN] {missing} image files had no matching .txt — skipped")
    if not texts:
        return None
    n = int(round(upsample))
    images = images * max(1, n)
    texts = texts * max(1, n)
    print(
        f"[INFO] Loaded {len(images)} labeled real-handwriting pairs from {pdf_dir}"
        f" (upsample ×{n})"
    )
    return Dataset.from_dict({"image": images, "text": texts}).cast_column("image", HFImage())


_load_labeled_pairs = _load_pdf_labeled


def _balance_train_by_length(
    train,
    *,
    target_size: int,
    short_max_chars: int,
    short_fraction: float,
    seed: int,
):
    """Build a train set of `target_size` with ~short_fraction lines ≤ short_max_chars.

    Samples with replacement when the pool is smaller than the requested bucket.
    Fixes length bias from long-line-heavy synthetic corpora without re-uploading Hub data.
    """
    import random

    short_fraction = max(0.0, min(0.9, float(short_fraction)))
    if short_fraction <= 0 or target_size <= 0:
        return train

    texts = train["text"]
    short_idx = [i for i, t in enumerate(texts) if len((t or "").strip()) <= short_max_chars]
    long_idx = [i for i in range(len(train)) if i not in set(short_idx)]
    if not short_idx:
        print(
            f"[WARN] No train lines ≤ {short_max_chars} chars — skipping short-line balance"
        )
        return train
    if not long_idx:
        print("[WARN] All train lines are short — skipping length balance")
        return train

    rng = random.Random(seed)
    n_short = int(round(target_size * short_fraction))
    n_long = max(0, target_size - n_short)

    def _draw(pool: list[int], n: int) -> list[int]:
        if n <= 0:
            return []
        if len(pool) >= n:
            return rng.sample(pool, n)
        # With replacement to hit target mix when Hub pool is smaller than target_size.
        return [rng.choice(pool) for _ in range(n)]

    pick = _draw(short_idx, n_short) + _draw(long_idx, n_long)
    rng.shuffle(pick)
    balanced = train.select(pick)
    print(
        f"[INFO] Length-balanced train: {len(balanced)} samples "
        f"(short≤{short_max_chars}: {n_short}/{len(balanced)} = {n_short / max(1, len(balanced)):.1%}; "
        f"pool short={len(short_idx)} long={len(long_idx)})"
    )
    return balanced


def _image_to_rgb(img) -> Image.Image:
    """Decode Hub parquet / Image feature values to an RGB PIL image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        raw_bytes = img.get("bytes")
        if raw_bytes:
            from io import BytesIO

            return Image.open(BytesIO(raw_bytes)).convert("RGB")
        path = img.get("path")
        if path:
            return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)!r}")


class TrocrDataCollator:
    """On-the-fly processor — avoids caching ~GB of float pixel_values on disk."""

    def __init__(self, processor: TrOCRProcessor, max_target_length: int):
        self.processor = processor
        self.max_target_length = max_target_length
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features: list) -> dict:
        images = [_image_to_rgb(f["image"]) for f in features]
        texts = [f["text"] for f in features]
        enc = self.processor(
            images=images,
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = enc["labels"].clone()
        labels[labels == self.pad_id] = -100
        return {"pixel_values": enc["pixel_values"], "labels": labels}


def build_dataset(config: TrainConfig, hf_token: Optional[str]) -> DatasetDict:
    """Load imagefolder / Hub data; keep compact image+text columns (no float map cache)."""

    if config.dataset_name:
        print(f"[INFO] Loading dataset from HF Hub: {config.dataset_name}")
        raw = load_dataset(config.dataset_name, token=hf_token)
        if "validation" not in raw:
            raw = raw["train"].train_test_split(test_size=0.1, seed=config.seed)
            raw = DatasetDict({"train": raw["train"], "validation": raw["test"]})
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

    if config.pdf_labeled_dir:
        labeled = _load_pdf_labeled(config.pdf_labeled_dir, config.upsample_labeled)
        if labeled is not None:
            from datasets import concatenate_datasets

            raw["train"] = concatenate_datasets([raw["train"], labeled])
            print(
                f"[INFO] Combined training set: {len(raw['train'])} examples "
                f"(synthetic + labeled real-handwriting)"
            )

    if config.extra_dataset_name:
        from datasets import concatenate_datasets

        extra_ids = [x.strip() for x in str(config.extra_dataset_name).split(",") if x.strip()]
        up_raw = str(config.extra_dataset_upsample)
        up_parts = [p.strip() for p in up_raw.split(",")] if "," in up_raw else [up_raw]
        ups: list[int] = []
        for i, _ in enumerate(extra_ids):
            try:
                ups.append(max(1, int(up_parts[i] if i < len(up_parts) else up_parts[-1])))
            except ValueError:
                ups.append(1)

        for split in raw:
            cols = raw[split].column_names
            if "image" in cols:
                raw[split] = raw[split].cast_column("image", HFImage())
            drop = [c for c in cols if c not in ("image", "text")]
            if drop:
                raw[split] = raw[split].remove_columns(drop)

        for extra_id, up in zip(extra_ids, ups):
            print(f"[INFO] Loading extra Hub dataset: {extra_id} (×{up})", flush=True)
            extra = load_dataset(extra_id, token=hf_token)
            if "train" not in extra:
                extra = DatasetDict({"train": extra[list(extra.keys())[0]]})
            for split in list(extra.keys()):
                cols = extra[split].column_names
                if "image" in cols:
                    extra[split] = extra[split].cast_column("image", HFImage())
                drop = [c for c in cols if c not in ("image", "text")]
                if drop:
                    extra[split] = extra[split].remove_columns(drop)
            extra_train = extra["train"]
            if up > 1:
                extra_train = concatenate_datasets([extra_train] * up)
            before = len(raw["train"])
            raw["train"] = concatenate_datasets([raw["train"], extra_train])
            print(
                f"[INFO] Mixed extra train: +{len(extra_train)} "
                f"(×{up} from {extra_id}) → {before} → {len(raw['train'])}",
                flush=True,
            )
            if "validation" in extra and "validation" in raw:
                raw["validation"] = concatenate_datasets(
                    [raw["validation"], extra["validation"]]
                )
                print(
                    f"[INFO] Mixed extra val → {len(raw['validation'])} examples",
                    flush=True,
                )

    # Length curriculum before truncation: build target-sized mix with short oversampling.
    if config.short_line_fraction and config.short_line_fraction > 0:
        target = (
            int(config.max_train_samples)
            if config.max_train_samples and config.max_train_samples > 0
            else len(raw["train"])
        )
        raw["train"] = _balance_train_by_length(
            raw["train"],
            target_size=target,
            short_max_chars=int(config.short_line_max_chars),
            short_fraction=float(config.short_line_fraction),
            seed=config.seed,
        )
    elif config.max_train_samples and config.max_train_samples > 0:
        n = min(int(config.max_train_samples), len(raw["train"]))
        raw["train"] = raw["train"].shuffle(seed=config.seed).select(range(n))
        print(f"[INFO] Truncated train split to {n} samples (smoke / debug)")

    # Hub parquet often stores images as {"bytes": ...} until cast/decoded.
    for split in raw:
        cols = raw[split].column_names
        if "image" in cols:
            raw[split] = raw[split].cast_column("image", HFImage())
        # Drop leftover metadata columns; collator needs image + text.
        drop = [c for c in cols if c not in ("image", "text")]
        if drop:
            raw[split] = raw[split].remove_columns(drop)

    print("[INFO] Using on-the-fly image encode (no pixel_values map cache)")
    return raw


def push_dataset_to_hub(raw, config: TrainConfig, hf_token: str) -> str:
    user = _hf_username() or _require_env("HF_USERNAME")
    # Prefer explicit hub dataset id; otherwise publish under {user}/javanese-dataset.
    if config.dataset_name and "/" in config.dataset_name and not Path(config.dataset_name).exists():
        target = config.dataset_name
    else:
        target = f"{user}/javanese-dataset"
    # Always private: manuscript-derived samples must not be redistributed publicly.
    print(f"[INFO] Pushing PRIVATE dataset to HF Hub: {target}")
    raw.push_to_hub(target, token=hf_token, private=True)
    return target


def _final_cer(trainer: Seq2SeqTrainer, processor: TrOCRProcessor, cer_metric) -> Optional[float]:
    """One CER pass with generation after training (mid-train eval is loss-only by default)."""
    print("[INFO] Running final CER eval with generation...")
    old = trainer.args.predict_with_generate
    trainer.args.predict_with_generate = True
    try:
        out = trainer.predict(trainer.eval_dataset, metric_key_prefix="final")
        preds = out.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = out.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(preds, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        print(f"[OK] Final val CER: {cer:.4f}")
        return float(cer)
    finally:
        trainer.args.predict_with_generate = old


def train(config: TrainConfig) -> str:
    hf_token = _hf_token()
    hf_user = _hf_username()

    if config.push_to_hub and not hf_token:
        raise RuntimeError(
            "HF_TOKEN is unset — cannot push the trained model. "
            "Locally: set it in ../../.env. "
            "HF Jobs: pass --secrets HF_TOKEN. "
            "Space: Settings → Variables and secrets. "
            "Training aborted so GPU time is not wasted."
        )
    if config.push_to_hub and not hf_user and not config.hub_model_id:
        raise RuntimeError(
            "Cannot resolve Hub model id: set --hub_model_id, or HF_USERNAME / "
            "SPACE_AUTHOR_NAME, or use a token that whoami() can resolve."
        )

    log_device()
    print(f"[INFO] Base model: {config.base_model}")
    attn_impl = attn_implementation()
    model_load_kw = {}
    if attn_impl:
        model_load_kw["attn_implementation"] = attn_impl
        print(f"[INFO] attn_implementation={attn_impl} (required for stable XPU train)")
    if config.tokenizer_dir:
        print(f"[INFO] Loading processor from tokenizer_dir={config.tokenizer_dir}")
        processor = TrOCRProcessor.from_pretrained(str(config.tokenizer_dir))
        model = VisionEncoderDecoderModel.from_pretrained(config.base_model, **model_load_kw)
        tok_len = len(processor.tokenizer)
        emb_len = model.decoder.get_input_embeddings().weight.shape[0]
        if emb_len != tok_len:
            model.decoder.resize_token_embeddings(tok_len)
            model.config.vocab_size = tok_len
            if getattr(model.config, "decoder", None) is not None:
                model.config.decoder.vocab_size = tok_len
            print(f"[INFO] Resized decoder embeddings {emb_len} → {tok_len} for tokenizer_dir")
    else:
        processor = TrOCRProcessor.from_pretrained(config.base_model)
        model = VisionEncoderDecoderModel.from_pretrained(config.base_model, **model_load_kw)
        if config.expand_javanese_tokenizer:
            corpus = config.javanese_corpus
            if corpus is None:
                cand = Path(__file__).resolve().parent.parent / "javanese_corpus_ocr.txt"
                corpus = cand if cand.is_file() else None
            print("[INFO] Expanding tokenizer with atomic Javanese characters…")
            _expand_javanese_tokenizer(processor, model, corpus_path=corpus)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.bos_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # Keep generation_config in sync — Trainer may prefer it over model.config at generate().
    # A wrong decoder_start (e.g. EOS=2) yields garbage free-run CER while teacher-forced
    # train loss still looks healthy.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.generation_config.bos_token_id = processor.tokenizer.cls_token_id
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.sep_token_id
        model.generation_config.max_length = config.max_target_length
        model.generation_config.early_stopping = True
        # Keep ngram=0: historically ngram=3 broke byte-level BPE free-gen.
        # Atomic Javanese vocab still uses ngram=0; free-gen anti-runaway
        # (cecak/sandhangan loops) is applied at score time via
        # generation_utils.NoRunawayMarksLogitsProcessor — not serializable here.
        model.generation_config.no_repeat_ngram_size = 0
        model.generation_config.length_penalty = 1.0
        model.generation_config.num_beams = config.num_beams
    model.config.max_length = config.max_target_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0
    model.config.num_beams = config.num_beams

    if config.freeze_encoder:
        _freeze_vision_encoder(model)

    raw = build_dataset(config, hf_token)

    if config.push_dataset and hf_token:
        push_dataset_to_hub(raw, config, hf_token)

    collator = TrocrDataCollator(processor, config.max_target_length)

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

    use_fp16, use_bf16 = use_amp()

    hub_id = None
    hub_kwargs = {}
    if config.push_to_hub and hf_token:
        hub_id = config.hub_model_id or f"{hf_user}/javanese-trocr-handwritten"
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": hub_id,
            "hub_token": hf_token,
            # Push on every checkpoint save so a Job timeout still leaves weights on Hub.
            "hub_strategy": "every_save",
        }
        print(f"[INFO] Trained model will be pushed to: {hub_id} (hub_strategy=every_save)")

    # Eval cadence: every N epochs. Prefer epoch-aligned eval (and always when saving
    # per epoch) so Hub tags / post-train scoring line up with eval_loss logs.
    n_train = len(raw["train"])
    steps_per_epoch = max(1, (n_train + config.batch_size - 1) // config.batch_size)
    eval_every = max(1, int(config.eval_every_epochs))
    use_epoch_eval = True  # save_strategy=epoch; keep eval on the same boundary
    if config.predict_with_generate:
        eval_strategy = "epoch"
        eval_steps = steps_per_epoch * eval_every
        metrics_fn = compute_metrics
        print(
            f"[INFO] Mid-train CER enabled every ~{eval_every} epoch(s); "
            f"this is slow on TrOCR."
        )
    else:
        eval_strategy = "epoch"
        eval_steps = steps_per_epoch * eval_every
        metrics_fn = None
        print(
            f"[INFO] Mid-train eval is loss-only every ~{eval_every} epoch(s); "
            f"CER runs once at the end."
        )

    # gradient_checkpointing helps large batches fit on 24GB; disable when VRAM allows for speed.
    use_gc = bool(config.gradient_checkpointing)
    if use_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[INFO] Gradient checkpointing enabled (memory)")
    else:
        print("[INFO] Gradient checkpointing disabled (speed)")

    train_kwargs = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": min(config.eval_batch_size, config.batch_size),
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "logging_steps": 50,
        "eval_strategy": eval_strategy,
        "eval_steps": eval_steps,
        "save_strategy": "epoch",
        "save_total_limit": config.save_total_limit,
        "load_best_model_at_end": bool(config.load_best_model_at_end),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "predict_with_generate": config.predict_with_generate,
        "generation_max_length": config.max_target_length,
        "generation_num_beams": config.num_beams,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "report_to": "none",
        # Collator decodes PNG+runs processor; a couple workers help hide decode latency.
        **dataloader_kwargs(),
        "gradient_checkpointing": use_gc,
        "remove_unused_columns": False,
        # TrOCR + XPU: keep compile off until smoke-proven on the SKU.
        "torch_compile": False,
        **hub_kwargs,
    }
    if eval_strategy != "steps":
        train_kwargs.pop("eval_steps", None)
    if config.max_steps is not None and config.max_steps > 0:
        train_kwargs["max_steps"] = int(config.max_steps)
        print(f"[INFO] max_steps={config.max_steps} (smoke cap)")

    if (
        _HAS_XPU_HELPER
        and build_seq2seq_training_arguments is not None
        and ultra_255h_config is not None
        and pick_device() == "xpu"
    ):
        xpu_cfg = ultra_255h_config(
            torch_compile=False,
            gradient_checkpointing=use_gc,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=min(config.eval_batch_size, config.batch_size),
            # TrOCR cooks use micro-batch only; helper's default accum=4 would
            # silently 4× the effective batch and shrink steps/epoch.
            gradient_accumulation_steps=1,
        )
        # Drop keys the factory already sets from xpu_cfg so overrides stay authoritative.
        override_kwargs = {
            k: v
            for k, v in train_kwargs.items()
            if k
            not in {
                "output_dir",
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_checkpointing",
                "torch_compile",
            }
        }
        args = build_seq2seq_training_arguments(
            str(config.output_dir),
            config=xpu_cfg,
            **override_kwargs,
        )
        print("[INFO] Seq2SeqTrainingArguments via transformers-xpu-helper", flush=True)
    else:
        train_kwargs.pop("torch_compile", None)
        args = Seq2SeqTrainingArguments(**train_kwargs)

    callbacks = []
    if config.early_stopping_patience and config.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(config.early_stopping_patience)
            )
        )
        print(
            f"[INFO] EarlyStopping patience={config.early_stopping_patience} on eval_loss",
            flush=True,
        )
    if config.push_to_hub and hub_id and config.hub_tag_epochs:
        callbacks.append(HubEpochTagCallback(hub_id, hf_token))
        print(f"[INFO] Hub epoch tags enabled on {hub_id}", flush=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=raw["train"],
        eval_dataset=raw["validation"],
        data_collator=collator,
        compute_metrics=metrics_fn,
        processing_class=processor,
        callbacks=callbacks or None,
    )

    resume = config.resume_from_checkpoint
    if resume is True:
        print("[INFO] resume_from_checkpoint=True (latest under output_dir)", flush=True)
    elif isinstance(resume, str) and resume.strip():
        resume = resume.strip()
        print(f"[INFO] resume_from_checkpoint={resume}", flush=True)
    else:
        resume = None

    print("[INFO] Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = config.output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"[OK] Model + processor saved to {final_dir.resolve()}")

    try:
        if not config.skip_final_cer:
            _final_cer(trainer, processor, cer_metric)
        else:
            print("[INFO] Skipping final CER (chunked run)", flush=True)
    except Exception as exc:
        print(f"[WARN] Final CER eval failed (model still saved): {exc}")

    if config.push_to_hub and hf_token:
        # Push the final/ folder to Hub *repo root* so loaders without subfolder
        # get the trained weights + processor (avoids stale nested copies).
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            api.create_repo(hub_id, repo_type="model", private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(final_dir),
                repo_id=hub_id,
                repo_type="model",
                commit_message="End of training — final weights at repo root",
            )
            print(f"[OK] Model pushed to HF Hub root: {hub_id}")
        except Exception as exc:
            print(f"[WARN] upload_folder failed ({exc}); falling back to trainer.push_to_hub")
            trainer.push_to_hub(commit_message="End of training — final weights")
            print(f"[OK] Model pushed to HF Hub: {hub_id}")

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
                   type=str, default=None,
                   help="Dir of label_XXXX.{png,jpg} + .txt pairs. "
                        "Omit, or pass 'none'/'-' to skip (Jobs-friendly).")
    p.add_argument("--upsample_labeled", type=int, default=1,
                   help="Duplicate labeled pairs by this factor.")
    p.add_argument("--eval_every_epochs", type=int, default=2,
                   help="Run validation every N epochs (default 2).")
    p.add_argument("--predict_with_generate", action="store_true",
                   help="Enable slow mid-train CER via generation (off by default).")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap optimizer steps (smoke Jobs).")
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Truncate train split (smoke / debug).")
    p.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade speed for VRAM (default on). Use --no-gradient_checkpointing when batch fits.",
    )
    p.add_argument(
        "--expand_javanese_tokenizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add atomic Javanese chars to the BPE vocab (default on). Critical for free-run CER.",
    )
    p.add_argument(
        "--tokenizer_dir",
        type=Path,
        default=None,
        help="Optional pre-expanded processor dir from expand_javanese_tokenizer.py",
    )
    p.add_argument(
        "--javanese_corpus",
        type=Path,
        default=None,
        help="Corpus used to discover extra Javanese codepoints when expanding tokenizer",
    )
    p.add_argument(
        "--short_line_max_chars",
        type=int,
        default=8,
        help="Lines with ≤ this many chars count as 'short' for length balancing.",
    )
    p.add_argument(
        "--short_line_fraction",
        type=float,
        default=0.0,
        help="Fraction of train that should be short lines (0=off, 0.25 recommended). "
        "Uses sampling with replacement when the Hub pool is smaller than --max_train_samples.",
    )
    p.add_argument(
        "--extra_dataset_name",
        default=None,
        help="Optional Hub dataset(s) mixed into train/val. Comma-separated OK, "
        "e.g. 'user/javanese-dataset,user/javanese-nusaaksara-ocr'.",
    )
    p.add_argument(
        "--extra_dataset_upsample",
        default="8",
        help="Upsample factor(s) for --extra_dataset_name. Single int or comma list "
        "aligned to names (default 8). Example: '1,8'.",
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="LR warmup ratio (use 0 on resume chunks after the first).",
    )
    p.add_argument(
        "--skip_final_cer",
        action="store_true",
        help="Skip slow final generate-CER (recommended for chunked HF Jobs / NAS).",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help="Resume Trainer from checkpoint. Pass a path, or flag alone for latest "
        "under --output_dir (NAS hands-off restarts).",
    )
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after N evals without eval_loss improvement (0=off). Implies "
        "load_best_model_at_end.",
    )
    p.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Keep best eval_loss checkpoint as the final model.",
    )
    p.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze the vision encoder; train decoder (+ cross-attn) only.",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Max checkpoints kept on disk. Omit/0 = keep all epoch checkpoints.",
    )
    p.add_argument(
        "--no_hub_tag_epochs",
        action="store_true",
        help="Do not create Hub tags epoch-N after each epoch push.",
    )
    a = p.parse_args()

    if a.dataset_dir is None and a.dataset_name is None:
        a.dataset_dir = Path(__file__).resolve().parent / "trocr_dataset"

    labeled_raw = (a.pdf_labeled_dir or "").strip()
    if labeled_raw.lower() in ("", "none", "null", "-"):
        labeled_dir = None
    else:
        labeled_dir = Path(labeled_raw)

    extra_name = (a.extra_dataset_name or "").strip() or None
    patience = max(0, int(a.early_stopping_patience))
    load_best = bool(a.load_best_model_at_end) or patience > 0
    resume: bool | str = False
    if a.resume_from_checkpoint == "auto":
        resume = True
    elif a.resume_from_checkpoint:
        resume = a.resume_from_checkpoint
    save_limit = a.save_total_limit
    if save_limit is not None and int(save_limit) <= 0:
        save_limit = None

    return TrainConfig(
        base_model=a.base_model,
        dataset_dir=a.dataset_dir,
        dataset_name=a.dataset_name,
        output_dir=a.output_dir,
        epochs=a.epochs,
        batch_size=a.batch_size,
        eval_batch_size=a.eval_batch_size,
        learning_rate=a.lr,
        warmup_ratio=max(0.0, float(a.warmup_ratio)),
        max_target_length=a.max_target_length,
        hub_model_id=a.hub_model_id,
        push_to_hub=not a.no_push,
        push_dataset=a.push_dataset,
        num_beams=a.num_beams,
        pdf_labeled_dir=labeled_dir,
        upsample_labeled=float(max(1, int(a.upsample_labeled))),
        eval_every_epochs=max(1, int(a.eval_every_epochs)),
        predict_with_generate=bool(a.predict_with_generate),
        max_steps=a.max_steps,
        max_train_samples=a.max_train_samples,
        gradient_checkpointing=bool(a.gradient_checkpointing),
        expand_javanese_tokenizer=bool(a.expand_javanese_tokenizer),
        tokenizer_dir=a.tokenizer_dir,
        javanese_corpus=a.javanese_corpus,
        short_line_max_chars=max(1, int(a.short_line_max_chars)),
        short_line_fraction=max(0.0, float(a.short_line_fraction)),
        extra_dataset_name=extra_name,
        extra_dataset_upsample=a.extra_dataset_upsample,
        skip_final_cer=bool(a.skip_final_cer),
        resume_from_checkpoint=resume,
        early_stopping_patience=patience,
        load_best_model_at_end=load_best,
        freeze_encoder=bool(a.freeze_encoder),
        save_total_limit=save_limit,
        hub_tag_epochs=not bool(a.no_hub_tag_epochs),
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
