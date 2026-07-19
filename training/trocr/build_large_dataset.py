#!/usr/bin/env python3
"""
build_large_dataset.py — Generate a large synthetic TrOCR imagefolder dataset in parallel.

Uses the cleaned OCR corpus (javanese_corpus_ocr.txt) by default.
Writes train/ + validation/ with per-sample .txt labels (crash-safe) + metadata.jsonl.
Optionally pushes a PRIVATE dataset to the Hub.

Example:
  python build_large_dataset.py --num_train 1000000 --num_val 10000 --workers 10 --push
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from generate_trocr_dataset import TrOCRDatasetGenerator

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.is_file():
        load_dotenv(_env)


_WORKER_STATE: dict = {}


def _init_worker(corpus: str, fonts_dir: str, pdfs_dir: str, seed: int) -> None:
    random.seed(seed + os.getpid())
    try:
        import numpy as np

        np.random.seed(seed + os.getpid())
    except Exception:
        pass
    _WORKER_STATE["gen"] = TrOCRDatasetGenerator(
        corpus_path=Path(corpus),
        fonts_dir=Path(fonts_dir),
        pdfs_dir=Path(pdfs_dir),
    )


def _render_one(idx: int) -> tuple[int, bytes, str]:
    gen: TrOCRDatasetGenerator = _WORKER_STATE["gen"]
    text = random.choice(gen.corpus)
    img = gen.render_sample(text)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return idx, buf.getvalue(), text


def _write_sample(split_dir: Path, idx: int, png: bytes, text: str) -> None:
    stem = f"sample_{idx:07d}"
    (split_dir / f"{stem}.png").write_bytes(png)
    (split_dir / f"{stem}.txt").write_text(text, encoding="utf-8")


def _existing_complete(split_dir: Path, count: int) -> set[int]:
    """Indices that already have both png and txt (resume support)."""
    done: set[int] = set()
    if not split_dir.exists():
        return done
    for i in range(count):
        stem = split_dir / f"sample_{i:07d}"
        if stem.with_suffix(".png").exists() and stem.with_suffix(".txt").exists():
            done.add(i)
    return done


def _write_metadata(split_dir: Path, count: int) -> None:
    meta_path = split_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for idx in range(count):
            stem = f"sample_{idx:07d}"
            text = (split_dir / f"{stem}.txt").read_text(encoding="utf-8")
            f.write(
                json.dumps({"file_name": f"{stem}.png", "text": text}, ensure_ascii=False)
                + "\n"
            )


def generate_split(
    gen_args: tuple,
    split_dir: Path,
    count: int,
    workers: int,
    seed: int,
    chunk_size: int = 4000,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    corpus, fonts_dir, pdfs_dir = gen_args

    already = _existing_complete(split_dir, count)
    todo = [i for i in range(count) if i not in already]
    print(
        f"[INFO] Generating {count} samples -> {split_dir} "
        f"({workers} workers; resume skip={len(already)}, todo={len(todo)})"
    )
    if not todo:
        print("[INFO] Nothing to generate; rebuilding metadata.jsonl")
        _write_metadata(split_dir, count)
        return

    t0 = time.time()
    done = len(already)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(corpus, fonts_dir, pdfs_dir, seed),
    ) as pool:
        for start in range(0, len(todo), chunk_size):
            batch = todo[start : start + chunk_size]
            futs = [pool.submit(_render_one, i) for i in batch]
            for fut in as_completed(futs):
                idx, png, text = fut.result()
                _write_sample(split_dir, idx, png, text)
                done += 1
                if done % 2000 == 0 or done == count:
                    elapsed = time.time() - t0
                    # Rate based on this run's newly completed work
                    newly = done - len(already)
                    rate = newly / max(elapsed, 1e-6)
                    eta = (count - done) / max(rate, 1e-6)
                    print(
                        f"  [{split_dir.name}] {done}/{count} "
                        f"({rate:.1f}/s, ETA {eta/60:.1f} min)",
                        flush=True,
                    )

    print(f"[INFO] Writing metadata.jsonl for {count} samples …", flush=True)
    _write_metadata(split_dir, count)
    print(f"[OK] {split_dir.name} done in {(time.time()-t0)/60:.1f} min", flush=True)


def push_private(dataset_dir: Path, repo_id: str, max_retries: int = 8) -> None:
    from datasets import DatasetDict, load_dataset
    import time as _time

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset — cannot push.")
    print(f"[INFO] Loading imagefolder from {dataset_dir} for private Hub push …", flush=True)
    train = load_dataset("imagefolder", data_dir=str(dataset_dir / "train"), split="train")
    val = load_dataset("imagefolder", data_dir=str(dataset_dir / "validation"), split="train")
    raw = DatasetDict({"train": train, "validation": val})
    print(f"[INFO] train={len(train)} validation={len(val)} -> PRIVATE {repo_id}", flush=True)

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw.push_to_hub(
                repo_id,
                token=token,
                private=True,
                max_shard_size="200MB",
                num_proc=1,
            )
            print(f"[OK] https://huggingface.co/datasets/{repo_id} (private)", flush=True)
            return
        except Exception as exc:
            last_err = exc
            wait = min(120, 10 * attempt)
            print(
                f"[WARN] push attempt {attempt}/{max_retries} failed: {exc!r}\n"
                f"       retrying in {wait}s …",
                flush=True,
            )
            _time.sleep(wait)
    raise RuntimeError(f"push_to_hub failed after {max_retries} attempts") from last_err


def main() -> None:
    here = Path(__file__).resolve().parent
    training = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=training / "javanese_corpus_ocr.txt")
    p.add_argument("--fonts_dir", type=Path, default=training / "fonts")
    p.add_argument("--pdfs_dir", type=Path, default=training / "pdfs")
    p.add_argument("--output_dir", type=Path, default=here / "trocr_dataset_1m")
    p.add_argument("--num_train", type=int, default=1_000_000)
    p.add_argument("--num_val", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo_id", default="thesimonharms/javanese-dataset")
    p.add_argument("--skip_generate", action="store_true")
    args = p.parse_args()

    if not args.corpus.exists():
        sys.exit(
            f"[ERROR] Corpus not found: {args.corpus}\n"
            f"        Run: python clean_javanese_corpus.py"
        )

    if not args.skip_generate:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        gen_args = (str(args.corpus), str(args.fonts_dir), str(args.pdfs_dir))
        generate_split(gen_args, args.output_dir / "train", args.num_train, args.workers, args.seed)
        generate_split(
            gen_args, args.output_dir / "validation", args.num_val, args.workers, args.seed + 1
        )

    if args.push:
        push_private(args.output_dir, args.repo_id)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
