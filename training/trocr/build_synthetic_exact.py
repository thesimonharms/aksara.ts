#!/usr/bin/env python3
"""Build the exact-match synthetic TrOCR set: short aksara, 384×384, clean print.

  python build_synthetic_exact.py --dry-run
  python build_synthetic_exact.py --num_train 60000 --num_val 2500 --workers 10
  python build_synthetic_exact.py --skip_generate --export-parquet --push
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

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    _env = Path(__file__).resolve().parents[2] / ".env"
    if _env.is_file():
        load_dotenv(_env)

from synthetic_exact.corpus import load_clean_pools
from synthetic_exact.primer import build_primer_texts
from synthetic_exact.render import ExactRenderer

_WORKER: dict = {}


def _init_worker(fonts_dir: str, plan_path: str, seed: int) -> None:
    random.seed(seed + os.getpid())
    try:
        import numpy as np

        np.random.seed(seed + os.getpid())
    except Exception:
        pass
    plan: dict[int, dict] = {}
    with Path(plan_path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            plan[int(row["idx"])] = row
    _WORKER["plan"] = plan
    _WORKER["renderer"] = ExactRenderer(Path(fonts_dir))


def _render_one(idx: int) -> tuple[int, bytes, str]:
    row = _WORKER["plan"][idx]
    renderer: ExactRenderer = _WORKER["renderer"]
    img = renderer.render_sample(row["text"], aug_seed=int(row["aug_seed"]))
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return idx, buf.getvalue(), row["text"]


def _bucket(n: int) -> str:
    if n <= 5:
        return "short"
    if n <= 9:
        return "mid"
    return "long"


def _write_plan(
    texts: list[str],
    count: int,
    seed: int,
    plan_path: Path,
    *,
    source: str,
    short_frac: float,
    mid_frac: float,
) -> None:
    rng = random.Random(seed)
    by_b = {"short": [], "mid": [], "long": []}
    for t in texts:
        by_b[_bucket(len(t))].append(t)
    all_t = texts[:] or ["ꦲ"]
    for b in by_b:
        if not by_b[b]:
            by_b[b] = all_t

    n_short = int(round(count * short_frac))
    n_mid = int(round(count * mid_frac))
    n_long = count - n_short - n_mid
    quotas = (["short"] * n_short) + (["mid"] * n_mid) + (["long"] * n_long)
    rng.shuffle(quotas)

    decks = {b: [] for b in by_b}
    for b, pool in by_b.items():
        need = quotas.count(b) + 8
        deck: list[str] = []
        while len(deck) < need:
            mix = list(pool)
            rng.shuffle(mix)
            deck.extend(mix)
        decks[b] = deck

    cursors = {b: 0 for b in by_b}
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        for idx, bucket in enumerate(quotas):
            text = decks[bucket][cursors[bucket]]
            cursors[bucket] += 1
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "text": text,
                        "bucket": bucket,
                        "source": source,
                        "aug_seed": rng.randint(0, 2**31 - 1),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"[OK] plan {plan_path.name} n={count} "
        f"short={n_short} mid={n_mid} long={n_long} source={source}",
        flush=True,
    )


def _merge_plans(primer_plan: Path, corpus_plan: Path, out: Path, seed: int) -> int:
    rows: list[dict] = []
    for path in (primer_plan, corpus_plan):
        with path.open(encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            row["idx"] = i
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] merged train plan {out} n={len(rows)}", flush=True)
    return len(rows)


def generate_split(
    *,
    fonts_dir: Path,
    plan_path: Path,
    split_dir: Path,
    count: int,
    workers: int,
    seed: int,
    chunk_size: int = 1000,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for i in range(count):
        png = split_dir / f"sample_{i:07d}.png"
        txt = split_dir / f"sample_{i:07d}.txt"
        if not (png.exists() and txt.exists()):
            todo.append(i)
    print(
        f"[INFO] render {split_dir.name}: {count} todo={len(todo)} workers={workers}",
        flush=True,
    )
    if not todo:
        _write_metadata(split_dir, count, plan_path)
        return

    t0 = time.time()
    done = count - len(todo)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(fonts_dir), str(plan_path), seed),
    ) as pool:
        for start in range(0, len(todo), chunk_size):
            batch = todo[start : start + chunk_size]
            futs = [pool.submit(_render_one, i) for i in batch]
            for fut in as_completed(futs):
                idx, png, text = fut.result()
                stem = split_dir / f"sample_{idx:07d}"
                stem.with_suffix(".png").write_bytes(png)
                stem.with_suffix(".txt").write_text(text, encoding="utf-8")
                done += 1
                if done % 1000 == 0 or done == count:
                    newly = done - (count - len(todo))
                    rate = newly / max(time.time() - t0, 1e-6)
                    eta = (count - done) / max(rate, 1e-6)
                    print(
                        f"  [{split_dir.name}] {done}/{count} "
                        f"({rate:.1f}/s ETA {eta/60:.1f}m)",
                        flush=True,
                    )
    _write_metadata(split_dir, count, plan_path)
    print(f"[OK] {split_dir.name} in {(time.time()-t0)/60:.1f} min", flush=True)


def _write_metadata(split_dir: Path, count: int, plan_path: Path) -> None:
    meta_path = split_dir / "metadata.jsonl"
    plan_by_idx: dict[int, dict] = {}
    with plan_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            plan_by_idx[int(row["idx"])] = row
    with meta_path.open("w", encoding="utf-8") as mf:
        for idx in range(count):
            text = plan_by_idx[idx]["text"]
            mf.write(
                json.dumps(
                    {"file_name": f"sample_{idx:07d}.png", "text": text},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[OK] metadata {meta_path}", flush=True)


def export_parquet(
    dataset_dir: Path,
    export_dir: Path,
    num_train: int,
    num_val: int,
    train_shards: int,
    val_shards: int,
    *,
    jpeg_quality: int = 90,
) -> None:
    """Write Hub parquet with JPEG bytes so 384×384 paper-grain PNGs don't explode."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    data_dir = export_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "README.md").write_text(
        "---\n"
        "license: other\n"
        "pretty_name: Javanese synthetic exact OCR (private)\n"
        "dataset_info:\n"
        "  features:\n"
        "  - name: image\n"
        "    dtype: image\n"
        "  - name: text\n"
        "    dtype: string\n"
        f"  splits:\n"
        f"  - name: train\n"
        f"    num_examples: {num_train}\n"
        f"  - name: validation\n"
        f"    num_examples: {num_val}\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/train-*\n"
        "  - split: validation\n"
        "    path: data/validation-*\n"
        "---\n\n"
        "# javanese-synthetic-exact\n\n"
        "Private short clean synthetic Javanese Aksara OCR (384×384 JPEG).\n"
        "Max 12 aksara, no Latin, no manuscript backgrounds.\n"
        "Not for redistribution.\n",
        encoding="utf-8",
    )

    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

    def _write_split(split_dir: Path, split: str, count: int, num_shards: int) -> None:
        per = (count + num_shards - 1) // num_shards
        print(
            f"[INFO] {split}: {count} samples -> {num_shards} JPEG shards (~{per}/shard)",
            flush=True,
        )
        for shard_i in range(num_shards):
            start = shard_i * per
            end = min(count, start + per)
            if start >= count:
                break
            out = data_dir / f"{split}-{shard_i:05d}-of-{num_shards:05d}.parquet"
            if out.exists() and 1_000_000 < out.stat().st_size < 100_000_000:
                print(f"  skip {out.name} ({out.stat().st_size/1e6:.1f}MB)", flush=True)
                continue
            images = []
            texts = []
            for i in range(start, end):
                stem = split_dir / f"sample_{i:07d}"
                img = Image.open(stem.with_suffix(".png")).convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
                images.append(
                    {"bytes": buf.getvalue(), "path": f"sample_{i:07d}.jpg"}
                )
                texts.append(stem.with_suffix(".txt").read_text(encoding="utf-8"))
                if (i - start + 1) % 2000 == 0:
                    print(
                        f"  [{split} shard {shard_i}] {i - start + 1}/{end - start}",
                        flush=True,
                    )
            table = pa.table(
                {
                    "image": pa.array(images, type=image_type),
                    "text": pa.array(texts, type=pa.string()),
                }
            )
            pq.write_table(table, out, compression="zstd")
            print(
                f"  wrote {out.name} rows={end - start} "
                f"size={out.stat().st_size / 1e6:.1f}MB",
                flush=True,
            )
        print(f"[OK] {split} export done", flush=True)

    _write_split(dataset_dir / "train", "train", num_train, train_shards)
    _write_split(dataset_dir / "validation", "validation", num_val, val_shards)


def main() -> None:
    here = Path(__file__).resolve().parent
    training = here.parent
    p = argparse.ArgumentParser(description="Build exact-match synthetic Javanese OCR data.")
    p.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="Corpus txt (repeatable). Default: javanese_corpus_ocr.txt",
    )
    p.add_argument("--fonts_dir", type=Path, default=training / "fonts")
    p.add_argument("--output_dir", type=Path, default=here / "trocr_dataset_exact")
    p.add_argument("--num_train", type=int, default=60_000)
    p.add_argument("--num_val", type=int, default=2_500)
    p.add_argument("--primer_repeats", type=int, default=8)
    p.add_argument("--max_chars", type=int, default=12)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--export-parquet", action="store_true")
    p.add_argument("--train_shards", type=int, default=10)
    p.add_argument("--val_shards", type=int, default=1)
    p.add_argument(
        "--export_dir",
        type=Path,
        default=here / "hub_export_javanese_synthetic_exact",
    )
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo_id", default="thesimonharms/javanese-synthetic-exact")
    args = p.parse_args()

    corpora = args.corpus or [training / "javanese_corpus_ocr.txt"]
    primer = build_primer_texts()
    train_pool, val_pool = load_clean_pools(corpora, min_len=2, max_len=args.max_chars)
    print(
        f"[INFO] primer={len(primer)} corpus_train={len(train_pool)} "
        f"corpus_val={len(val_pool)} fonts={len(list(args.fonts_dir.glob('*.ttf'))) + len(list(args.fonts_dir.glob('*.otf')))}",
        flush=True,
    )
    if len(val_pool) < args.num_val:
        sys.exit(f"[ERROR] val pool {len(val_pool)} < num_val {args.num_val}")
    if len(train_pool) < 1000:
        sys.exit(f"[ERROR] train pool too small: {len(train_pool)}")

    n_primer = min(args.num_train // 3, len(primer) * max(1, args.primer_repeats))
    n_corpus = args.num_train - n_primer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primer_plan = args.output_dir / "primer_plan.jsonl"
    corpus_plan = args.output_dir / "corpus_train_plan.jsonl"
    train_plan = args.output_dir / "train_plan.jsonl"
    val_plan = args.output_dir / "val_plan.jsonl"

    _write_plan(
        primer,
        n_primer,
        args.seed,
        primer_plan,
        source="primer",
        short_frac=0.55,
        mid_frac=0.40,
    )
    _write_plan(
        train_pool,
        n_corpus,
        args.seed + 7,
        corpus_plan,
        source="corpus",
        short_frac=0.30,
        mid_frac=0.50,
    )
    n_train = _merge_plans(primer_plan, corpus_plan, train_plan, args.seed + 13)
    _write_plan(
        val_pool,
        args.num_val,
        args.seed + 99,
        val_plan,
        source="corpus",
        short_frac=0.30,
        mid_frac=0.50,
    )

    if args.dry_run:
        print("[OK] dry-run complete", flush=True)
        return

    if not args.skip_generate:
        generate_split(
            fonts_dir=args.fonts_dir,
            plan_path=train_plan,
            split_dir=args.output_dir / "train",
            count=n_train,
            workers=args.workers,
            seed=args.seed,
        )
        generate_split(
            fonts_dir=args.fonts_dir,
            plan_path=val_plan,
            split_dir=args.output_dir / "validation",
            count=args.num_val,
            workers=args.workers,
            seed=args.seed + 1,
        )

    if args.export_parquet or args.push:
        export_parquet(
            args.output_dir,
            args.export_dir,
            n_train,
            args.num_val,
            args.train_shards,
            args.val_shards,
        )
    if args.push:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            sys.exit("[ERROR] HF_TOKEN unset")
        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
        print(f"[INFO] private Hub push -> {args.repo_id}", flush=True)
        api.upload_large_folder(
            folder_path=str(args.export_dir),
            repo_id=args.repo_id,
            repo_type="dataset",
            num_workers=2,
            print_report_every=5,
            ignore_patterns=[".cache", ".cache/**", "**/.cache/**"],
        )
        print(f"[OK] https://huggingface.co/datasets/{args.repo_id} (private)", flush=True)
    else:
        print("[INFO] Hub push skipped (pass --push when ready)", flush=True)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
