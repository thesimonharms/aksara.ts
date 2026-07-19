#!/usr/bin/env python3
"""Build a length-balanced unique Hub subset from trocr_dataset_1m and upload it.

Selects ~180k train (+ val) with ~15% short lines (≤8 chars) — unique indices only,
no replacement. Writes parquet shards then resumable Hub upload.

Always upload to a NEW dataset repo (default: javanese-dataset-180k).
Never overwrite an existing dataset unless --replace_remote is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.is_file():
        load_dotenv(env)


def _bucket(text: str, short_max: int) -> str:
    n = len(text.strip())
    if n <= short_max:
        return "short"
    if n <= 20:
        return "mid"
    return "long"


def scan_metadata(meta_path: Path, short_max: int) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    with meta_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            text = (row.get("text") or "").strip()
            if not text:
                continue
            file_name = row["file_name"]
            buckets[_bucket(text, short_max)].append(
                {"file_name": file_name, "text": text, "idx": i}
            )
            if (i + 1) % 200_000 == 0:
                print(
                    f"  scanned {i+1}: "
                    + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()),
                    flush=True,
                )
    return buckets


def select_balanced(
    buckets: dict[str, list[dict]],
    n_total: int,
    short_fraction: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    n_short = int(round(n_total * short_fraction))
    # Split remainder ~35% mid / 65% long among non-short (matches natural OCR mix a bit).
    n_rest = n_total - n_short
    n_mid = int(round(n_rest * 0.35))
    n_long = n_rest - n_mid

    def take(name: str, n: int) -> list[dict]:
        pool = buckets.get(name, [])
        if len(pool) < n:
            raise SystemExit(
                f"[ERROR] Need {n} '{name}' samples but only have {len(pool)}. "
                f"Lower --num_train or --short_fraction."
            )
        return rng.sample(pool, n)

    chosen = take("short", n_short) + take("mid", n_mid) + take("long", n_long)
    rng.shuffle(chosen)
    print(
        f"[INFO] Selected {len(chosen)}: short={n_short} mid={n_mid} long={n_long}",
        flush=True,
    )
    return chosen


def write_parquet_shards(
    split_dir: Path,
    rows: list[dict],
    out_dir: Path,
    split: str,
    num_shards: int,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_size = (len(rows) + num_shards - 1) // num_shards

    for si in range(num_shards):
        chunk = rows[si * shard_size : (si + 1) * shard_size]
        if not chunk:
            continue
        out = out_dir / f"{split}-{si:05d}-of-{num_shards:05d}.parquet"
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"  skip existing {out.name}", flush=True)
            continue

        images: list[dict] = []
        texts: list[str] = []
        t0 = time.time()
        for j, row in enumerate(chunk):
            png_path = split_dir / row["file_name"]
            if not png_path.is_file():
                # metadata sometimes lacks path prefix
                alt = split_dir / Path(row["file_name"]).name
                png_path = alt if alt.is_file() else png_path
            raw = png_path.read_bytes()
            # Hub Image feature expects {"bytes": ...} or PIL; store bytes.
            images.append({"bytes": raw})
            texts.append(row["text"])
            if (j + 1) % 2000 == 0:
                print(
                    f"  {split} shard {si}: {j+1}/{len(chunk)} "
                    f"({(j+1)/(time.time()-t0):.1f} img/s)",
                    flush=True,
                )

        table = pa.table(
            {
                "image": pa.array(images),
                "text": pa.array(texts, type=pa.string()),
            }
        )
        pq.write_table(table, out, compression="zstd")
        print(
            f"  wrote {out.name} rows={len(chunk)} "
            f"({out.stat().st_size/1e6:.1f} MB)",
            flush=True,
        )


def upload(export_dir: Path, repo_id: str, replace_remote: bool) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    if replace_remote:
        print("[INFO] Clearing remote data/*.parquet so old 60k shards cannot linger…", flush=True)
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset")
            stale = [f for f in files if f.startswith("data/") and f.endswith(".parquet")]
            for i in range(0, len(stale), 20):
                batch = stale[i : i + 20]
                if batch:
                    api.delete_files(batch, repo_id=repo_id, repo_type="dataset")
                    print(f"  deleted {len(batch)} remote parquet files", flush=True)
        except Exception as exc:
            print(f"[WARN] remote cleanup: {exc}", flush=True)

    print(f"[INFO] upload_large_folder -> {repo_id}", flush=True)
    api.upload_large_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=4,
        print_report_every=5,
    )
    print(f"[OK] https://huggingface.co/datasets/{repo_id} (private)", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=Path, default=here / "trocr_dataset_1m")
    p.add_argument("--export_dir", type=Path, default=here / "hub_export_javanese_200k")
    p.add_argument(
        "--repo_id",
        default="thesimonharms/javanese-dataset-180k",
        help="Always a NEW dataset repo — never overwrite an existing dataset.",
    )
    p.add_argument("--num_train", type=int, default=180_000)
    p.add_argument("--num_val", type=int, default=3_000)
    p.add_argument("--short_max_chars", type=int, default=8)
    p.add_argument("--short_fraction", type=float, default=0.15)
    p.add_argument("--train_shards", type=int, default=18)
    p.add_argument("--val_shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_export", action="store_true")
    p.add_argument("--skip_upload", action="store_true")
    p.add_argument(
        "--replace_remote",
        action="store_true",
        help="DANGEROUS: delete existing remote parquet before upload. Off by default.",
    )
    args = p.parse_args()

    train_dir = args.dataset_dir / "train"
    val_dir = args.dataset_dir / "validation"
    train_meta = train_dir / "metadata.jsonl"
    val_meta = val_dir / "metadata.jsonl"
    if not train_meta.is_file():
        sys.exit(f"[ERROR] missing {train_meta}")

    export_data = args.export_dir / "data"
    readme = args.export_dir / "README.md"
    args.export_dir.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "---\n"
        "license: other\n"
        "pretty_name: Javanese TrOCR synthetic 180k (private)\n"
        "---\n\n"
        "Private length-balanced unique synthetic Javanese Aksara lines "
        f"(~{args.num_train} train, ~{args.short_fraction:.0%} short ≤"
        f"{args.short_max_chars} chars). Not for redistribution.\n",
        encoding="utf-8",
    )

    if not args.skip_export:
        print("[INFO] Scanning train metadata…", flush=True)
        train_buckets = scan_metadata(train_meta, args.short_max_chars)
        print(
            "[INFO] train pools: "
            + ", ".join(f"{k}={len(v)}" for k, v in sorted(train_buckets.items())),
            flush=True,
        )
        train_rows = select_balanced(
            train_buckets, args.num_train, args.short_fraction, args.seed
        )
        print("[INFO] Writing train parquet…", flush=True)
        write_parquet_shards(
            train_dir, train_rows, export_data, "train", args.train_shards
        )

        if val_meta.is_file():
            print("[INFO] Scanning validation metadata…", flush=True)
            val_buckets = scan_metadata(val_meta, args.short_max_chars)
            # Keep val short fraction a bit lower so eval isn't short-skewed.
            val_rows = select_balanced(
                val_buckets, args.num_val, min(0.12, args.short_fraction), args.seed + 1
            )
            print("[INFO] Writing validation parquet…", flush=True)
            write_parquet_shards(
                val_dir, val_rows, export_data, "validation", args.val_shards
            )

    if not args.skip_upload:
        upload(args.export_dir, args.repo_id, replace_remote=bool(args.replace_remote))


if __name__ == "__main__":
    main()
