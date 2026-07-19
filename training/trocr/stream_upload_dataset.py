#!/usr/bin/env python3
"""Stream local PNG+TXT samples into parquet shards, then resumable Hub upload.

Does NOT use datasets.load_dataset('imagefolder') — that scan is too slow at 1M files.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.is_file():
        load_dotenv(env)


def iter_samples(split_dir: Path, count: int):
    for i in range(count):
        stem = split_dir / f"sample_{i:07d}"
        png = stem.with_suffix(".png")
        txt = stem.with_suffix(".txt")
        if not png.exists() or not txt.exists():
            raise FileNotFoundError(f"Missing sample pair for index {i}: {png.name} / {txt.name}")
        yield {
            "image": {"bytes": png.read_bytes(), "path": png.name},
            "text": txt.read_text(encoding="utf-8"),
        }


def write_split_parquets(split_dir: Path, out_dir: Path, split: str, count: int, num_shards: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    per = (count + num_shards - 1) // num_shards
    print(f"[INFO] {split}: {count} samples -> {num_shards} shards (~{per}/shard)", flush=True)

    idx = 0
    for shard_i in range(num_shards):
        out = out_dir / f"{split}-{shard_i:05d}-of-{num_shards:05d}.parquet"
        start = shard_i * per
        end = min(count, start + per)
        if start >= count:
            break
        if out.exists() and out.stat().st_size > 10_000:
            print(f"  skip {out.name}", flush=True)
            idx = end
            continue

        images_bytes = []
        images_path = []
        texts = []
        for i in range(start, end):
            stem = split_dir / f"sample_{i:07d}"
            images_bytes.append(stem.with_suffix(".png").read_bytes())
            images_path.append(stem.with_suffix(".png").name)
            texts.append(stem.with_suffix(".txt").read_text(encoding="utf-8"))
            if (i - start + 1) % 2000 == 0:
                print(f"  [{split} shard {shard_i}] {i - start + 1}/{end - start}", flush=True)

        # HF datasets image feature in parquet: struct {bytes, path}
        image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        table = pa.table(
            {
                "image": pa.array(
                    [{"bytes": b, "path": p} for b, p in zip(images_bytes, images_path)],
                    type=image_type,
                ),
                "text": pa.array(texts, type=pa.string()),
            }
        )
        pq.write_table(table, out, compression="zstd")
        print(f"  wrote {out.name} rows={end - start} size={out.stat().st_size/1e6:.1f}MB", flush=True)
        idx = end

    print(f"[OK] {split} export done", flush=True)


def upload(export_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"[INFO] upload_large_folder -> {repo_id}", flush=True)
    api.upload_large_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=2,
        print_report_every=5,
    )
    print(f"[OK] https://huggingface.co/datasets/{repo_id} (private)", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=Path, default=here / "trocr_dataset_1m")
    p.add_argument("--export_dir", type=Path, default=here / "hub_export_javanese")
    p.add_argument("--repo_id", default="thesimonharms/javanese-dataset")
    p.add_argument("--num_train", type=int, default=1_000_000)
    p.add_argument("--num_val", type=int, default=10_000)
    p.add_argument("--train_shards", type=int, default=50)
    p.add_argument("--val_shards", type=int, default=1)
    p.add_argument("--skip_export", action="store_true")
    p.add_argument("--skip_upload", action="store_true")
    args = p.parse_args()

    data_dir = args.export_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    readme = args.export_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\nlicense: other\npretty_name: Javanese TrOCR synthetic (private)\n---\n\n"
            "Private synthetic Javanese Aksara OCR lines. Not for redistribution.\n",
            encoding="utf-8",
        )

    if not args.skip_export:
        write_split_parquets(
            args.dataset_dir / "train", data_dir, "train", args.num_train, args.train_shards
        )
        write_split_parquets(
            args.dataset_dir / "validation", data_dir, "validation", args.num_val, args.val_shards
        )
    if not args.skip_upload:
        upload(args.export_dir, args.repo_id)


if __name__ == "__main__":
    main()
