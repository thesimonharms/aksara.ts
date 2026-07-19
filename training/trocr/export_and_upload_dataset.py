#!/usr/bin/env python3
"""Export trocr_dataset_1m to Hub-ready parquet shards, then resumable upload.

This avoids datasets.push_to_hub restarting from scratch on every ConnectionError.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.is_file():
        load_dotenv(env)


def export_shards(dataset_dir: Path, export_dir: Path, train_shards: int, val_shards: int) -> None:
    from datasets import load_dataset

    export_data = export_dir / "data"
    export_data.mkdir(parents=True, exist_ok=True)

    readme = export_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\n"
            "license: other\n"
            "pretty_name: Javanese TrOCR synthetic (private)\n"
            "---\n\n"
            "Private synthetic Javanese Aksara line images for TrOCR fine-tuning. "
            "Not for redistribution.\n",
            encoding="utf-8",
        )

    print("[INFO] Loading train imagefolder (this can take a while)…", flush=True)
    train = load_dataset("imagefolder", data_dir=str(dataset_dir / "train"), split="train")
    print(f"[INFO] train={len(train)} — writing {train_shards} parquet shards…", flush=True)
    for i in range(train_shards):
        out = export_data / f"train-{i:05d}-of-{train_shards:05d}.parquet"
        if out.exists() and out.stat().st_size > 0:
            print(f"  skip existing {out.name}", flush=True)
            continue
        shard = train.shard(num_shards=train_shards, index=i, contiguous=True)
        shard.to_parquet(str(out))
        print(f"  wrote {out.name} rows={len(shard)}", flush=True)

    print("[INFO] Loading validation imagefolder…", flush=True)
    val = load_dataset("imagefolder", data_dir=str(dataset_dir / "validation"), split="train")
    print(f"[INFO] validation={len(val)} — writing {val_shards} parquet shards…", flush=True)
    for i in range(val_shards):
        out = export_data / f"validation-{i:05d}-of-{val_shards:05d}.parquet"
        if out.exists() and out.stat().st_size > 0:
            print(f"  skip existing {out.name}", flush=True)
            continue
        shard = val.shard(num_shards=val_shards, index=i, contiguous=True)
        shard.to_parquet(str(out))
        print(f"  wrote {out.name} rows={len(shard)}", flush=True)


def upload(export_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"[INFO] upload_large_folder -> {repo_id} (resumable, private)", flush=True)

    # upload_large_folder resumes incomplete uploads across runs
    api.upload_large_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=2,
        print_report_every=10,
    )
    print(f"[OK] https://huggingface.co/datasets/{repo_id} (private)", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", type=Path, default=here / "trocr_dataset_1m")
    p.add_argument("--export_dir", type=Path, default=here / "hub_export_javanese")
    p.add_argument("--repo_id", default="thesimonharms/javanese-dataset")
    p.add_argument("--train_shards", type=int, default=50)
    p.add_argument("--val_shards", type=int, default=1)
    p.add_argument("--skip_export", action="store_true")
    p.add_argument("--skip_upload", action="store_true")
    args = p.parse_args()

    if not args.skip_export:
        export_shards(args.dataset_dir, args.export_dir, args.train_shards, args.val_shards)
    if not args.skip_upload:
        upload(args.export_dir, args.repo_id)


if __name__ == "__main__":
    main()
