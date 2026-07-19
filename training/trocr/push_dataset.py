#!/usr/bin/env python3
"""Push local trocr_dataset/ imagefolder to the Hugging Face Hub.

Usage:
  python push_dataset.py
  python push_dataset.py --dataset_dir trocr_dataset --repo_id thesimonharms/javanese-dataset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    _file = Path(__file__).resolve()
    if len(_file.parents) > 2:
        _env = _file.parents[2] / ".env"
        if _env.is_file():
            load_dotenv(_env)

from datasets import DatasetDict, load_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Push trocr imagefolder dataset to HF Hub.")
    p.add_argument("--dataset_dir", type=Path, default=Path(__file__).resolve().parent / "trocr_dataset")
    p.add_argument("--repo_id", default=None, help="Hub id (default: {HF_USERNAME}/javanese-dataset)")
    p.add_argument(
        "--public",
        action="store_true",
        help="Make the dataset public (default: private — do not redistribute licensed scans).",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset — set it in ../../.env or the environment.")

    user = os.environ.get("HF_USERNAME")
    if not user:
        try:
            from huggingface_hub import whoami
            user = whoami(token=token).get("name")
        except Exception as exc:
            sys.exit(f"[ERROR] Cannot resolve username: {exc}")

    repo_id = args.repo_id or f"{user}/javanese-dataset"
    ddir = args.dataset_dir
    if not (ddir / "train").exists() or not (ddir / "validation").exists():
        sys.exit(f"[ERROR] Expected {ddir}/train and {ddir}/validation (imagefolder).")

    private = not args.public
    print(f"[INFO] Loading {ddir} …")
    train = load_dataset("imagefolder", data_dir=str(ddir / "train"), split="train")
    val = load_dataset("imagefolder", data_dir=str(ddir / "validation"), split="train")
    raw = DatasetDict({"train": train, "validation": val})
    vis = "private" if private else "public"
    print(f"[INFO] train={len(train)} validation={len(val)} -> pushing {vis} to {repo_id}")
    raw.push_to_hub(repo_id, token=token, private=private)
    print(f"[OK] Dataset live at https://huggingface.co/datasets/{repo_id} ({vis})")


if __name__ == "__main__":
    main()
