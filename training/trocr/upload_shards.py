#!/usr/bin/env python3
"""Upload hub_export_javanese/ file-by-file with retries (resumable)."""

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

# Force hf_transfer when available (faster multipart LFS uploads).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

LOG = Path(__file__).resolve().parent / "upload_shards.progress.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main() -> None:
    from huggingface_hub import HfApi

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--export_dir", type=Path, default=here / "hub_export_javanese")
    p.add_argument("--repo_id", default="thesimonharms/javanese-dataset")
    p.add_argument("--max_retries", type=int, default=12)
    p.add_argument(
        "--delete_stale",
        action="store_true",
        default=True,
        help="Delete old tiny shards (train-*-of-00002) after upload.",
    )
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset")

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)

    try:
        api.update_repo_visibility(args.repo_id, private=True, repo_type="dataset")
    except Exception as exc:
        log(f"[WARN] visibility update: {exc}")

    files: list[Path] = []
    readme = args.export_dir / "README.md"
    if readme.exists():
        files.append(readme)
    data = sorted((args.export_dir / "data").glob("*.parquet"))
    files.extend(data)
    log(f"[INFO] uploading {len(files)} files to {args.repo_id} (private)")

    # Batch size check once to avoid N round-trips.
    remote_files = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    dests = []
    for path in files:
        dest = "README.md" if path.name == "README.md" else f"data/{path.name}"
        dests.append((path, dest))

    size_map: dict[str, int] = {}
    existing = [d for _, d in dests if d in remote_files]
    if existing:
        for info in api.get_paths_info(args.repo_id, existing, repo_type="dataset"):
            size_map[info.path] = int(info.size or 0)

    for i, (path, dest) in enumerate(dests, 1):
        local_size = path.stat().st_size
        remote_size = size_map.get(dest, 0)
        if dest in remote_files and remote_size == local_size and local_size > 0:
            log(f"[{i}/{len(dests)}] skip {dest} ({local_size} bytes)")
            continue
        if dest in remote_files:
            log(f"[{i}/{len(dests)}] replace {dest} remote={remote_size} local={local_size}")
        else:
            log(f"[{i}/{len(dests)}] upload {dest} ({local_size / 1e9:.2f} GB)")

        last_err: Exception | None = None
        t0 = time.time()
        for attempt in range(1, args.max_retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=dest,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
                dt = time.time() - t0
                mbps = (local_size / 1e6) / dt if dt > 0 else 0
                log(f"  OK {dest} in {dt:.0f}s ({mbps:.1f} MB/s)")
                remote_files.add(dest)
                size_map[dest] = local_size
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                wait = min(180, 15 * attempt)
                log(f"  attempt {attempt}/{args.max_retries} failed: {exc!r}; sleep {wait}s")
                time.sleep(wait)
        if last_err is not None:
            raise RuntimeError(f"Failed to upload {dest}") from last_err

    if args.delete_stale:
        stale = [
            f
            for f in remote_files
            if f.startswith("data/train-") and "-of-00002.parquet" in f
        ]
        if stale:
            log(f"[INFO] deleting stale shards: {stale}")
            api.delete_files(
                stale,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message="Remove old tiny train shards",
            )

    log(f"[OK] https://huggingface.co/datasets/{args.repo_id} (private)")


if __name__ == "__main__":
    main()
