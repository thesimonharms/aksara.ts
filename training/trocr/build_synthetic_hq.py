#!/usr/bin/env python3
"""Build the synthetic HQ TrOCR dataset (stratified + richer aug + quality gate).

Defaults do NOT push to the Hub. Generation is opt-in via running this script;
use --dry-run to only write the sample plan / manifest.

Example (do not run until ready):
  python corpus_hq_prepare.py
  python build_synthetic_hq.py --dry-run
  python build_synthetic_hq.py --num_train 500000 --num_val 5000 --workers 8
  python build_synthetic_hq.py --skip_generate --export-parquet --push
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

from synthetic_hq.sampler import StratifiedCorpus, load_corpus_jsonl
from synthetic_hq.render import HqRenderer

_WORKER: dict = {}


def _init_worker(fonts_dir: str, pdfs_dir: str, plan_path: str, seed: int) -> None:
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
    _WORKER["renderer"] = HqRenderer(Path(fonts_dir), Path(pdfs_dir))


def _render_one(idx: int) -> tuple[int, bytes, dict]:
    row = _WORKER["plan"][idx]
    renderer: HqRenderer = _WORKER["renderer"]
    img = renderer.render_sample(row["text"], aug_seed=int(row["aug_seed"]))
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    meta = {
        "text": row["text"],
        "text_id": row.get("text_id", ""),
        "bucket": row.get("bucket", ""),
        "rare": bool(row.get("rare")),
        "aug_seed": int(row["aug_seed"]),
        "font": renderer.last_font,
        "bg_id": renderer.last_bg_id,
    }
    return idx, buf.getvalue(), meta


def _write_plan(corpus_jsonl: Path, count: int, seed: int, plan_path: Path) -> None:
    rows = load_corpus_jsonl(corpus_jsonl)
    if not rows:
        raise SystemExit(f"[ERROR] empty corpus metadata: {corpus_jsonl}")
    t0 = time.time()
    sampler = StratifiedCorpus(rows, rng=random.Random(seed))
    specs = sampler.plan(count, seed=seed)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", encoding="utf-8") as f:
        for s in specs:
            f.write(
                json.dumps(
                    {
                        "idx": s.idx,
                        "text": s.text,
                        "text_id": s.text_id,
                        "bucket": s.bucket,
                        "rare": s.rare,
                        "aug_seed": s.aug_seed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        f"[OK] wrote plan {plan_path} ({count} samples) in {time.time() - t0:.1f}s",
        flush=True,
    )


def _existing_complete(split_dir: Path, count: int) -> set[int]:
    """Return indices that already have png+txt.

    Fast path: if first/last samples exist and png count matches, assume contiguous
    complete (avoids 500k× exists() on Windows NTFS). Slow path fills holes.
    """
    if not split_dir.exists() or count <= 0:
        return set()
    first = split_dir / "sample_0000000.png"
    last = split_dir / f"sample_{count - 1:07d}.png"
    if first.exists() and last.exists():
        # Cheap completeness signal — glob is still heavy at 500k, so sample a few middles.
        mids = (count // 4, count // 2, (3 * count) // 4)
        if all((split_dir / f"sample_{i:07d}.png").exists() for i in mids):
            return set(range(count))
    done: set[int] = set()
    for i in range(count):
        stem = split_dir / f"sample_{i:07d}"
        if stem.with_suffix(".png").exists() and stem.with_suffix(".txt").exists():
            done.add(i)
    return done


def _write_metadata(
    split_dir: Path,
    count: int,
    manifest_path: Path,
    *,
    plan_path: Path | None = None,
) -> None:
    """Write imagefolder metadata.jsonl + provenance manifest.

    Prefer the stratified plan for labels (one sequential read) instead of opening
    500k .txt/.meta.json files on NTFS.
    """
    meta_path = split_dir / "metadata.jsonl"
    plan_by_idx: dict[int, dict] | None = None
    if plan_path and plan_path.is_file():
        plan_by_idx = {}
        with plan_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                plan_by_idx[int(row["idx"])] = row
        print(f"[INFO] metadata from plan {plan_path.name}", flush=True)

    t0 = time.time()
    with meta_path.open("w", encoding="utf-8") as mf, manifest_path.open(
        "w", encoding="utf-8"
    ) as man:
        for idx in range(count):
            stem = f"sample_{idx:07d}"
            if plan_by_idx is not None:
                row = plan_by_idx[idx]
                text = row["text"]
                # Plan fields only — skip per-sample .meta.json (500k NTFS opens).
                extra = {
                    "text_id": row.get("text_id", ""),
                    "bucket": row.get("bucket", ""),
                    "rare": bool(row.get("rare")),
                    "aug_seed": int(row.get("aug_seed", 0)),
                }
            else:
                text = (split_dir / f"{stem}.txt").read_text(encoding="utf-8")
                extra = {}
                side = split_dir / f"{stem}.meta.json"
                if side.exists():
                    extra = json.loads(side.read_text(encoding="utf-8"))
            mf.write(
                json.dumps({"file_name": f"{stem}.png", "text": text}, ensure_ascii=False)
                + "\n"
            )
            man.write(
                json.dumps({"idx": idx, "file_name": f"{stem}.png", **extra}, ensure_ascii=False)
                + "\n"
            )
            if idx and idx % 50_000 == 0:
                print(f"  [metadata] {idx}/{count}", flush=True)
    print(
        f"[OK] metadata {meta_path.name} + {manifest_path.name} in {time.time() - t0:.1f}s",
        flush=True,
    )


def generate_split(
    *,
    fonts_dir: Path,
    pdfs_dir: Path,
    plan_path: Path,
    split_dir: Path,
    count: int,
    workers: int,
    seed: int,
    chunk_size: int = 2000,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    already = _existing_complete(split_dir, count)
    todo = [i for i in range(count) if i not in already]
    print(
        f"[INFO] render {split_dir.name}: {count} "
        f"(skip={len(already)} todo={len(todo)} workers={workers})",
        flush=True,
    )
    if not todo:
        _write_metadata(
            split_dir, count, split_dir / "manifest.jsonl", plan_path=plan_path
        )
        return

    t0 = time.time()
    done = len(already)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(fonts_dir), str(pdfs_dir), str(plan_path), seed),
    ) as pool:
        for start in range(0, len(todo), chunk_size):
            batch = todo[start : start + chunk_size]
            futs = [pool.submit(_render_one, i) for i in batch]
            for fut in as_completed(futs):
                idx, png, meta = fut.result()
                stem = split_dir / f"sample_{idx:07d}"
                stem.with_suffix(".png").write_bytes(png)
                stem.with_suffix(".txt").write_text(meta["text"], encoding="utf-8")
                (split_dir / f"sample_{idx:07d}.meta.json").write_text(
                    json.dumps(meta, ensure_ascii=False), encoding="utf-8"
                )
                done += 1
                if done % 2000 == 0 or done == count:
                    newly = done - len(already)
                    rate = newly / max(time.time() - t0, 1e-6)
                    eta = (count - done) / max(rate, 1e-6)
                    print(
                        f"  [{split_dir.name}] {done}/{count} "
                        f"({rate:.1f}/s ETA {eta/60:.1f}m)",
                        flush=True,
                    )

    _write_metadata(
        split_dir, count, split_dir / "manifest.jsonl", plan_path=plan_path
    )
    print(f"[OK] {split_dir.name} in {(time.time()-t0)/60:.1f} min", flush=True)


def export_parquet(
    dataset_dir: Path,
    export_dir: Path,
    num_train: int,
    num_val: int,
    train_shards: int,
    val_shards: int,
) -> None:
    from stream_upload_dataset import write_split_parquets

    data_dir = export_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    readme = export_dir / "README.md"
    readme.write_text(
        "---\n"
        "license: other\n"
        "pretty_name: Javanese synthetic HQ OCR (private)\n"
        "---\n\n"
        "# javanese-synthetic-hq\n\n"
        "Private synthetic Javanese Aksara OCR line images.\n\n"
        "- Stratified length quotas (15% short / 55% mid / 30% long)\n"
        "- Held-out text val pool (hash split)\n"
        "- Richer degrade augments + quality gate\n"
        "- Font set exhausted; variety is layout/degrade/length\n\n"
        "Not for redistribution.\n",
        encoding="utf-8",
    )
    write_split_parquets(
        dataset_dir / "train", data_dir, "train", num_train, train_shards
    )
    write_split_parquets(
        dataset_dir / "validation", data_dir, "validation", num_val, val_shards
    )


def push_private(export_dir: Path, repo_id: str) -> None:
    from stream_upload_dataset import upload

    upload(export_dir, repo_id)


def main() -> None:
    here = Path(__file__).resolve().parent
    training = here.parent
    p = argparse.ArgumentParser(description="Build synthetic HQ Javanese OCR dataset.")
    p.add_argument(
        "--corpus-train",
        type=Path,
        default=training / "corpus_hq" / "corpus_hq_train.jsonl",
    )
    p.add_argument(
        "--corpus-val",
        type=Path,
        default=training / "corpus_hq" / "corpus_hq_val.jsonl",
    )
    p.add_argument("--fonts_dir", type=Path, default=training / "fonts")
    p.add_argument("--pdfs_dir", type=Path, default=training / "pdfs")
    p.add_argument("--output_dir", type=Path, default=here / "trocr_dataset_hq")
    p.add_argument("--num_train", type=int, default=500_000)
    p.add_argument("--num_val", type=int, default=5_000)
    p.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write stratified sample plans; do not render.",
    )
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--export-parquet", action="store_true")
    p.add_argument("--train_shards", type=int, default=50)
    p.add_argument("--val_shards", type=int, default=5)
    p.add_argument(
        "--export_dir",
        type=Path,
        default=here / "hub_export_javanese_synthetic_hq",
    )
    p.add_argument(
        "--push",
        action="store_true",
        help="Upload parquet export to a PRIVATE Hub dataset (off by default).",
    )
    p.add_argument("--repo_id", default="thesimonharms/javanese-synthetic-hq")
    p.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Kept for CLI clarity; uploads are always private.",
    )
    args = p.parse_args()

    for label, path in (("train", args.corpus_train), ("val", args.corpus_val)):
        if not path.is_file():
            sys.exit(
                f"[ERROR] missing {label} corpus metadata: {path}\n"
                f"        Run: python corpus_hq_prepare.py"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_plan = args.output_dir / "train_plan.jsonl"
    val_plan = args.output_dir / "val_plan.jsonl"

    print("[INFO] writing stratified plans …", flush=True)
    _write_plan(args.corpus_train, args.num_train, args.seed, train_plan)
    _write_plan(args.corpus_val, args.num_val, args.seed + 1, val_plan)

    if args.dry_run:
        print("[OK] dry-run complete (plans only; no render / no push)", flush=True)
        return

    if not args.skip_generate:
        generate_split(
            fonts_dir=args.fonts_dir,
            pdfs_dir=args.pdfs_dir,
            plan_path=train_plan,
            split_dir=args.output_dir / "train",
            count=args.num_train,
            workers=args.workers,
            seed=args.seed,
        )
        generate_split(
            fonts_dir=args.fonts_dir,
            pdfs_dir=args.pdfs_dir,
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
            args.num_train,
            args.num_val,
            args.train_shards,
            args.val_shards,
        )

    if args.push:
        print(f"[INFO] private Hub push -> {args.repo_id}", flush=True)
        push_private(args.export_dir, args.repo_id)
    else:
        print("[INFO] Hub push skipped (pass --push when ready)", flush=True)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
