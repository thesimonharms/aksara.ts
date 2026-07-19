#!/usr/bin/env python3
"""Inspect NusaAksara OCR split and build a private image+text Hub dataset."""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.is_file():
        load_dotenv(env)


def aksara_count(s: str) -> int:
    return sum(1 for c in (s or "") if 0xA980 <= ord(c) <= 0xA9DF)


def inspect() -> None:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ocr = load_dataset(
        "NusaAksara/NusaAksara",
        "Image Transcription (OCR)",
        split="train",
        token=token,
    )
    print(f"[INFO] OCR rows={len(ocr)} cols={ocr.column_names}")
    scripts = Counter(ocr["script"])
    print("[INFO] scripts:", dict(scripts))
    for i in range(min(3, len(ocr))):
        ex =ocr[i]
        img = ex["image"]
        print(f"--- sample {i} script={ex['script']!r}")
        print(f"  transcription={ex['transcription'][:100]!r}")
        print(f"  image type={type(img).__name__} preview={str(img)[:160]!r}")

    for script, n in scripts.most_common():
        subset = [t for t, s in zip(ocr["transcription"], ocr["script"]) if s == script]
        aks_rows = sum(1 for t in subset if aksara_count(t) > 0)
        sample = subset[0][:80] if subset else ""
        print(f"  script={script!r} n={n} aksara_rows={aks_rows} sample={sample!r}")

    exq = load_dataset("Exqrch/NusaAksara-java", split="train", token=token)
    print(f"[INFO] Exqrch rows={len(exq)} cols={exq.column_names}")
    print("[INFO] Exqrch features:", exq.features)
    for i in range(min(2, len(exq))):
        ex = exq[i]
        for k, v in ex.items():
            if isinstance(v, str):
                print(f"  [{i}] {k}={v[:100]!r}")
            else:
                print(f"  [{i}] {k}={type(v).__name__}")


def _decode_image_field(raw):
    """NusaAksara may store image as URL, path, data-URI, or PIL/dict."""
    from PIL import Image

    if raw is None:
        return None
    if hasattr(raw, "convert"):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        if raw.get("bytes"):
            return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        if raw.get("path"):
            return Image.open(raw["path"]).convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("data:image"):
            b64 = s.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        if s.startswith("http://") or s.startswith("https://"):
            req = Request(s, headers={"User-Agent": "aksara-dataset-builder/1.0"})
            with urlopen(req, timeout=60) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        p = Path(s)
        if p.is_file():
            return Image.open(p).convert("RGB")
        # Sometimes HF stores relative repo paths — try hub download
        return None
    return None


def build_and_upload(
    repo_id: str,
    export_dir: Path,
    min_aksara: int,
    max_chars: int,
    seed: int,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from PIL import Image

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("[ERROR] HF_TOKEN unset")

    ocr = load_dataset(
        "NusaAksara/NusaAksara",
        "Image Transcription (OCR)",
        split="train",
        token=token,
    )

    # Prefer explicit Jawa script; also keep any row with enough aksara.
    jawa_names = {
        s
        for s in set(ocr["script"])
        if re.search(r"jaw|java|hanac|carak", s or "", re.I)
    }
    print(f"[INFO] Javanese-like script labels: {sorted(jawa_names)}")

    rows_meta = []
    for i in range(len(ocr)):
        ex = ocr[i]
        text = (ex.get("transcription") or "").strip()
        script = ex.get("script") or ""
        if not text:
            continue
        if script not in jawa_names:
            continue
        if aksara_count(text) < min_aksara:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]
        rows_meta.append({"i": i, "text": text, "script": script, "image": ex["image"]})

    print(f"[INFO] Kept {len(rows_meta)} / {len(ocr)} OCR rows after Jawa+aksara filter")
    if not rows_meta:
        sys.exit("[ERROR] No usable Javanese OCR rows")

    # Exqrch/NusaAksara-java is segmentation boxes only (no transcription) — skip.
    export_dir.mkdir(parents=True, exist_ok=True)
    data_dir = export_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "README.md").write_text(
        "---\n"
        "license: other\n"
        "pretty_name: Javanese NusaAksara OCR (private derived)\n"
        "---\n\n"
        "Private derived subset of NusaAksara Image Transcription (OCR) "
        "(+ optional Exqrch/NusaAksara-java blocks) filtered to Javanese script "
        "with Unicode Aksara labels, reformatted as `image` + `text` for TrOCR.\n\n"
        "Source: https://huggingface.co/datasets/NusaAksara/NusaAksara "
        "(cite the NusaAksara paper). Not for redistribution beyond private training.\n",
        encoding="utf-8",
    )

    images: list[dict] = []
    texts: list[str] = []
    failed = 0
    t0 = time.time()
    for k, row in enumerate(rows_meta):
        try:
            img = _decode_image_field(row["image"])
            if img is None and isinstance(row["image"], str):
                # Hub-relative path inside NusaAksara repo?
                from huggingface_hub import hf_hub_download

                try:
                    local = hf_hub_download(
                        "NusaAksara/NusaAksara",
                        row["image"],
                        repo_type="dataset",
                        token=token,
                    )
                    img = Image.open(local).convert("RGB")
                except Exception:
                    img = None
            if img is None:
                failed += 1
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            images.append({"bytes": buf.getvalue()})
            texts.append(row["text"])
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print(f"[WARN] decode fail {row['i']}: {exc}", flush=True)
        if (k + 1) % 200 == 0 or k + 1 == len(rows_meta):
            print(
                f"  decoded {len(images)}/{k+1} (failed={failed}) "
                f"{(k+1)/(time.time()-t0):.1f}/s",
                flush=True,
            )

    if len(images) < 50:
        sys.exit(f"[ERROR] Too few decoded images ({len(images)}); aborting upload")

    # Train/val split
    import random

    rng = random.Random(seed)
    order = list(range(len(images)))
    rng.shuffle(order)
    n_val = max(50, min(500, len(order) // 20))
    val_idx = set(order[:n_val])
    train_img, train_txt, val_img, val_txt = [], [], [], []
    for i in range(len(images)):
        if i in val_idx:
            val_img.append(images[i])
            val_txt.append(texts[i])
        else:
            train_img.append(images[i])
            train_txt.append(texts[i])

    def write_split(name: str, imgs: list, txts: list) -> None:
        table = pa.table(
            {
                "image": pa.array(imgs),
                "text": pa.array(txts, type=pa.string()),
            }
        )
        out = data_dir / f"{name}-00000-of-00001.parquet"
        pq.write_table(table, out, compression="zstd")
        print(f"[OK] wrote {out.name} rows={len(txts)} ({out.stat().st_size/1e6:.1f} MB)")

    write_split("train", train_img, train_txt)
    write_split("validation", val_img, val_txt)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"[INFO] Uploading NEW private dataset {repo_id} …", flush=True)
    api.upload_large_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=4,
        print_report_every=5,
    )
    print(f"[OK] https://huggingface.co/datasets/{repo_id} (private)")


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--inspect_only", action="store_true")
    p.add_argument(
        "--repo_id",
        default="thesimonharms/javanese-nusaaksara-ocr",
        help="NEW private dataset repo (never overwrites old ones by reusing names casually).",
    )
    p.add_argument(
        "--export_dir",
        type=Path,
        default=here / "hub_export_nusaaksara_ocr",
    )
    p.add_argument("--min_aksara", type=int, default=1)
    p.add_argument("--max_chars", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.inspect_only:
        inspect()
        return
    build_and_upload(
        args.repo_id, args.export_dir, args.min_aksara, args.max_chars, args.seed
    )


if __name__ == "__main__":
    main()
