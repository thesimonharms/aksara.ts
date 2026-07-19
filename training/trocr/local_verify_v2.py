#!/usr/bin/env python3
"""Local smoke verify for trocr-javanese-synthetic-v2 on synthetic Aksara images."""

from __future__ import annotations

import os
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from device_utils import pick_device

try:
    import editdistance
except ImportError:
    editdistance = None

REPO_ID = os.environ.get("HUB_MODEL_ID", "thesimonharms/trocr-javanese-synthetic-v2")
FONT_URL = (
    "https://github.com/notofonts/javanese/releases/download/"
    "NotoSansJavanese-v2.005/NotoSansJavanese-v2.005.zip"
)
FONT_SUBPATH = "NotoSansJavanese/googlefonts/ttf/NotoSansJavanese-Regular.ttf"
WORK_DIR = Path.home() / "tmp" / "trocr-test"

SAMPLES = [
    ("basa jawa", "ꦧꦱ ꦗꦮ"),
    ("Kabeh panganan", "ꦏꦧꦼꦃ ꦥꦔꦤꦤ꧀"),
    ("tulis aksara jawa", "ꦠꦸꦭꦶꦱ꧀ ꦲꦏ꧀ꦱꦫ ꦗꦮ"),
    ("sinau basa jawa", "ꦱꦶꦤꦲꦸ ꦧꦱ ꦗꦮ"),
    ("hanacaraka", "ꦲꦤꦕꦫꦏ"),
    ("a", "ꦲ"),
]


def normalize(s: str) -> str:
    return "".join(s.split())


def cer(pred: str, ref: str) -> float:
    a, b = normalize(pred), normalize(ref)
    if not b:
        return 0.0 if not a else 1.0
    if editdistance is not None:
        return editdistance.distance(a, b) / max(1, len(b))
    # fallback Levenshtein
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[-1] / max(1, len(b))


def download_font() -> Path:
    extract_dir = WORK_DIR / "noto-javanese"
    path = extract_dir / FONT_SUBPATH
    if path.is_file():
        return path
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = WORK_DIR / "noto-javanese.zip"
    print(f"Downloading font …")
    urllib.request.urlretrieve(FONT_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    return path


def render_aksara(text: str, font_path: Path, width: int = 600, height: int = 160) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    size = 64
    while size > 24:
        font = ImageFont.truetype(str(font_path), size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) < width - 60 and (bbox[3] - bbox[1]) < height - 40:
            break
        size -= 4
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(0, 0, 0))
    return img


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    font_path = download_font()
    device = pick_device()
    print(f"Loading {REPO_ID} on {device} …")
    processor = TrOCRProcessor.from_pretrained(REPO_ID, token=token)
    model = VisionEncoderDecoderModel.from_pretrained(REPO_ID, token=token).to(device)
    model.eval()
    cls = processor.tokenizer.cls_token_id
    model.config.decoder_start_token_id = cls
    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = cls
        model.generation_config.no_repeat_ngram_size = 0
    print(
        f"vocab={len(processor.tokenizer)} "
        f"decoder_start={model.generation_config.decoder_start_token_id}"
    )

    rows = []
    with torch.inference_mode():
        for latin, truth in SAMPLES:
            img = render_aksara(truth, font_path)
            pv = processor(images=img, return_tensors="pt").pixel_values.to(device)
            ids = model.generate(
                pv,
                max_new_tokens=64,
                num_beams=1,
                do_sample=False,
                decoder_start_token_id=cls,
                no_repeat_ngram_size=0,
            )
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            score = cer(pred, truth)
            rows.append((latin, truth, pred, score))
            print(f"--- {latin}  CER={score:.1%}")
            print(f"  REF : {truth}")
            print(f"  PRED: {pred}")

    mean = sum(r[3] for r in rows) / len(rows)
    print(f"\nAverage CER ({len(rows)} samples): {mean:.1%}")


if __name__ == "__main__":
    main()
