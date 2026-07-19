#!/usr/bin/env python3
"""
generate_trocr_dataset.py — Synthetic dataset generator for Hugging Face TrOCR fine-tuning.

Generates Javanese Aksara line images and formats them as a Hugging Face `imagefolder`
dataset with `metadata.jsonl` files suitable for AutoTrain Advanced fine-tuning of
`microsoft/trocr-base-handwritten`.

Directory layout created:
  <output_dir>/
    train/
      sample_000000.png
      ...
      metadata.jsonl
    validation/
      sample_000000.png
      ...
      metadata.jsonl

Usage:
  python generate_trocr_dataset.py --corpus ../javanese_corpus_clean.txt --num_train 5000 --num_val 500
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


DEFAULT_JAPANESE_FALLBACK_TEXT = [
    "ꦲꦤꦕꦫꦏ",
    "ꦢꦠꦱꦮꦭ",
    "ꦥꦝꦗꦪꦚ",
    "ꦩꦒꦧꦛꦔ",
    "ꦲꦏ꧀ꦱꦫꦗꦮ",
    "ꦤꦸꦭꦶꦱ꧀ꦲꦏ꧀ꦱꦫ",
]


class TrOCRDatasetGenerator:
    def __init__(
        self,
        corpus_path: Optional[Path],
        fonts_dir: Path,
        pdfs_dir: Path,
        image_height: int = 64,
    ):
        self.image_height = image_height
        self.corpus = self._load_corpus(corpus_path)
        self.fonts = self._discover_fonts(fonts_dir)
        self.backgrounds = self._load_pdf_backgrounds(pdfs_dir)

    def _load_corpus(self, path: Optional[Path]) -> List[str]:
        if path and path.exists():
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            # OCR-friendly: short, non-empty; reject obvious leftover markup
            cleaned = []
            for l in lines:
                if not l or len(l) < 2:
                    continue
                if "[[" in l or "]]" in l or "<" in l or ">" in l or "{{" in l:
                    continue
                if len(l) > 64:
                    l = l[:64]
                cleaned.append(l)
            if cleaned:
                print(f"Loaded {len(cleaned)} corpus lines from {path}")
                return cleaned
        print("Warning: Using built-in fallback Javanese corpus lines.")
        return DEFAULT_JAPANESE_FALLBACK_TEXT

    def _discover_fonts(self, fonts_dir: Path) -> List[Path]:
        fonts = []
        if fonts_dir.exists():
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                fonts.extend(fonts_dir.glob(ext))
        if not fonts:
            print(f"Warning: No TTF/OTF font files found in {fonts_dir}. Will fall back to default PIL font.")
        else:
            print(f"Discovered {len(fonts)} font file(s) in {fonts_dir}")
        return fonts

    def _load_pdf_backgrounds(self, pdfs_dir: Path) -> List[Image.Image]:
        bg_images = []
        if not pdfs_dir.exists():
            return bg_images

        exts = ("*.pdf", "*.PDF", "*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
        files = sorted({p.resolve() for ext in exts for p in pdfs_dir.glob(ext)})

        for file_path in files:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                if not HAS_PYMUPDF:
                    continue
                try:
                    doc = fitz.open(file_path)
                    for page_idx in range(min(5, len(doc))):
                        page = doc[page_idx]
                        pix = page.get_pixmap(dpi=150)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        bg_images.append(img)
                    doc.close()
                except Exception as exc:
                    print(f"Could not load background PDF {file_path}: {exc}")
            elif ext in (".png", ".jpg", ".jpeg"):
                try:
                    img = Image.open(file_path).convert("RGB")
                    bg_images.append(img)
                except Exception as exc:
                    print(f"Could not load background image {file_path}: {exc}")

        if bg_images:
            print(f"Loaded {len(bg_images)} background page(s)/image(s) from {pdfs_dir}")
        return bg_images

    _load_backgrounds = _load_pdf_backgrounds

    def _get_font(self, size: int):
        if not self.fonts:
            return ImageFont.load_default()
        font_path = random.choice(self.fonts)
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except Exception:
            return ImageFont.load_default()

    def _create_background(self, width: int, height: int) -> Image.Image:
        if self.backgrounds and random.random() < 0.6:
            page = random.choice(self.backgrounds)
            if page.width > width and page.height > height:
                x0 = random.randint(0, page.width - width)
                y0 = random.randint(0, page.height - height)
                crop = page.crop((x0, y0, x0 + width, y0 + height))
                return crop

        # Generate textured aged paper background
        base_color = random.choice([
            (255, 255, 255),
            (250, 248, 240),
            (245, 240, 230),
            (238, 232, 218),
        ])
        img = Image.new("RGB", (width, height), color=base_color)
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, random.uniform(1.0, 5.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _apply_augmentations(self, img: Image.Image) -> Image.Image:
        # Slight rotation
        if random.random() < 0.5:
            angle = random.uniform(-2.5, 2.5)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))

        # Blur
        if random.random() < 0.4:
            radius = random.uniform(0.3, 1.1)
            img = img.filter(ImageFilter.GaussianBlur(radius))

        # Contrast / Brightness adjustment
        if random.random() < 0.5:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.7, 1.3))

        if random.random() < 0.5:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        return img

    def render_sample(self, text: str) -> Image.Image:
        font_size = random.randint(30, 48)
        font = self._get_font(font_size)

        # Measure text
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = max(32, bbox[2] - bbox[0] + random.randint(16, 40))
        height = self.image_height

        bg = self._create_background(text_width, height)
        draw = ImageDraw.Draw(bg)

        # Random ink shade
        ink = random.choice([
            (10, 10, 10),
            (25, 20, 20),
            (40, 35, 30),
            (15, 25, 35),
        ])
        y_offset = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
        x_offset = random.randint(8, 20)
        draw.text((x_offset, y_offset), text, font=font, fill=ink)

        return self._apply_augmentations(bg)

    def generate_split(self, output_dir: Path, split_name: str, count: int):
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = split_dir / "metadata.jsonl"

        print(f"Generating {count} samples for '{split_name}' split -> {split_dir} ...")
        with metadata_path.open("w", encoding="utf-8") as f:
            for idx in range(count):
                text = random.choice(self.corpus)
                file_name = f"sample_{idx:06d}.png"
                img = self.render_sample(text)
                img.save(split_dir / file_name)

                record = {
                    "file_name": file_name,
                    "text": text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if (idx + 1) % 500 == 0 or idx + 1 == count:
                    print(f"  [{split_name}] {idx + 1}/{count} samples saved.")


def main():
    parser = argparse.ArgumentParser(description="Generate Javanese TrOCR dataset formatted for AutoTrain Advanced.")
    parser.add_argument("--corpus", type=Path, default=Path("../javanese_corpus_ocr.txt"),
                        help="Text corpus path (prefer cleaned OCR corpus).")
    parser.add_argument("--fonts_dir", type=Path, default=Path("../fonts"), help="Directory containing TTF/OTF fonts.")
    parser.add_argument("--pdfs_dir", "--backgrounds_dir", "--images_dir",
                        dest="pdfs_dir", type=Path, default=Path("../pdfs"),
                        help="Directory containing background PDFs or manuscript images (PNG/JPG).")
    parser.add_argument("--output_dir", type=Path, default=Path("../trocr_dataset"), help="Output directory for dataset.")
    parser.add_argument("--num_train", type=int, default=5000, help="Number of training samples.")
    parser.add_argument("--num_val", type=int, default=500, help="Number of validation samples.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    generator = TrOCRDatasetGenerator(
        corpus_path=args.corpus,
        fonts_dir=args.fonts_dir,
        pdfs_dir=args.pdfs_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator.generate_split(args.output_dir, "train", args.num_train)
    generator.generate_split(args.output_dir, "validation", args.num_val)
    print(f"\nDataset generation complete! Ready for Hugging Face AutoTrain Advanced at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
