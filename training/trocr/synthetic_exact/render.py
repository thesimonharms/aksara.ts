"""384×384 clean printed aksara renderer (no manuscript backgrounds)."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

CANVAS = 384
PAD = 20


class ExactRenderer:
    def __init__(self, fonts_dir: Path):
        self.fonts = self._discover_fonts(fonts_dir)
        self.last_font = ""
        if not self.fonts:
            raise SystemExit(f"[ERROR] no TTF/OTF fonts in {fonts_dir}")

    @staticmethod
    def _discover_fonts(fonts_dir: Path) -> list[Path]:
        fonts: list[Path] = []
        if fonts_dir.exists():
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                fonts.extend(fonts_dir.glob(ext))
        return sorted({p.resolve() for p in fonts})

    def _fit(self, text: str, rng: random.Random) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
        path = rng.choice(self.fonts)
        self.last_font = path.name
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        max_w = CANVAS - 2 * PAD
        max_h = CANVAS - 2 * PAD
        start = rng.randint(34, 52)
        for size in range(start, 16, -1):
            font = ImageFont.truetype(str(path), size=size)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= max_w and th <= max_h:
                return font, bbox
        font = ImageFont.truetype(str(path), size=18)
        bbox = draw.textbbox((0, 0), text, font=font)
        return font, bbox

    def _paper(self, rng: random.Random) -> Image.Image:
        color = rng.choice(
            [
                (255, 255, 255),
                (252, 250, 244),
                (248, 244, 232),
                (242, 234, 214),
            ]
        )
        img = Image.new("RGB", (CANVAS, CANVAS), color)
        if rng.random() < 0.7:
            arr = np.array(img, dtype=np.int16)
            noise = np.random.normal(0, rng.uniform(0.8, 3.5), arr.shape)
            img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        return img

    def _light_aug(self, img: Image.Image, rng: random.Random) -> Image.Image:
        if rng.random() < 0.12:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 0.45)))
        if rng.random() < 0.12:
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.92, 1.08))
        if rng.random() < 0.10:
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.94, 1.06))
        if rng.random() < 0.10:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=rng.randint(82, 96))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img

    def render_sample(self, text: str, *, aug_seed: int) -> Image.Image:
        rng = random.Random(aug_seed)
        np.random.seed(aug_seed & 0x7FFFFFFF)
        font, bbox = self._fit(text, rng)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        canvas = self._paper(rng)
        draw = ImageDraw.Draw(canvas)
        ink = rng.choice([(12, 12, 12), (22, 18, 14), (8, 10, 16)])
        x = PAD - bbox[0]
        y = (CANVAS - th) // 2 - bbox[1]
        # Keep a little left jitter so the model doesn't memorize x=PAD.
        x += rng.randint(0, 8)
        draw.text((x, y), text, font=font, fill=ink)
        if rng.random() < 0.25:
            canvas = self._light_aug(canvas, rng)
        return canvas
