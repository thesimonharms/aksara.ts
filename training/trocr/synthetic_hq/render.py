"""HQ renderer: multi-page backgrounds, CRNN-grade degrade, provenance-friendly."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .quality import passes_quality_gate

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class HqRenderer:
    def __init__(
        self,
        fonts_dir: Path,
        pdfs_dir: Path,
        *,
        image_height: int = 64,
        max_pages_per_pdf: int = 50,
        manuscript_bg_prob: float = 0.70,
    ):
        self.image_height = image_height
        self.manuscript_bg_prob = manuscript_bg_prob
        self.fonts = self._discover_fonts(fonts_dir)
        self.backgrounds = self._load_backgrounds(pdfs_dir, max_pages_per_pdf)
        self.last_font: str = ""
        self.last_bg_id: str = "paper"

    @staticmethod
    def _discover_fonts(fonts_dir: Path) -> list[Path]:
        fonts: list[Path] = []
        if fonts_dir.exists():
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                fonts.extend(fonts_dir.glob(ext))
        return fonts

    @staticmethod
    def _load_backgrounds(pdfs_dir: Path, max_pages_per_pdf: int) -> list[tuple[str, Image.Image]]:
        out: list[tuple[str, Image.Image]] = []
        if not pdfs_dir.exists():
            return out
        exts = ("*.pdf", "*.PDF", "*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
        files = sorted({p.resolve() for ext in exts for p in pdfs_dir.glob(ext)})
        for file_path in files:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                if not HAS_PYMUPDF:
                    continue
                try:
                    doc = fitz.open(file_path)
                    n = min(max_pages_per_pdf, len(doc))
                    for page_idx in range(n):
                        page = doc[page_idx]
                        pix = page.get_pixmap(dpi=150)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        out.append((f"{file_path.name}:p{page_idx}", img))
                    doc.close()
                except Exception as exc:
                    print(f"[WARN] background PDF {file_path}: {exc}", flush=True)
            elif ext in (".png", ".jpg", ".jpeg"):
                try:
                    img = Image.open(file_path).convert("RGB")
                    out.append((file_path.name, img))
                except Exception as exc:
                    print(f"[WARN] background image {file_path}: {exc}", flush=True)
        return out

    def _get_font(self, size: int, rng: random.Random) -> ImageFont.ImageFont:
        if not self.fonts:
            self.last_font = "PIL-default"
            return ImageFont.load_default()
        font_path = rng.choice(self.fonts)
        self.last_font = font_path.name
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except Exception:
            self.last_font = "PIL-default"
            return ImageFont.load_default()

    def _paper_background(self, width: int, height: int, rng: random.Random) -> Image.Image:
        self.last_bg_id = "paper"
        base_color = rng.choice(
            [
                (255, 255, 255),
                (250, 248, 240),
                (245, 240, 230),
                (238, 232, 218),
                (232, 220, 200),
            ]
        )
        img = Image.new("RGB", (width, height), color=base_color)
        arr = np.array(img, dtype=np.int16)
        noise = np.random.normal(0, rng.uniform(1.0, 6.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _create_background(self, width: int, height: int, rng: random.Random) -> Image.Image:
        if self.backgrounds and rng.random() < self.manuscript_bg_prob:
            bg_id, page = rng.choice(self.backgrounds)
            if page.width > width and page.height > height:
                x0 = rng.randint(0, page.width - width)
                y0 = rng.randint(0, page.height - height)
                self.last_bg_id = bg_id
                return page.crop((x0, y0, x0 + width, y0 + height))
        return self._paper_background(width, height, rng)

    def _apply_augmentations(self, img: Image.Image, rng: random.Random) -> Image.Image:
        # Mild geometry
        if rng.random() < 0.55:
            angle = rng.uniform(-3.0, 3.0)
            fill = tuple(int(x) for x in np.asarray(img).reshape(-1, 3).mean(axis=0))
            img = img.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=fill,
            )

        # Slight horizontal shear via affine
        if rng.random() < 0.25:
            shear = rng.uniform(-0.08, 0.08)
            w, h = img.size
            img = img.transform(
                (w, h),
                Image.Transform.AFFINE,
                (1, shear, 0, 0, 1, 0),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(255, 255, 255),
            )

        if rng.random() < 0.45:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))

        if rng.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.65, 1.35))
        if rng.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))

        # Gaussian pixel noise
        if rng.random() < 0.40:
            arr = np.asarray(img, dtype=np.float32)
            arr = np.clip(arr + np.random.normal(0, rng.uniform(3.0, 12.0), arr.shape), 0, 255)
            img = Image.fromarray(arr.astype(np.uint8))

        # Salt & pepper
        if rng.random() < 0.20:
            arr = np.asarray(img).copy()
            density = rng.uniform(0.001, 0.008)
            noise = np.random.rand(arr.shape[0], arr.shape[1])
            arr[noise < density] = 0
            arr[noise > 1 - density] = 255
            img = Image.fromarray(arr)

        # Morphological ink bleed / erosion
        if rng.random() < 0.22:
            if rng.random() < 0.5:
                img = img.filter(ImageFilter.MinFilter(3))
            else:
                img = img.filter(ImageFilter.MaxFilter(3))

        # JPEG recompress (scan-like)
        if rng.random() < 0.35:
            buf = io.BytesIO()
            q = rng.randint(35, 85)
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Mild aliasing via downscale/upscale
        if rng.random() < 0.30:
            w, h = img.size
            scale = rng.uniform(0.55, 0.85)
            small = img.resize(
                (max(8, int(w * scale)), max(8, int(h * scale))),
                Image.Resampling.BILINEAR,
            )
            img = small.resize((w, h), Image.Resampling.NEAREST)

        return img

    def render_sample(self, text: str, *, aug_seed: int, max_tries: int = 6) -> Image.Image:
        rng = random.Random(aug_seed)
        # Keep numpy RNG loosely tied for noise ops inside this call.
        np.random.seed(aug_seed & 0x7FFFFFFF)

        last = None
        for attempt in range(max_tries):
            attempt_rng = random.Random(aug_seed + attempt * 9973)
            np.random.seed((aug_seed + attempt * 9973) & 0x7FFFFFFF)

            font_size = attempt_rng.randint(28, 48)
            font = self._get_font(font_size, attempt_rng)

            dummy = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(dummy)
            bbox = draw.textbbox((0, 0), text, font=font)
            pad_x = attempt_rng.randint(12, 36)
            text_width = max(32, bbox[2] - bbox[0] + pad_x)
            height = self.image_height

            bg = self._create_background(text_width, height, attempt_rng)
            draw = ImageDraw.Draw(bg)
            ink = attempt_rng.choice(
                [
                    (8, 8, 8),
                    (18, 14, 12),
                    (32, 28, 24),
                    (12, 20, 28),
                    (45, 35, 30),
                ]
            )
            y_offset = (height - (bbox[3] - bbox[1])) // 2 - bbox[1]
            x_offset = attempt_rng.randint(6, max(7, pad_x // 2))
            draw.text((x_offset, y_offset), text, font=font, fill=ink)

            img = self._apply_augmentations(bg, attempt_rng)
            last = img
            if passes_quality_gate(img):
                return img

        assert last is not None
        return last
