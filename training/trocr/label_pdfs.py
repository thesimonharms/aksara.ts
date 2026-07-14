#!/usr/bin/env python3
"""
label_pdfs.py — Human-in-the-loop Javanese Aksara manuscript labeler.

Loads every page from --pdfs_dir (PDF pages via PyMuPDF, or PNG/JPG images),
auto-detects candidate text-line strips by horizontal dark-pixel projection, and
shows each page in a local Gradio web UI.

**Draw a box** by clicking two opposite corners on the page preview, or step
through green auto-detected strips with Next/Prev. The active region is shown
with a red overlay; the strip preview updates live. Type the Latin
transliteration, save, repeat.

Multiple strips per page: saving **stays on the same page** by default (the
"Multiple strips on this page" checkbox) — after each save the box resets to
auto-detect so you can adjust for the next line. Uncheck it for a single-strip
page and saving advances to the next page.

Resume-safe: strips already saved are detected on startup and counted per page.

Usage:
    python label_pdfs.py --pdfs_dir ../pdfs --output_dir ../pdf_labeled
    # then open http://127.0.0.1:7861 in your browser
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
try:
    import gradio as gr
except ImportError:
    gr = None


# Display pages at most this wide in the UI (clicks map back to full-res).
_DISPLAY_MAX_W = 1200

# Overlay colors (RGB)
_RED = (220, 40, 40)
_GREEN = (30, 170, 50)
_YELLOW = (240, 200, 40)
_WHITE = (255, 255, 255)


# ---------------------------------------------------------------------------
# Strip detection — horizontal projection profile
# ---------------------------------------------------------------------------
def detect_strips(img: Image.Image,
                  min_strip_h: int = 14,
                  max_strip_h: int = 240,
                  ink_ratio_min: float = 0.004,
                  gap_min: int = 6,
                  bg_radius: int = 25,
                  bg_delta: int = 15) -> List[Tuple[int, int]]:
    """Return list of (y0, y1) candidate line strips on a page.

    Uses **local-contrast** binarization: each pixel is "ink" only where it is
    darker than a heavily-blurred version of the page (which estimates the local
    parchment background). This is robust to aged / dark / unevenly-lit scans
    where a fixed global threshold would mark the whole page as ink.

    Rows are flagged as text rows when their ink fraction exceeds an **adaptive**
    threshold (the larger of `ink_ratio_min` and half the median row density).
    Contiguous text rows are merged across short gaps and filtered by plausible
    line-height bounds.
    """
    gray = img.convert("L")
    arr = np.asarray(gray).astype(np.float32)
    bg = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=bg_radius))).astype(np.float32)
    ink = arr < (bg - bg_delta)
    row_density = ink.mean(axis=1)

    med = float(np.median(row_density)) if row_density.size else 0.0
    row_thr = max(ink_ratio_min, 0.5 * med)
    in_text = row_density > row_thr

    strips: List[Tuple[int, int]] = []
    y0: Optional[int] = None
    for y, t in enumerate(in_text):
        if t and y0 is None:
            y0 = y
        elif not t and y0 is not None:
            if y - y0 >= min_strip_h:
                strips.append((y0, y))
            y0 = None
    if y0 is not None and len(arr) - y0 >= min_strip_h:
        strips.append((y0, len(arr)))

    merged: List[Tuple[int, int]] = []
    for s in strips:
        if merged and s[0] - merged[-1][1] < gap_min:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)

    return [(a, b) for (a, b) in merged if min_strip_h <= (b - a) <= max_strip_h]


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int,
                w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(x0 + 1, min(w, int(x1)))
    y1 = max(y0 + 1, min(h, int(y1)))
    return x0, y0, x1, y1


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str,
                fill: Tuple[int, int, int]) -> None:
    x, y = xy
    # Shadow for readability on parchment
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill)


def _semi_fill(page: Image.Image, box: Tuple[int, int, int, int],
               rgb: Tuple[int, int, int], alpha: float = 0.28) -> None:
    """Alpha-blend a translucent rectangle onto an RGB page (in place)."""
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    overlay = Image.new("RGB", page.size, rgb)
    mask = Image.new("L", page.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=int(255 * alpha))
    blended = Image.composite(overlay, page, mask)
    page.paste(blended)


# ---------------------------------------------------------------------------
# Pages + labeler
# ---------------------------------------------------------------------------
@dataclass
class Page:
    file_key: str          # file stem (stable id for output naming)
    src: Path              # source file (pdf or image)
    idx: int               # page index within the source (0 for plain images)
    img: Image.Image       # RGB page
    strips: List[Tuple[int, int]] = field(default_factory=list)  # auto (y0,y1)


@dataclass
class Labeler:
    pdfs_dir: Path
    output_dir: Path
    dpi: int = 200
    bg_radius: int = 25
    bg_delta: int = 15
    pages: List[Page] = field(default_factory=list)
    page_ptr: int = 0
    # (file_key, page_idx) -> set of saved strip indices
    saved: dict = field(default_factory=dict)
    # Active crop bounds for the current candidate: (x0, y0, x1, y1).
    # None means "use auto-detect" (full page width + first detected strip,
    # or full page if detection found nothing).
    active_bounds: Optional[Tuple[int, int, int, int]] = None
    # First corner while drawing a box (full-res page coords), or None.
    pending_corner: Optional[Tuple[int, int]] = None
    # Index into page.strips for the current auto-suggest (when bounds are auto).
    auto_strip_idx: int = 0
    # Status line shown under the page (drawing hints / save feedback).
    status: str = ""

    # -- loading ------------------------------------------------------------
    def load(self) -> int:
        if not self.pdfs_dir.exists():
            print(f"[WARN] Directory not found: {self.pdfs_dir}", file=sys.stderr)
            return 0
        exts = ("*.pdf", "*.PDF", "*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")
        files = sorted({p.resolve() for ext in exts for p in self.pdfs_dir.glob(ext)})
        if not files:
            print(f"[WARN] No PDFs or manuscript images found in {self.pdfs_dir}", file=sys.stderr)
            return 0

        for fp in files:
            ext = fp.suffix.lower()
            if ext == ".pdf":
                if not HAS_PYMUPDF:
                    print(f"[WARN] PyMuPDF not installed, skipping {fp} "
                          "(install with: pip install pymupdf)", file=sys.stderr)
                    continue
                try:
                    doc = fitz.open(fp)
                except Exception as exc:
                    print(f"[WARN] could not open {fp}: {exc}", file=sys.stderr)
                    continue
                for page_idx in range(len(doc)):
                    page = doc[page_idx]
                    pix = page.get_pixmap(dpi=self.dpi)
                    page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    self._add_page(fp, page_idx, page_img)
                doc.close()
            elif ext in (".png", ".jpg", ".jpeg"):
                try:
                    page_img = Image.open(fp).convert("RGB")
                except Exception as exc:
                    print(f"[WARN] could not open image {fp}: {exc}", file=sys.stderr)
                    continue
                self._add_page(fp, 0, page_img)

        self._load_saved()
        if self.saved:
            for i, pg in enumerate(self.pages):
                if (pg.file_key, pg.idx) in self.saved:
                    self.page_ptr = i
        n_strips = sum(len(p.strips) for p in self.pages)
        print(f"[INFO] {len(self.pages)} page(s), {n_strips} auto-detected strips, "
              f"{sum(len(s) for s in self.saved.values())} already labeled "
              f"(resuming on page {self.page_ptr + 1}).")
        return len(self.pages)

    def _add_page(self, src: Path, page_idx: int, img: Image.Image) -> None:
        strips = detect_strips(img, bg_radius=self.bg_radius, bg_delta=self.bg_delta)
        self.pages.append(Page(file_key=src.stem, src=src, idx=page_idx, img=img, strips=strips))

    # -- saved labels -------------------------------------------------------
    _fname_re = re.compile(r"^(.+)_p(\d+)_s(\d+)\.txt$")

    def _load_saved(self) -> None:
        if not self.output_dir.exists():
            return
        for p in self.output_dir.glob("*.txt"):
            m = self._fname_re.match(p.name)
            if m:
                fk, pi, si = m.group(1), int(m.group(2)), int(m.group(3))
                self.saved.setdefault((fk, pi), set()).add(si)

    def _saved_on_page(self, p: Page) -> int:
        return len(self.saved.get((p.file_key, p.idx), set()))

    def _next_strip_id(self, p: Page) -> int:
        existing = self.saved.get((p.file_key, p.idx), set())
        return max(existing) + 1 if existing else 0

    def _png_path(self, p: Page, strip: int) -> Path:
        return self.output_dir / f"{p.file_key}_p{p.idx}_s{strip}.png"

    def _txt_path(self, p: Page, strip: int) -> Path:
        return self.output_dir / f"{p.file_key}_p{p.idx}_s{strip}.txt"

    # -- bounds -------------------------------------------------------------
    def _auto_bbox(self, p: Page) -> Tuple[int, int, int, int]:
        """Default bbox: current auto-detected strip, or full page if none."""
        if p.strips:
            i = max(0, min(self.auto_strip_idx, len(p.strips) - 1))
            y0, y1 = p.strips[i]
            return (0, max(0, y0 - 3), p.img.width, min(p.img.height, y1 + 3))
        return (0, 0, p.img.width, p.img.height)

    def _active_bbox(self) -> Tuple[int, int, int, int]:
        c = self.pages[self.page_ptr]
        if self.active_bounds is not None:
            return _clamp_bbox(*self.active_bounds, c.img.width, c.img.height)
        return self._auto_bbox(c)

    def _display_scale(self, p: Page) -> float:
        if p.img.width <= _DISPLAY_MAX_W:
            return 1.0
        return _DISPLAY_MAX_W / float(p.img.width)

    def _to_full_xy(self, x: int, y: int) -> Tuple[int, int]:
        """Map display-image click coords → full-resolution page coords."""
        p = self.pages[self.page_ptr]
        scale = self._display_scale(p)
        if scale >= 1.0:
            return int(x), int(y)
        return int(round(x / scale)), int(round(y / scale))

    def _compose_page(self) -> Image.Image:
        """Full-res page with green auto guides, red active box, pending corner."""
        p = self.pages[self.page_ptr]
        page = p.img.copy()
        draw = ImageDraw.Draw(page)

        for i, (y0, y1) in enumerate(p.strips):
            box = (0, y0, p.img.width - 1, y1)
            is_suggested = (self.active_bounds is None and i == self.auto_strip_idx)
            width = 3 if is_suggested else 2
            draw.rectangle(box, outline=_GREEN, width=width)
            tag = f"► {i + 1}" if is_suggested else f"{i + 1}"
            _draw_label(draw, (6, y0 + 2), tag, _GREEN)

        bbox = self._active_bbox()
        _semi_fill(page, bbox, _RED, alpha=0.22)
        draw.rectangle(bbox, outline=_RED, width=4)
        # Corner tick marks for clearer feedback
        x0, y0, x1, y1 = bbox
        tick = max(12, min(40, (x1 - x0) // 10))
        for (cx, cy, dx, dy) in (
            (x0, y0, tick, tick), (x1, y0, -tick, tick),
            (x0, y1, tick, -tick), (x1, y1, -tick, -tick),
        ):
            draw.line([(cx, cy), (cx + dx, cy)], fill=_RED, width=4)
            draw.line([(cx, cy), (cx, cy + dy)], fill=_RED, width=4)
        bw, bh = x1 - x0, y1 - y0
        _draw_label(draw, (x0 + 6, max(0, y0 - 18)), f"{bw}×{bh}px", _RED)

        if self.pending_corner is not None:
            px, py = self.pending_corner
            arm = 18
            draw.line([(px - arm, py), (px + arm, py)], fill=_YELLOW, width=3)
            draw.line([(px, py - arm), (px, py + arm)], fill=_YELLOW, width=3)
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=_YELLOW, width=2)
            _draw_label(draw, (px + 10, py - 22), "corner 1 — click opposite corner", _YELLOW)

        if not p.strips:
            _draw_label(draw, (8, 8), "(no strips auto-detected — draw a box)", _RED)

        return page

    def _page_display(self) -> Image.Image:
        """Downscaled composited page for Gradio (select coords match this size)."""
        page = self._compose_page()
        scale = self._display_scale(self.pages[self.page_ptr])
        if scale >= 1.0:
            return page
        w = max(1, int(round(page.width * scale)))
        h = max(1, int(round(page.height * scale)))
        return page.resize((w, h), Image.Resampling.LANCZOS)

    def _strip_preview(self) -> Image.Image:
        """Crop the active bbox; upscale tiny strips so the preview is readable."""
        p = self.pages[self.page_ptr]
        x0, y0, x1, y1 = self._active_bbox()
        crop = p.img.crop((x0, y0, x1, y1))
        # Visually frame the crop
        framed = Image.new("RGB", (crop.width + 8, crop.height + 8), _RED)
        framed.paste(crop, (4, 4))
        if framed.height < 72:
            scale = 72 / framed.height
            framed = framed.resize(
                (max(1, int(framed.width * scale)), 72),
                Image.Resampling.NEAREST,
            )
        return framed

    # -- views --------------------------------------------------------------
    # OUT: (page_display, strip_preview, header, status, text)
    def current_view(self) -> tuple:
        if not self.pages:
            blank = Image.new("RGB", (480, 120), (245, 245, 245))
            return (blank, blank,
                    "No pages — drop PDFs/images in the manuscripts folder and restart.",
                    "", "")
        self.active_bounds = None
        self.pending_corner = None
        self.auto_strip_idx = 0
        p = self.pages[self.page_ptr]
        self.status = self._default_status(p)
        return (self._page_display(), self._strip_preview(),
                self._header(p), self.status, "")

    def _default_status(self, p: Page) -> str:
        n = len(p.strips)
        if n:
            i = self.auto_strip_idx + 1
            return (f"🟢 Auto strip **{i}/{n}** selected (red overlay). "
                    "Click two corners to draw a custom box, or use "
                    "**Next/Prev auto strip** to move between green guides.")
        return "✏️ No auto strips — **click two opposite corners** on the page to draw a box."

    def _header(self, p: Page) -> str:
        n_det = len(p.strips)
        n_saved = self._saved_on_page(p)
        note = ("auto-detect found no strips."
                if n_det == 0 else
                f"auto-detect found {n_det} strip{'s' if n_det != 1 else ''}.")
        return (f"**Page {self.page_ptr + 1} / {len(self.pages)}**  "
                f"·  `{p.src.name}` p.{p.idx + 1}  "
                f"·  **{n_saved} saved** on this page  "
                f"·  {note}")

    def _view(self, text: str = "") -> tuple:
        """Re-render with current bounds / pending corner (keep text box value)."""
        if not self.pages:
            return self.current_view()
        p = self.pages[self.page_ptr]
        return (self._page_display(), self._strip_preview(),
                self._header(p), self.status, text)

    # -- gradio callbacks ---------------------------------------------------
    def on_page_click(self, evt: "gr.SelectData") -> tuple:
        """Two-click box drawing on the page preview."""
        if not self.pages:
            return self.current_view()
        if evt is None or evt.index is None:
            self.status = "⚠️ Click not registered — try again on the page image."
            return self._view()

        # evt.index is (x, y) in display-image coordinates
        try:
            ix, iy = evt.index[0], evt.index[1]
        except (TypeError, IndexError, KeyError):
            self.status = "⚠️ Could not read click coordinates — try again."
            return self._view()
        if ix is None or iy is None:
            self.status = "⚠️ Click coordinates missing — make sure you click the image itself."
            return self._view()

        x, y = self._to_full_xy(int(ix), int(iy))
        p = self.pages[self.page_ptr]
        x = max(0, min(p.img.width - 1, x))
        y = max(0, min(p.img.height - 1, y))

        if self.pending_corner is None:
            self.pending_corner = (x, y)
            self.status = (f"📍 Corner 1 set at **({x}, {y})**. "
                           "Now click the **opposite corner** to finish the box.")
            return self._view()

        # Second click → complete rectangle
        x0, y0 = self.pending_corner
        self.pending_corner = None
        bx0, by0, bx1, by1 = _clamp_bbox(
            min(x0, x), min(y0, y), max(x0, x), max(y0, y),
            p.img.width, p.img.height,
        )
        if (bx1 - bx0) < 4 or (by1 - by0) < 4:
            self.status = "⚠️ Box too small — click two corners farther apart."
            return self._view()

        self.active_bounds = (bx0, by0, bx1, by1)
        self.status = (f"✅ Box set **({bx0},{by0})–({bx1},{by1})** "
                       f"· {bx1 - bx0}×{by1 - by0}px. "
                       "Type the transliteration and submit, or click again to redraw.")
        return self._view()

    def cycle_auto_strip(self, delta: int) -> tuple:
        """Move red selection among auto-detected strips."""
        if not self.pages:
            return self.current_view()
        p = self.pages[self.page_ptr]
        if not p.strips:
            self.status = "No auto strips on this page — draw a box instead."
            return self._view()
        self.active_bounds = None
        self.pending_corner = None
        self.auto_strip_idx = (self.auto_strip_idx + delta) % len(p.strips)
        x0, y0, x1, y1 = self._auto_bbox(p)
        self.status = (f"🟢 Auto strip **{self.auto_strip_idx + 1}/{len(p.strips)}** "
                       f"· ({x0},{y0})–({x1},{y1}) · {x1 - x0}×{y1 - y0}px.")
        return self._view()

    def save_strip(self, text: str, multi: bool) -> tuple:
        """Save the current box + transliteration, then stay/advance."""
        if not self.pages:
            return self.current_view()
        p = self.pages[self.page_ptr]
        text = (text or "").strip()
        if not text:
            self.status = "⚠️ **Enter a transliteration first** (or click Skip)."
            return self._view()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        sid = self._next_strip_id(p)
        x0, y0, x1, y1 = self._active_bbox()
        crop = p.img.crop((x0, y0, x1, y1))
        crop.save(self._png_path(p, sid))
        self._txt_path(p, sid).write_text(text, encoding="utf-8")
        self.saved.setdefault((p.file_key, p.idx), set()).add(sid)
        print(f"[OK] saved {p.file_key}_p{p.idx}_s{sid}  ({p.src.name} p{p.idx + 1})  '{text}'")

        if not multi and self.page_ptr < len(self.pages) - 1:
            self.page_ptr += 1
            view = self.current_view()
            # current_view clears status — reinstate a success note
            self.status = f"💾 Saved strip **s{sid}** and advanced to next page."
            return (view[0], view[1], view[2], self.status, "")

        # Stay on page (multi): reset box to next auto candidate
        view = self.current_view()
        self.status = (f"💾 Saved strip **s{sid}**. Still on this page — "
                       "draw or snap the next strip.")
        return (view[0], view[1], view[2], self.status, "")

    def reset_bounds(self) -> tuple:
        self.active_bounds = None
        self.pending_corner = None
        self.auto_strip_idx = 0
        if self.pages:
            self.status = self._default_status(self.pages[self.page_ptr])
        return self._view()

    def skip_strip(self, multi: bool) -> tuple:
        """Skip without saving — reset box; advance page when multi is off."""
        if not self.pages:
            return self.current_view()
        if not multi and self.page_ptr < len(self.pages) - 1:
            self.page_ptr += 1
            view = self.current_view()
            self.status = "Skipped — advanced to next page."
            return (view[0], view[1], view[2], self.status, "")
        self.active_bounds = None
        self.pending_corner = None
        p = self.pages[self.page_ptr]
        if p.strips:
            self.auto_strip_idx = (self.auto_strip_idx + 1) % len(p.strips)
        self.status = "Skipped this strip — draw or pick the next auto strip."
        return self._view()

    def skip_page(self) -> tuple:
        if self.pages and self.page_ptr < len(self.pages) - 1:
            self.page_ptr += 1
        return self.current_view()

    def prev_page(self) -> tuple:
        if self.page_ptr > 0:
            self.page_ptr -= 1
        return self.current_view()

    def next_page(self) -> tuple:
        if self.page_ptr < len(self.pages) - 1:
            self.page_ptr += 1
        return self.current_view()


# ---------------------------------------------------------------------------
# CLI + Gradio UI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Javanese Aksara manuscript HITL labeler (PDFs, PNG, JPG).")
    p.add_argument("--pdfs_dir", "--manuscripts_dir", "--images_dir",
                   dest="pdfs_dir", type=Path, default=Path("../pdfs"),
                   help="Directory containing scanned manuscript PDFs or images (PNG/JPG).")
    p.add_argument("--output_dir", type=Path, default=Path("../pdf_labeled"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--bg_radius", type=int, default=25,
                   help="Gaussian blur radius (px) for local background estimation.")
    p.add_argument("--bg_delta", type=int, default=15,
                   help="Ink = pixel darker than local background by this many levels.")
    return p.parse_args()


def build_demo(labeler: "Labeler") -> "gr.Blocks":
    if gr is None:
        print("[ERROR] gradio is required to run the UI:  pip install gradio", file=sys.stderr)
        sys.exit(1)

    with gr.Blocks(title="Javanese Aksara — HITL labeler") as demo:
        gr.Markdown(
            "# Javanese Aksara — manuscript labeler\n"
            "**Green** boxes = auto-detected strips · **Red** overlay = what will be saved.\n\n"
            "1. Use **Next/Prev auto strip** to move the red box among green guides, "
            "**or click two opposite corners** on the page to draw a custom box "
            "(yellow crosshair marks the first corner).\n"
            "2. Check the strip preview — that exact crop is what gets saved.\n"
            "3. Type the Latin transliteration and press **Submit + next** (or Enter).\n\n"
            "Keep **Multi-strip** checked to stay on the page after each save."
        )
        with gr.Row():
            with gr.Column(scale=3):
                # interactive=False + sources=[]: display-only; .select still
                # returns pixel coords. format=png avoids webp artifacts on ink.
                page_img = gr.Image(
                    label="Page — click two corners to draw a box",
                    type="pil",
                    format="png",
                    interactive=False,
                    sources=[],
                    buttons=["fullscreen"],
                    elem_classes=["page-canvas"],
                    height=640,
                )
                status_box = gr.Markdown(elem_classes=["status-box"])
                with gr.Row():
                    prev_strip_btn = gr.Button("◀ Prev auto strip", variant="secondary")
                    next_strip_btn = gr.Button("Next auto strip ▶", variant="secondary")
                with gr.Row():
                    reset_btn = gr.Button("Reset to auto-detect", variant="secondary")
                    clear_corner_btn = gr.Button("Cancel corner", variant="stop")
            with gr.Column(scale=2):
                strip_img = gr.Image(
                    label="Strip preview (saved crop)",
                    type="pil",
                    format="png",
                    interactive=False,
                    sources=[],
                    buttons=["fullscreen"],
                    elem_classes=["strip-preview"],
                    height=180,
                )
                header_box = gr.Markdown("")
                multi_chk = gr.Checkbox(
                    value=True,
                    label="Multi-strip page (stay after saving)",
                    info="Checked: save stays on this page so you can "
                         "draw and save the next strip. "
                         "Unchecked: save advances to the next page.",
                )
                text_in = gr.Textbox(
                    label="Latin transliteration",
                    placeholder="e.g.  tatanah jawa kuna ...",
                    lines=2,
                    autofocus=True,
                )
                with gr.Row():
                    submit = gr.Button("Submit + next", variant="primary")
                    skip = gr.Button("Skip strip", variant="secondary")
                with gr.Row():
                    prev_btn = gr.Button("◀ Prev page", variant="secondary")
                    next_btn = gr.Button("Next page ▶", variant="secondary")
                    skip_btn = gr.Button("Skip page", variant="stop")

        OUT = [page_img, strip_img, header_box, status_box, text_in]

        def _cancel_corner():
            labeler.pending_corner = None
            if labeler.pages:
                labeler.status = labeler._default_status(labeler.pages[labeler.page_ptr])
            return labeler._view()

        # Proper initial load (avoids fragile temp-file pre-seeding).
        demo.load(labeler.current_view, inputs=None, outputs=OUT)

        page_img.select(labeler.on_page_click, inputs=None, outputs=OUT)
        prev_strip_btn.click(lambda: labeler.cycle_auto_strip(-1), inputs=None, outputs=OUT)
        next_strip_btn.click(lambda: labeler.cycle_auto_strip(+1), inputs=None, outputs=OUT)
        reset_btn.click(labeler.reset_bounds, inputs=None, outputs=OUT)
        clear_corner_btn.click(_cancel_corner, inputs=None, outputs=OUT)
        submit.click(labeler.save_strip, inputs=[text_in, multi_chk], outputs=OUT)
        text_in.submit(labeler.save_strip, inputs=[text_in, multi_chk], outputs=OUT)
        skip.click(labeler.skip_strip, inputs=[multi_chk], outputs=OUT)
        prev_btn.click(labeler.prev_page, inputs=None, outputs=OUT)
        next_btn.click(labeler.next_page, inputs=None, outputs=OUT)
        skip_btn.click(labeler.skip_page, inputs=None, outputs=OUT)

    return demo


def main():
    args = parse_args()
    labeler = Labeler(pdfs_dir=args.pdfs_dir.resolve(),
                      output_dir=args.output_dir.resolve(),
                      dpi=args.dpi, bg_radius=args.bg_radius, bg_delta=args.bg_delta)
    count = labeler.load()
    if count == 0:
        print("\nNothing to label. Drop scanned PDFs or images (PNG/JPG) into "
              f"{args.pdfs_dir.resolve()} and re-run.", file=sys.stderr)
        sys.exit(1)

    demo = build_demo(labeler)
    css = """
    .strip-preview img { object-fit: contain !important; background: #1a1a1a; }
    .page-canvas img { object-fit: contain !important; cursor: crosshair !important; }
    .status-box { min-height: 2.5em; }
    """
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.port, show_error=True,
        prevent_thread_lock=False, css=css,
    )


if __name__ == "__main__":
    main()
