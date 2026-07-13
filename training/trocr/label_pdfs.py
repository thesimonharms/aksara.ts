#!/usr/bin/env python3
"""
label_pdfs.py — Human-in-the-loop Javanese Aksara manuscript labeler.

Crops candidate text-line strips from PDFs in --pdfs_dir, auto-detected via
horizontal dark-pixel projection; shows each candidate in a local Gradio web
UI where you type the Latin transliteration. Pairs are saved as
{output_dir}/label_XXXX.png + {label_XXXX}.txt — same shape as the existing
ocr_corpus_clean/ dataset, so finetune_trocr.py can mix them with synthetic
data by adding output_dir to the dataset loader.

Resume-safe: walks PDFs in stable order and skips label ids already saved.
Manual boxes (drag-select) are also supported — when auto-detect misses a
line, click-drag on the page image and the box replaces the auto crop.

Usage:
    python label_pdfs.py --pdfs_dir ../pdfs --output_dir ../pdf_labeled
    # then open http://127.0.0.1:7861 in your browser
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERROR] PyMuPDF (fitz) is required:  uv pip install pymupdf", file=sys.stderr)
    sys.exit(1)

import gradio as gr


# ---------------------------------------------------------------------------
# Strip detection — horizontal projection profile
# ---------------------------------------------------------------------------
def _binarize(arr: np.ndarray) -> np.ndarray:
    """Cheap adaptive-ish binarization: dark pixels relative to row median."""
    gray = arr.mean(axis=2) if arr.ndim == 3 else arr
    # Per-page Otsu-ish threshold using global mean; works on aged lontar where
    # ink is dark on light parchment. Good enough for projection profile.
    thr = gray.mean() - 30
    return gray < thr


def detect_strips(img: Image.Image,
                  min_strip_h: int = 14,
                  max_strip_h: int = 240,
                  ink_ratio_min: float = 0.004,
                  gap_min: int = 6) -> List[Tuple[int, int]]:
    """Return list of (y0, y1) candidate line strips on a page.

    Uses horizontal projection of dark pixels; merges short gaps, drops thin /
    mostly-empty bands. Tuned for ~150–200 DPI palm-leaf and codex scans.
    """
    arr = np.asarray(img.convert("L"))
    dark = _binarize(arr).astype(np.uint8)
    row_density = dark.mean(axis=1)  # fraction of dark pixels per row

    # Threshold each row by ink_ratio_min to mark "text rows"; merge contiguous.
    in_text = row_density > ink_ratio_min
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

    # Merge strips separated by < gap_min rows.
    merged: List[Tuple[int, int]] = []
    for s in strips:
        if merged and s[0] - merged[-1][1] < gap_min:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)

    # Drop bands too tall (likely columns/margins) or too thin (noise).
    return [(a, b) for (a, b) in merged if min_strip_h <= (b - a) <= max_strip_h]


# ---------------------------------------------------------------------------
# PDF → page images → candidate strips
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    pdf_path: Path
    page_idx: int
    strip_idx: int
    bbox: Tuple[int, int, int, int]      # (x0, y0, x1, y1) in page coords
    img: Image.Image                     # cropped strip (RGB)
    page_img: Image.Image                # full page (for manual box fallback)


@dataclass
class Labeler:
    pdfs_dir: Path
    output_dir: Path
    dpi: int = 200
    pad_y: int = 3                        # white padding above/below each strip
    candidates: List[Candidate] = field(default_factory=list)
    idx: int = 0
    saved_labels: set = field(default_factory=set)
    # Active crop bounds for the current candidate. Defaults to the auto-
    # detected bbox; the user can override the vertical range via sliders
    # (strips span full page width, so y-bounds alone cover every case for
    # horizontal lontar / codex lines). None means "use auto-detect".
    active_bounds: Optional[Tuple[int, int]] = None  # (y0, y1) in page coords

    def load_candidates(self) -> int:
        """Walk pdfs_dir in sorted order, rasterize pages, detect strips."""
        pdfs = sorted(
            list(self.pdfs_dir.glob("*.pdf")) + list(self.pdfs_dir.glob("*.PDF"))
        ) if self.pdfs_dir.exists() else []
        if not pdfs:
            print(f"[WARN] No PDFs found in {self.pdfs_dir}")
            return 0

        # Deterministic ordering across runs: (pdf_idx, page_idx, strip_idx).
        for pidx, pdf_path in enumerate(pdfs):
            try:
                doc = fitz.open(pdf_path)
            except Exception as exc:
                print(f"[WARN] could not open {pdf_path}: {exc}")
                continue
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=self.dpi)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                arr_img = page_img.copy()
                for sidx, (y0, y1) in enumerate(detect_strips(page_img)):
                    y0p = max(0, y0 - self.pad_y)
                    y1p = min(page_img.height, y1 + self.pad_y)
                    crop = arr_img.crop((0, y0p, page_img.width, y1p))
                    self.candidates.append(Candidate(
                        pdf_path=pdf_path, page_idx=page_idx, strip_idx=sidx,
                        bbox=(0, y0p, page_img.width, y1p),
                        img=crop, page_img=arr_img,
                    ))
            doc.close()
        self._load_saved_labels()
        self._skip_already_labeled()
        print(f"[INFO] {len(self.candidates)} candidate strips after resume filter")
        return len(self.candidates)

    def _label_path(self, idx: int) -> Path:
        return self.output_dir / f"label_{idx:06d}.txt"

    def _png_path(self, idx: int) -> Path:
        return self.output_dir / f"label_{idx:06d}.png"

    def _load_saved_labels(self) -> None:
        if not self.output_dir.exists():
            return
        for p in self.output_dir.glob("label_*.txt"):
            m = re.match(r"label_(\d{6})\.txt", p.name)
            if m:
                self.saved_labels.add(int(m.group(1)))

    def _skip_already_labeled(self) -> None:
        while self.idx < len(self.candidates) and self.idx in self.saved_labels:
            self.idx += 1

    def _active_bbox(self) -> Tuple[int, int, int, int]:
        """Resolve the bbox the submit will actually crop — manual or auto."""
        c = self.candidates[self.idx]
        if self.active_bounds is not None:
            y0, y1 = self.active_bounds
            y0 = max(0, min(c.page_img.height - 1, y0))
            y1 = max(y0 + 1, min(c.page_img.height, y1))
            return (0, y0, c.page_img.width, y1)
        return c.bbox

    def _strip_preview(self) -> Image.Image:
        c = self.candidates[self.idx]
        x0, y0, x1, y1 = self._active_bbox()
        return c.page_img.crop((x0, y0, x1, y1))

    # -- gradio callbacks ---------------------------------------------------
    # Return tuple layout matches the UI outputs:
    #   (strip_preview, page_with_box, header, text_in, y0_slider, y1_slider,
    #    manual_status)
    def current_view(self) -> tuple:
        if not self.candidates:
            blank = Image.new("RGB", (320, 80), "white")
            return (blank, blank,
                    "No candidates — drop PDFs in training/pdfs/ and restart.",
                    "", 0, 1, "")
        c = self.candidates[self.idx]
        bbox = self._active_bbox()
        # Page preview with red rectangle at the active bbox.
        page = c.page_img.copy()
        from PIL import ImageDraw
        d = ImageDraw.Draw(page)
        d.rectangle(bbox, outline=(220, 30, 30), width=3)
        header = (
            f"**{self.idx + 1} / {len(self.candidates)}**  "
            f"·  `{c.pdf_path.name}` p.{c.page_idx + 1} strip #{c.strip_idx}  "
            f"·  {len(self.saved_labels)} labeled"
        )
        existing = ""
        if self.idx in self.saved_labels:
            try:
                existing = self._label_path(self.idx).read_text(encoding="utf-8")
            except Exception:
                existing = ""
        y0, y1 = bbox[1], bbox[3]
        status = "auto" if self.active_bounds is None else "manual"
        return (self._strip_preview(), page, header, existing,
                int(y0), int(y1), f"Bounds: **{status}**  ({bbox[1]}–{bbox[3]})")

    def apply_bounds(self, y0, y1) -> tuple:
        """Slider drag — set manual bounds and re-render."""
        if not self.candidates:
            return self.current_view()
        y0 = max(0, int(y0))
        y1 = max(y0 + 1, int(y1))
        c = self.candidates[self.idx]
        y1 = min(c.page_img.height, y1)
        # Mark manual only if it differs from auto-detect.
        auto_y = (c.bbox[1], c.bbox[3])
        self.active_bounds = None if (y0, y1) == auto_y else (y0, y1)
        return self.current_view()

    def reset_bounds(self) -> tuple:
        """Restore auto-detect bounds for the current candidate."""
        self.active_bounds = None
        return self.current_view()

    def submit_label(self, text: str) -> tuple:
        text = (text or "").strip()
        if not self.candidates:
            return self.current_view()
        c = self.candidates[self.idx]
        img = self._strip_preview()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        img.save(self._png_path(self.idx))
        self._label_path(self.idx).write_text(text, encoding="utf-8")
        self.saved_labels.add(self.idx)
        print(f"[OK] saved label_{self.idx:06d}  ({c.pdf_path.name} p{c.page_idx + 1})")
        return self._advance()

    def skip(self) -> tuple:
        return self._advance()

    def _advance(self) -> tuple:
        if not self.candidates:
            return self.current_view()
        self.idx += 1
        self.active_bounds = None
        while self.idx < len(self.candidates) and self.idx in self.saved_labels:
            self.idx += 1
        if self.idx >= len(self.candidates):
            blank = Image.new("RGB", (320, 80), "white")
            return (blank, blank,
                    f"**Done!** {len(self.saved_labels)} labeled. "
                    f"Restart with more PDFs to continue.",
                    "", 0, 1, "")
        return self.current_view()


# ---------------------------------------------------------------------------
# CLI + Gradio UI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Javanese Aksara manuscript HITL labeler.")
    p.add_argument("--pdfs_dir",   type=Path, default=Path("../pdfs"))
    p.add_argument("--output_dir", type=Path, default=Path("../pdf_labeled"))
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--port", type=int, default=7861)
    return p.parse_args()


def main():
    args = parse_args()
    labeler = Labeler(pdfs_dir=args.pdfs_dir.resolve(),
                      output_dir=args.output_dir.resolve(),
                      dpi=args.dpi)
    count = labeler.load_candidates()
    if count == 0:
        print("\nNothing to label. Drop scanned PDFs into "
              f"{args.pdfs_dir.resolve()} and re-run.", file=sys.stderr)
        sys.exit(1)

    page_max_h = max((c.page_img.height for c in labeler.candidates), default=1)

    with gr.Blocks(title="Javanese Aksara — HITL labeler") as demo:
        gr.Markdown(
            "# Javanese Aksara — manuscript labeler\n"
            "The **red box** on the page preview marks the candidate text line.\n"
            "1. Inspect the strip preview on the left.\n"
            "2. Type the Latin transliteration in the box.\n"
            "3. Press **Submit + next** (or press Enter in the textbox).\n\n"
            "**If auto-detect is wrong**: drag the **Y bounds** sliders below the "
            "page preview to grow/shrink the strip, then submit — your manual "
            "bounds override. **Reset bounds** returns to auto-detect."
        )
        with gr.Row():
            with gr.Column(scale=2):
                strip_img = gr.Image(label="Strip preview (this is what gets saved)",
                                     type="pil", interactive=False)
                header_box = gr.Markdown("")
                with gr.Row():
                    y0_slider = gr.Slider(0, page_max_h, value=0, step=1,
                                          label="Y start (top)")
                    y1_slider = gr.Slider(0, page_max_h, value=1, step=1,
                                          label="Y end (bottom)")
                with gr.Row():
                    apply_btn = gr.Button("Apply manual bounds", variant="secondary")
                    reset_btn = gr.Button("Reset to auto-detect", variant="stop")
                manual_status = gr.Markdown("")
            with gr.Column(scale=2):
                page_img = gr.Image(label="Page (red box = current strip)",
                                    type="pil", interactive=False)
                text_in = gr.Textbox(label="Latin transliteration",
                                     placeholder="e.g.  tatanah jawa kuna ...",
                                     lines=2, autofocus=True)
                with gr.Row():
                    submit = gr.Button("Submit + next", variant="primary")
                    skip = gr.Button("Skip", variant="secondary")

        OUT = [strip_img, page_img, header_box, text_in,
               y0_slider, y1_slider, manual_status]

        # Initial render — current_view returns 7-tuple matching OUT order.
        try:
            initial = labeler.current_view()
        except Exception as exc:
            print(f"[ERROR] initial render failed: {exc}", file=sys.stderr)
            raise
        for comp, val in zip(OUT, initial):
            comp.value = val

        submit.click(labeler.submit_label, inputs=[text_in], outputs=OUT)
        skip.click(labeler.skip, inputs=None, outputs=OUT)
        apply_btn.click(labeler.apply_bounds, inputs=[y0_slider, y1_slider], outputs=OUT)
        reset_btn.click(labeler.reset_bounds, inputs=None, outputs=OUT)

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.port, show_error=True,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()