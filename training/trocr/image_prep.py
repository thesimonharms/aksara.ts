"""Line-image prep for TrOCR's square ViT encoder.

TrOCR resizes to 384×384. Stretching a 64px-tall line to 384px tall destroys
aksara proportions (sandhangan above/below the body). Pad to square first so
resize is uniform.
"""

from __future__ import annotations

from PIL import Image


def pad_to_square(img: Image.Image, fill: tuple[int, int, int] | None = None) -> Image.Image:
    """Left-align, vertically center on a square canvas. No-op if already square."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w == h:
        return rgb
    side = max(w, h)
    if fill is None:
        fill = rgb.getpixel((0, 0))
        if not isinstance(fill, tuple) or len(fill) < 3:
            fill = (250, 248, 240)
        else:
            fill = (int(fill[0]), int(fill[1]), int(fill[2]))
    canvas = Image.new("RGB", (side, side), fill)
    y = (side - h) // 2
    canvas.paste(rgb, (0, y))
    return canvas


def ink_bbox(image: Image.Image, *, dark_below: int = 140) -> tuple[int, int, int, int] | None:
    """Bounding box of dark ink, or None if blank."""
    bw = image.convert("L").point(lambda p: 255 if p < dark_below else 0)
    return bw.getbbox()


def estimate_char_budget(image: Image.Image) -> int:
    """Aksara count from ink aspect ratio (works on square-padded canvases)."""
    box = ink_bbox(image)
    if box is None:
        return 8
    ink_w = max(1, box[2] - box[0])
    ink_h = max(1, box[3] - box[1])
    return max(1, int(round(ink_w / max(ink_h * 0.55, 1.0))))
