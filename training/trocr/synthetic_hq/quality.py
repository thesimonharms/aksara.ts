"""Quality gates for synthetic HQ line images."""

from __future__ import annotations

import numpy as np
from PIL import Image


def passes_quality_gate(
    img: Image.Image,
    *,
    min_ink_frac: float = 0.004,
    max_ink_frac: float = 0.55,
    min_contrast: float = 12.0,
) -> bool:
    """Reject blank, clipped-looking, or ultra-low-contrast renders.

    Ink is estimated as dark pixels relative to a light paper background.
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return False

    gray = arr.mean(axis=2)
    # Dark ink on light paper
    ink_mask = gray < 140.0
    ink_frac = float(ink_mask.mean())
    if ink_frac < min_ink_frac or ink_frac > max_ink_frac:
        return False

    contrast = float(gray.std())
    if contrast < min_contrast:
        return False

    # Edge clipping: too much ink touching the border suggests cut-off glyphs.
    h, w = gray.shape
    border = np.concatenate(
        [
            ink_mask[0, :].ravel(),
            ink_mask[-1, :].ravel(),
            ink_mask[:, 0].ravel(),
            ink_mask[:, -1].ravel(),
        ]
    )
    if float(border.mean()) > 0.35:
        return False

    return True
