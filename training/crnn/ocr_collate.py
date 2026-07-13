# aksara.ts/training/ocr_collate.py

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


def collate_fn(batch: List[Tuple[Image.Image, torch.Tensor]]):
    """
    Collate function for the OCR dataset.

    Parameters
    ----------
    batch : list of (PIL image, label_tensor)

    Returns
    -------
    images   : FloatTensor [B, 1, H, W]
        Batch of grayscale images resized to (32, 128) and normalized to [0, 1].
    labels   : LongTensor
        Concatenated ground‑truth indices for the whole batch.
    lengths  : LongTensor
        Length of each label sequence in the batch.
    """
    # Resize images to (H=32, W=128) and convert to tensor
    imgs = []
    for img, _ in batch:
        # First resize, then convert to grayscale
        resized_img = img.resize((128, 32)).convert("L")
        # Convert PIL Image to numpy array -> [H, W]
        arr = np.array(resized_img)
        # Convert numpy to torch tensor and normalize to [0, 1]
        tensor = torch.from_numpy(arr).float() / 255.0  # [H, W]
        # Add channel dimension: [1, H, W]
        imgs.append(tensor.unsqueeze(0))

    images = torch.stack(imgs, dim=0)  # [B, 1, H, W]

    # Concatenate labels and record their lengths
    labels = torch.cat([lbl for _, lbl in batch])
    lengths = torch.tensor([len(lbl) for _, lbl in batch], dtype=torch.long)

    return images, labels, lengths
