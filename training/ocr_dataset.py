# aksara.ts/training/ocr_dataset.py

import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

# The same alphabet used by the model (blank token + Javanese characters)
ALPHABET = ["[blank]"] + [chr(i) for i in range(0xA98F, 0xA9C1)]  # consonants + HA + sandhangan + pangkon
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def text_to_indices(text: str) -> List[int]:
    """Convert a string of Javanese characters into a list of integer indices."""
    return [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]


class OcrDataset(Dataset):
    """
    Simple PyTorch Dataset that pairs PNG images with their corresponding
    ground‑truth text files. The dataset expects each image file to have a
    matching .txt file containing the exact string of Javanese characters.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        # Collect all PNGs; ensure a corresponding .txt exists
        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(self.data_dir.glob("*.png")):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.is_file():
                self.samples.append((img_path, txt_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, txt_path = self.samples[idx]
        image = Image.open(img_path).convert("L")  # grayscale
        label_text = txt_path.read_text(encoding="utf-8")
        labels = torch.tensor(text_to_indices(label_text), dtype=torch.long)

        return image, labels
