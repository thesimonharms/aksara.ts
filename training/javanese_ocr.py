# aksara.ts/training/javanese_ocr.py
"""
Javanese OCR End-to-End Pipeline

Generates synthetic data, trains a CRNN model with CTC loss, and provides
inference, self-training, and language-model-assisted decoding.

Recommended workflow to improve accuracy without manual labelling:

  Step 1 — Generate training data from a real text corpus + manuscript backgrounds:
    python javanese_ocr.py --mode generate_from_corpus \\
        --corpus javanese_text.txt --background_pdf PDFA.pdf \\
        --data_dir ./ocr_corpus --num_samples 5000

  Step 2 — (Re)train the model:
    python javanese_ocr.py --mode train \\
        --data_dir ./ocr_data ./ocr_corpus --epochs 30 --lr 0.001

  Step 3 — Train a character n-gram language model:
    python javanese_ocr.py --mode train_lm \\
        --corpus javanese_text.txt --output_path javanese_lm.pkl

  Step 4 — Auto-label manuscript strips (pseudo-labelling / self-training):
    python javanese_ocr.py --mode pseudo_label \\
        --unlabeled_dir ./manuscript_strips --data_dir ./pseudo_labeled \\
        --lm_path javanese_lm.pkl --threshold 0.92

  Step 5 — Retrain with expanded dataset, repeat from Step 4.

  Predict with LM-assisted beam search:
    python javanese_ocr.py --mode predict --pdf PDFA.pdf \\
        --lm_path javanese_lm.pkl --beam_width 10

  Export to ONNX for TypeScript:
    python javanese_ocr.py --mode export_onnx --output_path javanese_ocr.onnx
"""

import argparse
import glob
import os
import pickle
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import fitz  # pymupdf — no poppler required
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import DataLoader, Dataset

from ocr_dataset import text_to_indices

# ---------------------------------------------------------------------------
# DEVICE SELECTION  (DirectML > CUDA > CPU)
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    try:
        import torch_directml
        dev = torch_directml.device()
        print(f"Device: DirectML (AMD/Intel GPU)")
        return dev
    except ImportError:
        pass
    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("Device: CPU")
    return torch.device("cpu")

DEVICE = _select_device()

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 128
JAVANESE_CHARS = [chr(i) for i in range(0xA98F, 0xA9C1)]  # consonants + HA + sandhangan + pangkon
ALPHABET = ["[blank]"] + JAVANESE_CHARS  # index 0 = CTC blank token
NUM_CLASSES = len(ALPHABET)
ALPHABET_MAP = {i: ch for i, ch in enumerate(ALPHABET)}
BLANK_IDX = 0

DEFAULT_MODEL_PATH = Path(__file__).parent / "javanese_ocr.pth"

# ---------------------------------------------------------------------------
# 2. NEURAL NETWORK (CRNN)
# ---------------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — adds position information to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.drop(x)


class SimpleCRNN(nn.Module):
    """
    CNN + Transformer encoder for CTC-based OCR.

    Uses only matmul/softmax ops — no fused LSTM kernel — so it runs natively
    on DirectML (AMD/Intel GPU via torch-directml).

    Architecture:
        CNN backbone       -> [B, C*H, W/4] feature sequence
        Linear proj        -> [B, W/4, 256]
        Positional encoding-> [B, W/4, 256]
        Transformer        -> [B, W/4, 256]  (2 layers, 4 heads)
        Linear head        -> [B, W/4, num_classes]
    """

    _PROJ_DIM = 256

    def __init__(self, num_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        cnn_feat_dim = 64 * (IMAGE_HEIGHT // 4)
        self.proj = nn.Linear(cnn_feat_dim, self._PROJ_DIM)
        self.pos_enc = _PositionalEncoding(self._PROJ_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._PROJ_DIM,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(self._PROJ_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x)                               # [B, 64, H/4, W/4]
        b, c, h, w = conv.size()
        feat = conv.view(b, c * h, w).permute(0, 2, 1)  # [B, W/4, C*H]
        feat = self.proj(feat)                           # [B, W/4, 256]
        feat = self.pos_enc(feat)                        # [B, W/4, 256]
        out  = self.transformer(feat)                    # [B, W/4, 256]
        return self.fc(out)                              # [B, W/4, num_classes]


# ---------------------------------------------------------------------------
# 3. LANGUAGE MODEL  (Strategy 4)
# ---------------------------------------------------------------------------


class CharNgramLM:
    """
    Character-level n-gram language model with Laplace smoothing.

    Trained on raw Javanese Unicode text, it assigns probabilities to
    character sequences.  Used during CTC beam search to prefer linguistically
    plausible outputs over acoustically similar but nonsensical alternatives.

    Typical usage:
        lm = CharNgramLM(n=3)
        lm.train(Path("javanese_text.txt").read_text(encoding="utf-8"))
        lm.save("javanese_lm.pkl")

        lm = CharNgramLM.load("javanese_lm.pkl")
        log_p = lm.log_prob("ꦲꦤ", "ꦸ")   # log P("ꦸ" | "ꦲꦤ")
    """

    def __init__(self, n: int = 3, smoothing: float = 0.1):
        self.n = n
        self.smoothing = smoothing
        self._counts: Dict[str, Counter] = defaultdict(Counter)
        self._vocab: set = set()

    def train(self, text: str) -> None:
        """Count n-grams from `text`."""
        self._vocab.update(text)
        pad = "\x00" * (self.n - 1)
        padded = pad + text
        for i in range(len(padded) - self.n + 1):
            ctx = padded[i : i + self.n - 1]
            ch = padded[i + self.n - 1]
            self._counts[ctx][ch] += 1

    def log_prob(self, prefix: str, char: str) -> float:
        """Laplace-smoothed log P(char | last n-1 chars of prefix)."""
        pad = "\x00" * (self.n - 1)
        ctx = (pad + prefix)[-(self.n - 1) :]
        counts = self._counts.get(ctx, Counter())
        vocab_size = max(len(self._vocab), 1)
        total = sum(counts.values()) + self.smoothing * vocab_size
        return float(np.log((counts.get(char, 0) + self.smoothing) / total))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CharNgramLM":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CharNgramLM, got {type(obj)}")
        return obj


def train_lm(
    corpus_path: str,
    output_path: str,
    n: int = 3,
    smoothing: float = 0.1,
) -> None:
    """
    Train a character n-gram LM on a Javanese text corpus and save it.

    Only Javanese Unicode characters (U+A98F–U+A9B1) are used; Latin text,
    punctuation, and spaces are stripped before training.

    The saved .pkl file can be passed to --lm_path during prediction or
    pseudo-labelling to enable LM-assisted beam-search decoding.
    """
    text = Path(corpus_path).read_text(encoding="utf-8")
    javanese_set = set(JAVANESE_CHARS)
    javanese_text = "".join(ch for ch in text if ch in javanese_set)

    if len(javanese_text) < 10:
        raise ValueError(
            f"'{corpus_path}' contains fewer than 10 Javanese characters "
            f"(U+A98F–U+A9B1).  Check that the file uses Unicode Javanese script."
        )

    print(f"Training {n}-gram LM on {len(javanese_text):,} Javanese characters…")
    lm = CharNgramLM(n=n, smoothing=smoothing)
    lm.train(javanese_text)
    lm.save(output_path)

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"Vocab          : {len(lm._vocab)} unique characters")
    print(f"Unique contexts: {len(lm._counts)}")
    print(f"Saved          : {Path(output_path).resolve()} ({size_kb:.1f} KB)")
    print()
    print("Use with prediction:")
    print(f"  --lm_path {output_path} --beam_width 10")


# ---------------------------------------------------------------------------
# 4. MODEL LOADING & DECODING
# ---------------------------------------------------------------------------


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> torch.nn.Module:
    """Instantiate SimpleCRNN and load pretrained weights."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    model = SimpleCRNN(NUM_CLASSES)
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def ctc_greedy_decode(logits: torch.Tensor) -> Tuple[str, float]:
    """
    Fast greedy CTC decoder (no language model).

    Returns (decoded_text, avg_confidence) where avg_confidence is the mean
    softmax probability of the winning class at each timestep.
    """
    probs = logits.squeeze(0)  # [T, C]
    softmax = torch.softmax(probs, dim=-1)
    avg_confidence = softmax.max(dim=-1).values.mean().item()

    preds = torch.argmax(probs, dim=-1).tolist()
    decoded: List[str] = []
    prev = None
    for p in preds:
        if p != BLANK_IDX and p != prev:
            decoded.append(ALPHABET_MAP[p])
        prev = p
    return "".join(decoded), avg_confidence


def ctc_beam_decode(
    logits: torch.Tensor,
    lm: Optional[CharNgramLM] = None,
    beam_width: int = 10,
    lm_weight: float = 0.5,
) -> Tuple[str, float]:
    """
    CTC beam-search decoder with optional character n-gram language model.
    (Strategy 4 — Language Model Reranking)

    Uses shallow fusion: at each timestep, the LM log-probability of the new
    character is added to the beam score (weighted by lm_weight).  This biases
    the search toward linguistically valid Javanese sequences without retraining.

    Parameters:
        logits      Raw model output, shape [1, T, num_classes].
        lm          Trained CharNgramLM, or None to run without LM.
        beam_width  Number of hypotheses to maintain.  10–20 is a good range.
        lm_weight   How strongly to weight the LM (0 = CTC only, 1 = equal weight).
                    Start at 0.3–0.5; raise if output is more fluent but less accurate.

    Returns:
        (decoded_text, avg_confidence_proxy)
    """
    log_probs = torch.log_softmax(logits.squeeze(0), dim=-1).cpu().numpy()  # [T, C]
    T, C = log_probs.shape
    NEG_INF = float("-inf")

    # State: {(prefix_str, last_char_idx): [log_p_blank, log_p_non_blank]}
    # log_p_blank    = log prob of all CTC paths ending in blank that decode to prefix
    # log_p_non_blank = same but ending in a non-blank character
    beams: Dict[Tuple[str, int], List[float]] = {("", BLANK_IDX): [0.0, NEG_INF]}

    for t in range(T):
        new_beams: Dict[Tuple[str, int], List[float]] = {}

        def _merge(key: Tuple[str, int], p_b: float = NEG_INF, p_nb: float = NEG_INF) -> None:
            if key not in new_beams:
                new_beams[key] = [NEG_INF, NEG_INF]
            new_beams[key][0] = np.logaddexp(new_beams[key][0], p_b)
            new_beams[key][1] = np.logaddexp(new_beams[key][1], p_nb)

        for (prefix, last_c), (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)

            # Blank: prefix unchanged, last char resets to blank
            _merge((prefix, BLANK_IDX), p_b=p_total + log_probs[t, BLANK_IDX])

            # Non-blank characters
            for c_idx in range(1, C):
                char = ALPHABET_MAP[c_idx]
                ctc_lp = log_probs[t, c_idx]
                lm_lp = lm.log_prob(prefix, char) * lm_weight if lm else 0.0
                new_prefix = prefix + char

                if c_idx == last_c:
                    # Repeating the same char: only blank-ending paths may emit it
                    _merge((new_prefix, c_idx), p_nb=p_b + ctc_lp + lm_lp)
                else:
                    _merge((new_prefix, c_idx), p_nb=p_total + ctc_lp + lm_lp)

        # Prune to beam_width
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_key, (p_b, p_nb) = max(
        beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1])
    )
    prefix = best_key[0]
    total_log_prob = np.logaddexp(p_b, p_nb)
    # Normalise by sequence length as a per-character confidence proxy
    conf = float(np.exp(np.clip(total_log_prob / max(len(prefix), 1), -100, 0)))
    return prefix, conf


# ---------------------------------------------------------------------------
# 5. IMAGE SEGMENTATION
# ---------------------------------------------------------------------------


def segment_page_into_lines(
    gray: Image.Image,
    min_line_height: int = 8,
    bg_blur_radius: int = 40,
    ink_ratio_threshold: float = 0.85,
    valley_percentile: float = 35,
    smooth_window: int = 20,
    min_valley_distance: int = 12,
) -> List[Image.Image]:
    """
    Split a greyscale page image into text-line strips.

    Works on clean scans AND noisy/textured manuscript pages by using local
    background subtraction rather than a fixed pixel threshold:

    1. Estimate background with a heavy Gaussian blur.
    2. Pixels darker than ink_ratio_threshold × local background = ink.
    3. Sum ink per row → horizontal projection profile.
    4. Box-filter smooth the profile.
    5. Detect inter-line valleys as local minima below valley_percentile.
    6. Crop between consecutive valley rows.

    Falls back to the full image as a single strip when the image is too short
    to segment (e.g. an already-cropped 32 px tile).
    """
    arr = np.array(gray).astype(np.float32)
    bg = gray.filter(ImageFilter.GaussianBlur(radius=bg_blur_radius))
    bg_arr = np.array(bg).astype(np.float32)
    ink_mask = arr / (bg_arr + 1e-5) < ink_ratio_threshold
    ink_per_row = ink_mask.sum(axis=1).astype(np.float32)

    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(ink_per_row, kernel, mode="same")

    margin = smooth_window
    if len(smoothed) <= 2 * margin:
        return [gray]

    interior = smoothed[margin:-margin]
    threshold = float(np.percentile(interior, valley_percentile))

    boundaries: List[int] = [0]
    last_valley = -min_valley_distance * 2
    for i in range(margin, len(smoothed) - margin):
        if smoothed[i] < threshold:
            window = smoothed[max(0, i - min_valley_distance) : i + min_valley_distance + 1]
            if smoothed[i] == window.min() and i - last_valley >= min_valley_distance:
                boundaries.append(i)
                last_valley = i
    boundaries.append(gray.height)

    lines: List[Image.Image] = []
    for top, bot in zip(boundaries, boundaries[1:]):
        if bot - top >= min_line_height:
            lines.append(gray.crop((0, top, gray.width, bot)))

    return lines if lines else [gray]


def tile_line_strip(
    line: Image.Image,
    tile_width: int = IMAGE_WIDTH,
    tile_height: int = IMAGE_HEIGHT,
) -> List[Image.Image]:
    """
    Scale a line strip to tile_height (preserving aspect ratio) then slice it
    left-to-right into tile_width-wide chunks.  The last tile is right-padded
    with white.  Converts any-width lines into fixed 128 × 32 model inputs.
    """
    scale = tile_height / line.height
    scaled_width = max(tile_width, int(line.width * scale))
    scaled = line.resize((scaled_width, tile_height), Image.LANCZOS)

    tiles: List[Image.Image] = []
    for x in range(0, scaled_width, tile_width):
        tile = Image.new("L", (tile_width, tile_height), color=255)
        chunk = scaled.crop((x, 0, min(x + tile_width, scaled_width), tile_height))
        tile.paste(chunk, (0, 0))
        tiles.append(tile)
    return tiles


def _run_tiles(
    tiles: List[Image.Image],
    model: torch.nn.Module,
    lm: Optional[CharNgramLM] = None,
    beam_width: int = 1,
    lm_weight: float = 0.5,
) -> Tuple[str, float]:
    """
    Run a list of image tiles through the model and concatenate decoded text.

    Uses beam search when lm is provided or beam_width > 1, otherwise greedy.
    Returns (full_text, mean_confidence).
    """
    use_beam = lm is not None or beam_width > 1
    texts: List[str] = []
    confidences: List[float] = []

    for tile in tiles:
        arr = torch.from_numpy(np.array(tile)).float() / 255.0
        tensor = arr.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        with torch.no_grad():
            logits = model(tensor)
        if use_beam:
            text, conf = ctc_beam_decode(logits, lm=lm, beam_width=beam_width, lm_weight=lm_weight)
        else:
            text, conf = ctc_greedy_decode(logits)
        texts.append(text)
        confidences.append(conf)

    return "".join(texts), (sum(confidences) / len(confidences) if confidences else 0.0)


# ---------------------------------------------------------------------------
# 6. INFERENCE PIPELINE
# ---------------------------------------------------------------------------


def predict_pdf_first_page(
    pdf_path: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    lm: Optional[CharNgramLM] = None,
    beam_width: int = 1,
    lm_weight: float = 0.5,
) -> None:
    """
    Run OCR on the first page of a PDF and print line-by-line results.

    Pipeline:
      1. Render the first page at 150 DPI.
      2. Segment into text-line strips (valley detection on ink profile).
      3. Tile each strip into 128 × 32 chunks.
      4. Decode each chunk (greedy, or beam search if lm / beam_width > 1).
      5. Print per-line and aggregate results.
    """
    model_path = Path(model_path)
    decoder = f"beam (width={beam_width}, lm_weight={lm_weight})" if (lm or beam_width > 1) else "greedy"
    print(f"Model   : {model_path.resolve()}")
    print(f"PDF     : {Path(pdf_path).resolve()}")
    print(f"Decoder : {decoder}")

    doc = fitz.open(pdf_path)
    if doc.page_count == 0:
        raise ValueError("No pages found in PDF.")
    pix = doc[0].get_pixmap(dpi=150)
    page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
    print(f"Page    : {page.width} x {page.height} px")

    lines = segment_page_into_lines(page)
    print(f"Lines   : {len(lines)} detected")

    print("Loading model…")
    model = load_model(model_path)

    print()
    all_parts: List[str] = []
    total_conf = 0.0
    for i, line_img in enumerate(lines, 1):
        tiles = tile_line_strip(line_img)
        line_text, conf = _run_tiles(tiles, model, lm=lm, beam_width=beam_width, lm_weight=lm_weight)
        all_parts.append(line_text)
        total_conf += conf
        print(
            f"  Line {i:2d}  [{len(tiles)} tile(s), {line_img.width}x{line_img.height}px]"
            f"  conf={conf:.1%}  -> {line_text!r}"
        )

    full_text = "\n".join(all_parts)
    avg_conf = total_conf / len(lines)

    print()
    print("=" * 60)
    print(f"Full text ({len(full_text.replace(chr(10), ''))} chars, avg conf {avg_conf:.1%}):")
    print(full_text if full_text.strip() else "(empty — model needs more training data)")
    if full_text.strip():
        codepoints = " ".join(f"U+{ord(c):04X}" for c in full_text if c != "\n")
        print(f"Unicode : {codepoints}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 7. ONNX EXPORT
# ---------------------------------------------------------------------------


def export_to_onnx(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = "javanese_ocr.onnx",
) -> None:
    """
    Export the trained CRNN to ONNX for TypeScript / ONNX Runtime Web.

    Input  : float32[batch, 1, 32, 128]  — greyscale, normalised to [0, 1]
    Output : float32[batch, time, num_classes]  — raw logits; apply softmax
             then greedy CTC decode in JavaScript.

    Note: the LM (Strategy 4) runs in Python only.  For TypeScript, implement
    a character n-gram lookup table and add the LM score to the beam scores
    during JS-side CTC decoding.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    print(f"Loading model : {model_path.resolve()}")
    model = load_model(model_path)
    model.eval()

    dummy_input = torch.zeros(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    print("Exporting to ONNX…")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch", 1: "time"}},
        verbose=False,
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved  : {output_path.resolve()} ({size_kb:.1f} KB)")
    print()
    print("TypeScript (onnxruntime-web / onnxruntime-node):")
    print("  const session = await InferenceSession.create('javanese_ocr.onnx');")
    print("  // Input:  Float32Array[1, 1, 32, 128], values in [0, 1]")
    print("  // Output: Float32Array[1, T, num_classes] — greedy CTC decode")
    print(f"  // Alphabet: {NUM_CLASSES} classes, index 0 = CTC blank")


# ---------------------------------------------------------------------------
# 8. DATA GENERATION  (Strategy 1)
# ---------------------------------------------------------------------------


def _load_fonts() -> List[Path]:
    """Find and return preferred Javanese TTF fonts (falls back to any .ttf)."""
    candidates = list(Path(".").rglob("*.ttf"))
    if not candidates:
        raise RuntimeError("No .ttf fonts found.  Add NotoSansJavanese-Regular.ttf.")
    preferred = ["NotoSansJavanese-Regular.ttf", "TuladhaJejeg_gr.ttf", "TuladhaJejeg-Regular.ttf"]
    selected = [next((p for p in candidates if p.name == name), None) for name in preferred]
    selected = [p for p in selected if p is not None]
    return selected or [candidates[0]]


def extract_background_patches(
    pdf_path: str,
    num_patches: int = 500,
    dpi: int = 150,
) -> List[Image.Image]:
    """
    Sample random IMAGE_HEIGHT × IMAGE_WIDTH greyscale patches from a PDF.
    (Strategy 1b — Manuscript texture backgrounds)

    Only keeps patches whose mean brightness is above 100 (avoids heavy-ink
    text areas being used as backgrounds, which would obscure drawn text).
    """
    doc = fitz.open(pdf_path)
    patches: List[Image.Image] = []
    attempts = 0
    max_attempts = num_patches * 20

    while len(patches) < num_patches and attempts < max_attempts:
        attempts += 1
        page_num = random.randint(0, doc.page_count - 1)
        pix = doc[page_num].get_pixmap(dpi=dpi)
        page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
        if page.width < IMAGE_WIDTH or page.height < IMAGE_HEIGHT:
            continue
        x = random.randint(0, page.width - IMAGE_WIDTH)
        y = random.randint(0, page.height - IMAGE_HEIGHT)
        patch = page.crop((x, y, x + IMAGE_WIDTH, y + IMAGE_HEIGHT))
        if np.array(patch).mean() > 100:
            patches.append(patch)

    return patches


def generate_synthetic_data(
    output_dir: str,
    num_samples: int = 1000,
    noise_level: float = 0.0,
    noise_type: str = "gaussian",
) -> None:
    """Generate synthetic training images using random Javanese character sequences."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} synthetic samples in '{output_dir}'…")
    fonts = _load_fonts()

    for i in range(num_samples):
        font_path = fonts[i % len(fonts)]
        try:
            font = ImageFont.truetype(str(font_path), size=24)
        except OSError as exc:
            print(f"Could not load '{font_path}': {exc}. Skipping {i}.")
            continue

        text = "".join(random.choices(JAVANESE_CHARS, k=random.randint(3, 6)))
        img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
        ImageDraw.Draw(img).text((5, 2), text, font=font, fill=0)
        if noise_level > 0:
            img = add_noise_to_image(img, noise_level, noise_type)
        img.save(os.path.join(output_dir, f"synth_{i}.png"))
        Path(os.path.join(output_dir, f"synth_{i}.txt")).write_text(text, encoding="utf-8")

    print("Generation complete.")


def generate_from_corpus(
    corpus_path: str,
    output_dir: str,
    num_samples: int = 1000,
    background_pdf: Optional[str] = None,
    noise_level: float = 0.0,
    noise_type: str = "gaussian",
) -> None:
    """
    Generate synthetic training images from a real Javanese text corpus.
    (Strategy 1a + 1b)

    Compared to generate_synthetic_data():
    - Text sequences come from the corpus, so character co-occurrences match
      real Javanese — the model learns which characters actually follow each other.
    - If background_pdf is given, each image is rendered on a parchment patch
      sampled from that PDF rather than a white background, closing the domain
      gap between synthetic training data and real manuscript scans.
    - Wide augmentation variety: font sizes, positions, ink shades, rotation,
      blur, noise, and background brightness are all randomised per sample.

    Parameters:
        corpus_path     Path to a UTF-8 text file containing Javanese Unicode text.
        output_dir      Where to write PNG + TXT pairs.
        num_samples     How many training images to generate.
        background_pdf  Optional PDF to sample manuscript background textures from.
        noise_level     Additional noise strength (0 = none).
        noise_type      'gaussian', 'salt_pepper', or 'blur'.
    """
    # --- Load and tokenise corpus ---
    text = Path(corpus_path).read_text(encoding="utf-8")
    javanese_set = set(JAVANESE_CHARS)

    sequences: List[str] = []
    current = ""
    for ch in text:
        if ch in javanese_set:
            current += ch
        else:
            if len(current) >= 3:
                sequences.append(current)
            current = ""
    if len(current) >= 3:
        sequences.append(current)

    if not sequences:
        raise ValueError(
            f"No Javanese character sequences (U+A98F-U+A9B1) found in '{corpus_path}'.\n"
            "Check that the file uses Unicode Javanese script, not romanised transliteration."
        )

    total_chars = sum(len(s) for s in sequences)
    print(f"Corpus   : {len(sequences):,} sequences, {total_chars:,} Javanese characters")

    fonts = _load_fonts()

    # --- Extract background patches at multiple brightness levels ---
    bg_patches: Optional[List[Image.Image]] = None
    if background_pdf:
        print(f"Sampling background patches from '{background_pdf}'...")
        bg_patches = extract_background_patches(
            background_pdf, num_patches=min(4000, num_samples)
        )
        if bg_patches:
            print(f"  Got {len(bg_patches)} usable patches.")
        else:
            print("  Warning: no usable patches found - using white backgrounds.")
            bg_patches = None

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} corpus-based samples in '{output_dir}'...")

    for i in range(num_samples):
        # --- Random text chunk (3-14 chars) ---
        seq = random.choice(sequences)
        max_len = min(14, len(seq))
        min_len = min(3, len(seq))
        if len(seq) > min_len:
            start = random.randint(0, len(seq) - min_len)
            length = random.randint(min_len, min(max_len, len(seq) - start))
            chunk = seq[start : start + length]
        else:
            chunk = seq

        # --- Background ---
        # 15% chance of plain white even when PDF available (covers printed/digital docs)
        use_white = (bg_patches is None) or (random.random() < 0.15)
        if use_white:
            bg_brightness = random.randint(230, 255)
            img = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), color=bg_brightness)
        else:
            bg_arr = np.array(random.choice(bg_patches)).astype(np.float32)
            # Randomise how much we lighten the patch
            scale  = random.uniform(0.40, 0.70)
            offset = random.uniform(80, 140)
            bg_arr = np.clip(bg_arr * scale + offset, 0, 255).astype(np.uint8)
            img = Image.fromarray(bg_arr).convert("L")

        # --- Font & size ---
        font_path = fonts[i % len(fonts)]
        font_size = random.randint(16, 28)
        try:
            font = ImageFont.truetype(str(font_path), size=font_size)
        except OSError:
            font = ImageFont.load_default()

        # --- Text position jitter ---
        x_off = random.randint(1, 10)
        y_off = random.randint(0, max(0, IMAGE_HEIGHT - font_size - 2))

        # --- Ink colour (near-black to faded brown) ---
        ink = random.randint(8, 60)

        ImageDraw.Draw(img).text((x_off, y_off), chunk, font=font, fill=ink)

        # --- Random augmentations ---

        # Slight rotation to simulate manuscript tilt (40% of samples)
        if random.random() < 0.40:
            angle = random.uniform(-4.0, 4.0)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        # Soft blur (ink spread / out-of-focus scan) — 35% of samples
        if random.random() < 0.35:
            radius = random.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # Gaussian pixel noise — 40% of samples
        if random.random() < 0.40:
            arr = np.array(img).astype(np.float32)
            sigma = random.uniform(3, 18)
            arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        # Salt-and-pepper (ink flakes / paper damage) — 20% of samples
        if random.random() < 0.20:
            arr = np.array(img)
            density = random.uniform(0.002, 0.015)
            noise = np.random.rand(*arr.shape)
            arr[noise < density] = 0
            arr[noise > 1 - density] = 255
            img = Image.fromarray(arr)

        # Random horizontal contrast stretch — 25% of samples
        if random.random() < 0.25:
            from PIL import ImageEnhance
            factor = random.uniform(0.6, 1.5)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Extra noise if caller requested it
        if noise_level > 0:
            img = add_noise_to_image(img, noise_level, noise_type)

        img.save(os.path.join(output_dir, f"corpus_{i}.png"))
        Path(os.path.join(output_dir, f"corpus_{i}.txt")).write_text(chunk, encoding="utf-8")

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,} / {num_samples:,} generated...")

    print(f"Done. {num_samples} samples written to '{output_dir}'.")


def add_noise_to_image(img: Image.Image, noise_level: float, noise_type: str) -> Image.Image:
    """Add noise to simulate manuscript degradation (gaussian, salt_pepper, blur)."""
    arr = np.array(img).astype(np.float32)
    if noise_type == "gaussian":
        arr = np.clip(arr + np.random.normal(0, noise_level * 10, arr.shape), 0, 255).astype(np.uint8)
    elif noise_type == "salt_pepper":
        noise = np.random.rand(*arr.shape)
        arr[noise < noise_level / 100] = 0
        arr[noise > 1 - noise_level / 100] = 255
    elif noise_type == "blur":
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            def gaussian_filter(x, sigma):
                return x
        arr = np.clip(gaussian_filter(arr, sigma=noise_level * 0.1), 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("L")


# ---------------------------------------------------------------------------
# 9. INGESTION
# ---------------------------------------------------------------------------


def process_directory(input_dir: str, output_dir: str) -> None:
    """
    Recursively ingest PDFs and images from input_dir, segment each page into
    text-line strips, tile each strip into IMAGE_WIDTH × IMAGE_HEIGHT chunks,
    and save them as PNGs in output_dir.

    Each PNG needs a matching .txt with ground-truth Javanese text before it
    can be used for training.  Use --mode pseudo_label to auto-generate labels
    for high-confidence strips.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Scanning '{input_dir}' for PDFs and images…")
    saved = 0

    def _process_page(gray: Image.Image, stem: str) -> int:
        count = 0
        for line_idx, line in enumerate(segment_page_into_lines(gray)):
            for tile_idx, tile in enumerate(tile_line_strip(line)):
                tile.save(os.path.join(output_dir, f"{stem}_l{line_idx:03d}_t{tile_idx:02d}.png"))
                count += 1
        return count

    for pdf in glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True):
        print(f"  PDF: {pdf}")
        doc = fitz.open(pdf)
        base = Path(pdf).stem
        for page_num in range(doc.page_count):
            pix = doc[page_num].get_pixmap(dpi=300)
            gray = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")
            n = _process_page(gray, f"{base}_p{page_num:03d}")
            print(f"    page {page_num}: {n} strip(s)")
            saved += n

    image_files = sorted(
        glob.glob(os.path.join(input_dir, "**/*.png"), recursive=True)
        + glob.glob(os.path.join(input_dir, "**/*.jpg"), recursive=True)
    )
    for img_path in image_files:
        print(f"  Image: {img_path}")
        n = _process_page(Image.open(img_path).convert("L"), Path(img_path).stem)
        print(f"    {n} strip(s)")
        saved += n

    print(f"\nSaved {saved:,} strip(s) to '{output_dir}'.")
    print("Next: run --mode pseudo_label to auto-generate labels, or add .txt files manually.")


# ---------------------------------------------------------------------------
# 10. PSEUDO-LABELLING  (Strategy 2 — Self-training)
# ---------------------------------------------------------------------------


def pseudo_label(
    model_path: str | Path,
    unlabeled_dir: str,
    output_dir: str,
    threshold: float = 0.92,
    lm: Optional[CharNgramLM] = None,
    beam_width: int = 1,
    lm_weight: float = 0.5,
) -> None:
    """
    Auto-label manuscript strips using the trained model.
    (Strategy 2 — Self-training / Pseudo-labelling)

    For each PNG in unlabeled_dir that has no matching .txt, runs the model and
    saves a label if the prediction confidence meets the threshold.  Labeled
    pairs (PNG + TXT) are copied to output_dir, ready for --mode train.

    Iterative workflow:
      1. Run pseudo_label with threshold=0.92 → get ~N confident labels.
      2. Retrain with original data + pseudo_labeled.
      3. Run pseudo_label again — the improved model labels more strips.
      4. Repeat until accuracy plateaus.

    Threshold guide:
        0.95+   Very conservative — few labels, high accuracy.
        0.90    Good starting point for a partially trained model.
        0.80    Aggressive — more labels but higher error rate; review samples.

    Parameters:
        model_path    Path to .pth checkpoint.
        unlabeled_dir Directory of PNG strips without .txt labels.
        output_dir    Destination for confident (PNG, TXT) pairs.
        threshold     Minimum confidence to accept a prediction as a label.
        lm            Optional CharNgramLM for beam-search decoding.
        beam_width    Beam width (>1 requires lm or explicit override).
        lm_weight     LM contribution weight (0–1).
    """
    model = load_model(model_path)
    use_beam = lm is not None or beam_width > 1

    png_files = sorted(Path(unlabeled_dir).glob("*.png"))
    unlabeled = [p for p in png_files if not p.with_suffix(".txt").exists()]

    if not unlabeled:
        print("No unlabeled PNGs found (all strips already have .txt files).")
        return

    os.makedirs(output_dir, exist_ok=True)
    total = len(unlabeled)
    labeled = skipped_blank = 0
    conf_sum = 0.0

    decoder_info = f"beam (width={beam_width})" if use_beam else "greedy"
    print(f"Strips to process : {total:,}")
    print(f"Confidence threshold : {threshold:.0%}")
    print(f"Decoder : {decoder_info}")
    print()

    for i, png_path in enumerate(unlabeled, 1):
        img = Image.open(png_path).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        tensor = arr.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)

        if use_beam:
            text, conf = ctc_beam_decode(logits, lm=lm, beam_width=beam_width, lm_weight=lm_weight)
        else:
            text, conf = ctc_greedy_decode(logits)

        if i % 1000 == 0 or i == total:
            print(f"  {i:,}/{total:,} — labeled so far: {labeled:,}")

        if not text.strip():
            skipped_blank += 1
            continue

        if conf >= threshold:
            dest_png = Path(output_dir) / png_path.name
            shutil.copy2(png_path, dest_png)
            dest_png.with_suffix(".txt").write_text(text, encoding="utf-8")
            labeled += 1
            conf_sum += conf

    avg_conf = conf_sum / labeled if labeled else 0.0
    print()
    print("=" * 50)
    print(f"Processed  : {total:,}")
    print(f"Labeled    : {labeled:,} ({labeled/total:.1%}) -> '{output_dir}'")
    print(f"Blank/skip : {skipped_blank:,}")
    print(f"Avg conf   : {avg_conf:.1%}")
    print("=" * 50)
    if labeled > 0:
        print(f"\nNext step:")
        print(f"  python javanese_ocr.py --mode train \\")
        print(f"      --data_dir ./ocr_data {output_dir} --epochs 20")


# ---------------------------------------------------------------------------
# 11. MODEL TRAINING
# ---------------------------------------------------------------------------


def train_model(
    data_dirs: List[str],
    learning_rate: float = 0.001,
    epochs: int = 20,
    output_path: str = "javanese_ocr.pth",
) -> None:
    """Train the CRNN on OCR data from one or more directories."""
    from ocr_collate import collate_fn
    from ocr_dataset import OcrDataset

    print(f"Initialising training from {len(data_dirs)} director(y/ies)…")

    combined_samples: List[Tuple[Path, Path]] = []
    for data_dir in data_dirs:
        dataset = OcrDataset(Path(data_dir))
        if len(dataset) == 0:
            print(f"  Warning: no samples in '{data_dir}'. Skipping.")
            continue
        print(f"  {data_dir}: {len(dataset)} samples")
        combined_samples.extend(dataset.samples)

    if not combined_samples:
        raise RuntimeError(
            "No training samples found.  Each PNG needs a matching .txt label file."
        )

    print(f"Total samples: {len(combined_samples):,}")

    class CombinedDataset(Dataset):
        def __init__(self, samples: List[Tuple[Path, Path]]):
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            img_path, txt_path = self.samples[idx]
            image = Image.open(img_path).convert("L")
            labels = torch.tensor(
                text_to_indices(txt_path.read_text(encoding="utf-8")), dtype=torch.long
            )
            return image, labels

    loader = DataLoader(
        CombinedDataset(combined_samples), batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    model = SimpleCRNN(NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # CTC loss always runs on CPU (DirectML fallback has broken gradient flow).
    # We explicitly move log_probs to CPU; autograd routes gradients back to DEVICE.
    criterion = nn.CTCLoss(blank=0)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for imgs, labels, lengths in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            output_lengths = torch.full((imgs.size(0),), outputs.size(1), dtype=torch.long)
            log_probs = nn.functional.log_softmax(outputs, dim=2).permute(1, 0, 2).cpu()
            loss = criterion(log_probs, labels, output_lengths, lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} — avg loss: {total_loss / len(loader):.4f}")

    save_path = Path(os.getcwd()) / output_path
    torch.save(model.state_dict(), str(save_path))
    print(f"Model saved to: {save_path}")


# ---------------------------------------------------------------------------
# 12. COMMAND LINE INTERFACE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Javanese OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended improvement workflow (no manual labelling):

  # 1. Generate corpus-based training data with manuscript backgrounds
  python javanese_ocr.py --mode generate_from_corpus \\
      --corpus javanese_text.txt --background_pdf PDFA.pdf \\
      --data_dir ./ocr_corpus --num_samples 5000

  # 2. Train the language model
  python javanese_ocr.py --mode train_lm \\
      --corpus javanese_text.txt --output_path javanese_lm.pkl

  # 3. Retrain the CRNN
  python javanese_ocr.py --mode train \\
      --data_dir ./ocr_data ./ocr_corpus --epochs 30 --lr 0.001

  # 4. Ingest the manuscript into strips
  python javanese_ocr.py --mode ingest \\
      --input_dir ./manuscripts --data_dir ./manuscript_strips

  # 5. Auto-label high-confidence strips (pseudo-labelling)
  python javanese_ocr.py --mode pseudo_label \\
      --unlabeled_dir ./manuscript_strips --data_dir ./pseudo_labeled \\
      --lm_path javanese_lm.pkl --threshold 0.92

  # 6. Retrain with expanded dataset, repeat from step 5
  python javanese_ocr.py --mode train \\
      --data_dir ./ocr_data ./ocr_corpus ./pseudo_labeled --epochs 30

  # Predict with LM-assisted beam search
  python javanese_ocr.py --mode predict --pdf PDFA.pdf \\
      --lm_path javanese_lm.pkl --beam_width 10

  # Export to ONNX
  python javanese_ocr.py --mode export_onnx --output_path javanese_ocr.onnx
""",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "generate",
            "generate_with_noise",
            "generate_from_corpus",
            "ingest",
            "train",
            "train_lm",
            "pseudo_label",
            "predict",
            "export_onnx",
        ],
        help="Pipeline stage to run.",
    )

    # --- Input/output paths ---
    parser.add_argument("--pdf", help="PDF to run OCR on (--mode predict).")
    parser.add_argument(
        "--model_path", default=str(DEFAULT_MODEL_PATH),
        help=f"Path to .pth checkpoint. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument("--input_dir", help="Source directory of PDFs/images (--mode ingest).")
    parser.add_argument(
        "--data_dir", nargs="+", default=["./ocr_data"],
        help="Training data directory/ies (PNG + TXT pairs).  Output dir for ingest/pseudo_label.",
    )
    parser.add_argument("--output_path", default="javanese_ocr.pth",
                        help="Output path for model or ONNX file.")
    parser.add_argument("--unlabeled_dir",
                        help="Directory of unlabeled PNG strips (--mode pseudo_label).")

    # --- Corpus & LM ---
    parser.add_argument("--corpus",
                        help="Path to UTF-8 Javanese text corpus (generate_from_corpus / train_lm).")
    parser.add_argument("--background_pdf",
                        help="PDF to sample manuscript background textures from (generate_from_corpus).")
    parser.add_argument("--lm_path",
                        help="Path to trained CharNgramLM .pkl file (predict / pseudo_label).")
    parser.add_argument("--lm_n", type=int, default=3,
                        help="N-gram order for train_lm. Default: 3.")
    parser.add_argument("--beam_width", type=int, default=10,
                        help="Beam width for CTC beam search. Default: 10.")
    parser.add_argument("--lm_weight", type=float, default=0.5,
                        help="LM score weight for beam search (0=CTC only, 1=equal). Default: 0.5.")

    # --- Generation ---
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Synthetic samples to generate. Default: 1000.")
    parser.add_argument("--noise_level", type=float, default=0.0,
                        help="Noise intensity (0–100). Default: 0.")
    parser.add_argument("--noise_type", choices=["gaussian", "salt_pepper", "blur"],
                        default="gaussian", help="Noise type. Default: gaussian.")

    # --- Training ---
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate. Default: 0.001.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs. Default: 20.")

    # --- Pseudo-labelling ---
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="Minimum confidence for pseudo-label acceptance. Default: 0.92.")

    args = parser.parse_args()

    # Load LM if requested (used by predict and pseudo_label)
    lm: Optional[CharNgramLM] = None
    if args.lm_path:
        print(f"Loading language model: {args.lm_path}")
        lm = CharNgramLM.load(args.lm_path)
        print(f"  {lm.n}-gram LM, vocab={len(lm._vocab)} chars")

    # Dispatch
    if args.mode == "predict":
        if not args.pdf:
            parser.error("--pdf is required for --mode predict")
        predict_pdf_first_page(
            args.pdf,
            model_path=args.model_path,
            lm=lm,
            beam_width=args.beam_width,
            lm_weight=args.lm_weight,
        )

    elif args.mode == "export_onnx":
        out = args.output_path if args.output_path.endswith(".onnx") else "javanese_ocr.onnx"
        export_to_onnx(model_path=args.model_path, output_path=out)

    elif args.mode == "generate":
        generate_synthetic_data(
            output_dir=args.data_dir[0],
            num_samples=args.num_samples,
        )

    elif args.mode == "generate_with_noise":
        generate_synthetic_data(
            output_dir=args.data_dir[0],
            num_samples=args.num_samples,
            noise_level=args.noise_level,
            noise_type=args.noise_type,
        )

    elif args.mode == "generate_from_corpus":
        if not args.corpus:
            parser.error("--corpus is required for --mode generate_from_corpus")
        generate_from_corpus(
            corpus_path=args.corpus,
            output_dir=args.data_dir[0],
            num_samples=args.num_samples,
            background_pdf=args.background_pdf,
            noise_level=args.noise_level,
            noise_type=args.noise_type,
        )

    elif args.mode == "ingest":
        if not args.input_dir:
            parser.error("--input_dir is required for --mode ingest")
        process_directory(args.input_dir, args.data_dir[0])

    elif args.mode == "train_lm":
        if not args.corpus:
            parser.error("--corpus is required for --mode train_lm")
        out = args.output_path if args.output_path.endswith(".pkl") else "javanese_lm.pkl"
        train_lm(
            corpus_path=args.corpus,
            output_path=out,
            n=args.lm_n,
        )

    elif args.mode == "pseudo_label":
        if not args.unlabeled_dir:
            parser.error("--unlabeled_dir is required for --mode pseudo_label")
        pseudo_label(
            model_path=args.model_path,
            unlabeled_dir=args.unlabeled_dir,
            output_dir=args.data_dir[0],
            threshold=args.threshold,
            lm=lm,
            beam_width=args.beam_width,
            lm_weight=args.lm_weight,
        )

    elif args.mode == "train":
        train_model(
            data_dirs=args.data_dir,
            learning_rate=args.lr,
            epochs=args.epochs,
            output_path=args.output_path,
        )
