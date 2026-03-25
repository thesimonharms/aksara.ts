"""
hitl.py — Human-in-the-Loop labelling tool for Javanese OCR strips.

Shows you the strip image + the model's best guess.  You confirm, correct,
or skip each one.  Labels are saved to --output_dir as ground-truth pairs
for use in the next training round.

Strips are shown in ascending confidence order (hardest first) so your time
is spent where it matters most.

Usage:
    python hitl.py --unlabeled_dir training\\manuscript_strips \\
                   --model_path   training\\javanese_ocr.pth   \\
                   --output_dir   training\\human_labeled       \\
                   --lm_path      training\\javanese_lm.pkl     \\
                   --rate 1

Controls:
    <Enter>         Accept model prediction as-is
    <text> <Enter>  Replace with your corrected Javanese text
    s               Skip (do not save this strip)
    q               Quit and save progress summary
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Reuse model + decode from javanese_ocr.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from javanese_ocr import (
    SimpleCRNN,
    CharNgramLM,
    ctc_beam_decode,
    ctc_greedy_decode,
    JAVANESE_CHARS,
    NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    DEFAULT_MODEL_PATH,
)

import torchvision.transforms as T

TO_TENSOR = T.Compose([
    T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    T.ToTensor(),
])


def load_model(model_path: str) -> torch.nn.Module:
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SimpleCRNN(num_classes=NUM_CLASSES)
    # checkpoint is either a raw state_dict or a wrapped dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def predict_strip(
    img: Image.Image,
    model: torch.nn.Module,
    lm=None,
    beam_width: int = 10,
    lm_weight: float = 0.3,
):
    """Return (text, confidence 0-1)."""
    tensor = TO_TENSOR(img).unsqueeze(0)          # [1, 1, H, W]
    with torch.no_grad():
        logits = model(tensor)                     # [1, T, C]
    if lm and beam_width > 1:
        text, conf = ctc_beam_decode(logits, lm=lm,
                                     beam_width=beam_width,
                                     lm_weight=lm_weight)
    else:
        text, conf = ctc_greedy_decode(logits)
    return text, conf


def open_image(img_path: Path) -> None:
    """Open the strip in the system default image viewer."""
    try:
        img = Image.open(img_path)
        # Scale up 4x so it's actually visible
        w, h = img.size
        img = img.resize((w * 4, h * 4), Image.NEAREST)
        img.show(title=img_path.name)
    except Exception as e:
        print(f"  [!] Could not open image: {e}")


def collect_candidates(
    unlabeled_dir: Path,
    output_dir: Path,
    model,
    lm,
    beam_width: int,
    lm_weight: float,
    already_done: set,
) -> list:
    """
    Score every unlabeled strip, skip ones already in output_dir,
    return list sorted by confidence ascending (hardest first).
    """
    print("Scoring strips to find the hardest examples…")
    strips = sorted(unlabeled_dir.glob("*.png"))
    results = []
    for i, png in enumerate(strips):
        if png.stem in already_done:
            continue
        try:
            img = Image.open(png).convert("L")
            text, conf = predict_strip(img, model, lm, beam_width, lm_weight)
            results.append((conf, png, text))
        except Exception:
            continue
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(strips)} scored…")

    results.sort(key=lambda x: x[0])   # ascending: lowest conf first
    print(f"  {len(results)} strips ready (sorted by difficulty).")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Human-in-the-loop labeller for Javanese OCR strips."
    )
    parser.add_argument("--unlabeled_dir", required=True,
                        help="Directory of PNG manuscript strips.")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH),
                        help="Path to trained .pth checkpoint.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save confirmed (PNG + TXT) pairs.")
    parser.add_argument("--lm_path", default=None,
                        help="Path to CharNgramLM .pkl file.")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--lm_weight", type=float, default=0.3)
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Target labels per minute (paces the session). Default: 1.")
    parser.add_argument("--session", type=int, default=0,
                        help="Max labels this session (0 = unlimited).")
    args = parser.parse_args()

    unlabeled_dir = Path(args.unlabeled_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interval = 60.0 / args.rate   # seconds between labels

    # Load model
    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path)

    # Load LM
    lm = None
    if args.lm_path and Path(args.lm_path).exists():
        print(f"Loading LM: {args.lm_path}")
        lm = CharNgramLM.load(args.lm_path)
        print(f"  {lm.n}-gram LM, vocab={len(lm._vocab)} chars")

    # Which strips already have labels in output_dir?
    already_done = {p.stem for p in output_dir.glob("*.txt")}
    print(f"Already labeled: {len(already_done)} strips.")

    # Score and sort all candidates
    candidates = collect_candidates(
        unlabeled_dir, output_dir, model, lm,
        args.beam_width, args.lm_weight, already_done
    )

    if not candidates:
        print("Nothing left to label. All done!")
        return

    # Session loop
    accepted = 0
    corrected = 0
    skipped = 0
    total_shown = 0
    session_limit = args.session if args.session > 0 else float("inf")

    print()
    print("=" * 60)
    print("  HITL Labelling Session")
    print(f"  Rate: {args.rate}/min  |  Interval: {interval:.0f}s")
    print("  Controls: Enter=accept  text+Enter=correct  s=skip  q=quit")
    print("=" * 60)
    print()

    next_allowed = time.monotonic()

    for conf, png, model_text in candidates:
        if total_shown >= session_limit:
            print(f"\nSession limit of {args.session} reached.")
            break

        # Rate limiting
        now = time.monotonic()
        wait = next_allowed - now
        if wait > 0:
            print(f"  (next strip in {wait:.0f}s — take your time)")
            time.sleep(wait)
        next_allowed = time.monotonic() + interval

        total_shown += 1
        print(f"\n[{total_shown}]  conf={conf*100:.1f}%  file={png.name}")
        print(f"  Model: {model_text if model_text else '(blank)'}")

        open_image(png)

        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted.")
            break

        if raw.lower() == "q":
            print("Quitting.")
            break
        elif raw.lower() == "s":
            skipped += 1
            print("  Skipped.")
            continue
        elif raw == "":
            # Accept model prediction
            label = model_text
            accepted += 1
            tag = "accepted"
        else:
            label = raw
            corrected += 1
            tag = "corrected"

        if not label:
            print("  No label to save (model was blank and nothing entered). Skipping.")
            skipped += 1
            continue

        # Save PNG + TXT to output_dir
        dst_png = output_dir / png.name
        dst_txt = dst_png.with_suffix(".txt")
        shutil.copy2(png, dst_png)
        dst_txt.write_text(label, encoding="utf-8")
        print(f"  Saved ({tag}): {label}")

    # Summary
    print()
    print("=" * 60)
    print(f"  Session complete")
    print(f"  Accepted : {accepted}")
    print(f"  Corrected: {corrected}")
    print(f"  Skipped  : {skipped}")
    print(f"  Total saved: {accepted + corrected}")
    print(f"  Output dir : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
