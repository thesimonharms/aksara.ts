#!/usr/bin/env python3
"""
app.py — HF Space front-end for the Javanese TrOCR fine-tune pipeline.

Exposes three buttons driven by background threads (so the UI never blocks
on a multi-hour training run); progress is visible in the Space **Logs** tab,
which captures stdout/stderr from these worker threads.

Run locally (no Space):     python app.py
In a HF Space:              started by Dockerfile ENTRYPOINT.

Required Secrets (set in Space → Settings → Repository secrets):
    HF_TOKEN        — write-scoped token (model + dataset push)
    HF_USERNAME     — your HF username / org name

Optional Secrets:
    BASE_MODEL                      (default microsoft/trocr-base-handwritten)
    EPOCHS                          (default 5)
    PER_DEVICE_TRAIN_BATCH_SIZE     (default 8)
    DATASET_NAME                    (HF Hub dataset id; skips local generation)
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path

# Make the sibling finetune_trocr importable when the container runs from /app
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

from finetune_trocr import run_pipeline  # noqa: E402


BASE_MODEL_DEFAULT = os.environ.get("BASE_MODEL", "microsoft/trocr-base-handwritten")
EPOCHS_DEFAULT = int(os.environ.get("EPOCHS", "5"))
BATCH_DEFAULT = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "8"))
DATASET_NAME_DEFAULT = os.environ.get("DATASET_NAME", "")


# ---------------------------------------------------------------------------
# Shared background-task state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_state: dict[str, str] = {
    "phase": "idle",        # idle | running | ok | error
    "task":  "",            # generate | push_dataset | train
    "message": "",
    "running": "false",
}


def _set_state(phase: str, task: str, message: str, running: bool) -> None:
    with _state_lock:
        _state.update(phase=phase, task=task, message=message,
                      running="true" if running else "false")


def _guarded_start(task: str, fn, *args, **kwargs) -> str:
    with _state_lock:
        if _state["running"] == "true":
            return f"A job is already running ({_state['task']}). Watch Space → Logs. State: {_state['phase']}"
    _set_state("running", task, "queued", True)

    def worker():
        try:
            result = fn(*args, **kwargs)
            _set_state("ok", task, str(result), False)
        except Exception as exc:
            _set_state("error", task, f"{exc}\n{traceback.format_exc()}", False)

    threading.Thread(target=worker, daemon=True).start()
    return (
        f"Started **{task}** in the background. Watch **Space → Logs** for live output.\n\n"
        "Return here later — the status panel below updates automatically."
    )


# ---------------------------------------------------------------------------
# Pipeline tasks
# ---------------------------------------------------------------------------
def task_generate(num_train: int, num_val: int, fonts_dir: str, pdfs_dir: str, output_dir: str) -> str:
    """Run the synthetic dataset generator. Throws on nonzero exit."""
    args = [
        sys.executable, "generate_trocr_dataset.py",
        "--fonts_dir", fonts_dir,
        "--pdfs_dir", pdfs_dir,
        "--output_dir", output_dir,
        "--num_train", str(int(num_train)),
        "--num_val", str(int(num_val)),
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"generate_trocr_dataset.py exit {proc.returncode}\nSTDERR:\n{proc.stderr}")
    return f"Generate OK:\n{proc.stdout[-2000:]}"


def task_push_dataset(dataset_dir: str) -> str:
    """Load the local imagefolder dataset, then push it to HF Hub."""
    from datasets import DatasetDict, load_dataset
    hf_token = os.environ.get("HF_TOKEN")
    hf_user = os.environ.get("HF_USERNAME")
    if not (hf_token and hf_user):
        raise RuntimeError("HF_TOKEN / HF_USERNAME must be set as Space Secrets to push the dataset.")
    dd = Path(dataset_dir)
    if not dd.exists():
        raise RuntimeError(f"Dataset dir not found: {dd}. Run 'Generate' first.")
    train = load_dataset("imagefolder", data_dir=str(dd / "train"), split="train")
    val = load_dataset("imagefolder", data_dir=str(dd / "validation"), split="train")
    raw = DatasetDict({"train": train, "validation": val})
    target = f"{hf_user}/javanese-dataset"
    raw.push_to_hub(target, token=hf_token, private=False)
    return f"Dataset pushed to {target}"


def task_train(dataset_name: str, epochs: int, batch_size: int, base_model: str) -> str:
    overrides = {
        "dataset_name": dataset_name or None,
        "epochs": int(epochs) if epochs else 5,
        "batch_size": int(batch_size) if batch_size else 8,
        "base_model": base_model or BASE_MODEL_DEFAULT,
    }
    return run_pipeline(overrides)


# ---------------------------------------------------------------------------
# Gradio entry points (just dispatch to background threads)
# ---------------------------------------------------------------------------
def start_generate(num_train, num_val, fonts_dir, pdfs_dir, output_dir):
    return _guarded_start(
        "generate dataset", task_generate,
        num_train, num_val, fonts_dir, pdfs_dir, output_dir,
    )


def start_push_dataset(dataset_dir):
    return _guarded_start("push dataset", task_push_dataset, dataset_dir)


def start_train(dataset_name, epochs, batch_size, base_model):
    return _guarded_start(
        "fine-tune model", task_train,
        dataset_name, epochs, batch_size, base_model,
    )


def env_status_md() -> str:
    tok = "SET" if os.environ.get("HF_TOKEN") else "<b>MISSING</b>"
    user = os.environ.get("HF_USERNAME", "<b>MISSING</b>")
    return (
        "| Secret | Status |\n|---|---|\n"
        f"| `HF_TOKEN` | {tok} |\n"
        f"| `HF_USERNAME` | {user} |\n"
        "\nOptional: `BASE_MODEL`, `EPOCHS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `DATASET_NAME`."
    )


def poll_status() -> str:
    with _state_lock:
        return (
            f"Phase: **{_state['phase']}**  |  Task: **{_state['task'] or '—'}**  |  Running: **{_state['running']}**\n\n"
            f"Message:\n```\n{_state['message'][-1500:]}\n```"
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Javanese TrOCR Fine-tune") as demo:
    gr.Markdown(
        "# TrOCR Fine-tune — Javanese Aksara\n"
        "Generate a synthetic dataset from your fonts/PDFs, optionally publish it to the HF Hub, "
        "then fine-tune `microsoft/trocr-base-handwritten` on it. The fine-tuned model is pushed to "
        "your HF Hub automatically."
    )
    gr.Markdown(f"### Space secrets\n{env_status_md()}")

    with gr.Accordion("1. Generate dataset", open=True):
        with gr.Row():
            num_train = gr.Slider(500, 50000, value=5000, step=500, label="Train samples")
            num_val = gr.Slider(50, 5000, value=500, step=50, label="Val samples")
            fonts_dir = gr.Textbox("fonts", label="Fonts dir")
            pdfs_dir = gr.Textbox("pdfs", label="PDFs dir")
            gen_output_dir = gr.Textbox("trocr_dataset", label="Output dir")
        gen_btn = gr.Button("1. Generate dataset from fonts + PDFs", variant="secondary")
        gen_out = gr.Markdown("")
        gen_btn.click(start_generate,
                      [num_train, num_val, fonts_dir, pdfs_dir, gen_output_dir],
                      gen_out)

    with gr.Accordion("2. Push dataset to HF Hub", open=False):
        push_dir = gr.Textbox("trocr_dataset", label="Dataset dir")
        push_btn = gr.Button("2. Push dataset to HF Hub", variant="secondary")
        push_out = gr.Markdown("")
        push_btn.click(start_push_dataset, [push_dir], push_out)

    with gr.Accordion("3. Fine-tune TrOCR", open=True):
        with gr.Row():
            ds_name = gr.Textbox(DATASET_NAME_DEFAULT,
                                 label="HF dataset name (uses local dir if blank)")
            epochs_ui = gr.Slider(1, 20, value=EPOCHS_DEFAULT, step=1, label="Epochs")
            batch_ui = gr.Slider(1, 32, value=BATCH_DEFAULT, step=1, label="Per-device batch size")
            base_model_ui = gr.Textbox(BASE_MODEL_DEFAULT, label="Base model")
        train_btn = gr.Button("3. Run fine-tuning", variant="primary")
        train_out = gr.Markdown("")
        train_btn.click(start_train,
                       [ds_name, epochs_ui, batch_ui, base_model_ui],
                       train_out)

    gr.Markdown("### Live status")
    status_box = gr.Markdown(poll_status())
    timer = gr.Timer(5)
    timer.tick(poll_status, outputs=status_box)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)  # one pipeline at a time
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, prevent_thread_lock=False)