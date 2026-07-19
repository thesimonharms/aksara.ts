"""Pick CUDA / Intel XPU / CPU for local TrOCR train + inference.

Prefer Intel Arc (torch.xpu) when present so eGPU / NAS setups use the
dedicated card instead of falling back to CPU.
"""

from __future__ import annotations

import os
import sys

import torch


def xpu_available() -> bool:
    return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def pick_device() -> str:
    if xpu_available():
        return "xpu"
    if cuda_available():
        return "cuda"
    return "cpu"


def device_name(device: str | None = None) -> str:
    d = device or pick_device()
    if d == "xpu":
        try:
            return torch.xpu.get_device_name(0)
        except Exception:
            return "Intel XPU"
    if d == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA GPU"
    return "CPU"


def xpu_mem_gb(device_index: int = 0) -> float | None:
    if not xpu_available():
        return None
    try:
        props = torch.xpu.get_device_properties(device_index)
        total = getattr(props, "total_memory", None)
        if total is None:
            return None
        return float(total) / (1024**3)
    except Exception:
        return None


def recommend_batch_size(default: int = 16) -> int:
    """Heuristic batch for TrOCR-base on Arc Pro / Battlemage."""
    override = os.environ.get("TROCR_BATCH_SIZE", "").strip()
    if override.isdigit():
        return max(1, int(override))
    mem = xpu_mem_gb(0)
    if mem is None:
        return default
    # Linux + SDPA/bf16 can usually hold more than Windows eager/fp32.
    if sys.platform.startswith("linux") and mem >= 20:
        return 24
    if mem >= 20:
        return 16
    if mem >= 12:
        return 8
    return 4


def use_amp(device: str | None = None) -> tuple[bool, bool]:
    """Return (fp16, bf16) flags for TrainingArguments.

    Windows XPU: fp32 only (bf16 TrOCR backward was broken in our Arc B60 tests).
    Linux XPU: try bf16 unless TROCR_FORCE_FP32=1.
    CUDA: fp16 (Jobs recipe).
    """
    d = device or pick_device()
    if d == "xpu":
        if os.environ.get("TROCR_FORCE_FP32", "").strip() in ("1", "true", "yes"):
            return False, False
        if sys.platform.startswith("linux"):
            return False, True
        return False, False
    if d == "cuda":
        return True, False
    return False, False


def attn_implementation(device: str | None = None) -> str | None:
    """Attention backend for from_pretrained.

    Windows XPU: force eager (SDPA backward hangs/crashes for TrOCR).
    Linux XPU: default (SDPA) unless TROCR_ATTN=eager|sdpa is set.
    """
    d = device or pick_device()
    override = os.environ.get("TROCR_ATTN", "").strip().lower()
    if override in ("eager", "sdpa", "flash_attention_2"):
        return override
    if d == "xpu" and sys.platform == "win32":
        return "eager"
    return None


def dataloader_kwargs(device: str | None = None) -> dict:
    """DataLoader settings. pin_memory is CUDA-only; win32 must use workers=0."""
    d = device or pick_device()
    if sys.platform == "win32":
        return {"dataloader_num_workers": 0, "dataloader_pin_memory": False}
    if d == "cuda":
        return {"dataloader_num_workers": 4, "dataloader_pin_memory": True}
    if d == "xpu":
        return {"dataloader_num_workers": 4, "dataloader_pin_memory": False}
    return {"dataloader_num_workers": 0, "dataloader_pin_memory": False}


def configure_xpu_runtime() -> None:
    """Best-effort Level Zero / allocator knobs before heavy XPU work."""
    if not xpu_available():
        return
    os.environ.setdefault("SYCL_UR_USE_LEVEL_ZERO_V2", "0")
    os.environ.setdefault("UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS", "1")
    # Prefer discrete Arc when multiple Level Zero devices exist (iGPU + dGPU).
    if "ONEAPI_DEVICE_SELECTOR" not in os.environ:
        # Leave unset by default; detect_gpu.sh / entrypoint may set level_zero:N.
        pass
    try:
        frac = float(os.environ.get("TROCR_XPU_MEM_FRACTION", "0.90"))
        if hasattr(torch.xpu, "set_per_process_memory_fraction"):
            torch.xpu.set_per_process_memory_fraction(frac, device=0)
        elif hasattr(torch.xpu, "memory") and hasattr(
            torch.xpu.memory, "set_per_process_memory_fraction"
        ):
            torch.xpu.memory.set_per_process_memory_fraction(frac, device=0)
    except Exception as exc:
        print(f"[WARN] could not set XPU memory fraction: {exc}", flush=True)


def log_device() -> str:
    if xpu_available():
        configure_xpu_runtime()
    device = pick_device()
    print(f"[INFO] device={device} ({device_name(device)})")
    print(f"[INFO] CUDA available: {cuda_available()}")
    print(f"[INFO] XPU available: {xpu_available()}")
    if device == "xpu":
        mem = xpu_mem_gb(0)
        if mem is not None:
            print(f"[INFO] XPU memory: {mem:.1f} GiB")
        print(f"[INFO] recommended batch≈{recommend_batch_size()}")
        print(f"[INFO] xpu_count={torch.xpu.device_count()}")
        for i in range(torch.xpu.device_count()):
            try:
                print(f"[INFO] xpu:{i} {torch.xpu.get_device_name(i)}")
            except Exception:
                pass
    return device
