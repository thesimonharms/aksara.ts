"""Pick CUDA / Intel XPU / CPU for local TrOCR train + inference.

Prefer Intel Arc (torch.xpu) when present so eGPU / NAS setups use the
dedicated card instead of falling back to CPU.

When ``transformers-xpu-helper`` is installed, XPU runtime env, memory
fraction, dataloader knobs, and vision batch heuristics come from that
library (Core Ultra 7 255H / Arc 140T defaults). Local overrides
(``TROCR_BATCH_SIZE``, ``TROCR_XPU_MEM_FRACTION``, Windows XPU fp32) still win.
"""

from __future__ import annotations

import os
import sys

import torch


def _xpu_helper():
    try:
        import transformers_xpu_helper as helper  # noqa: F401

        return helper
    except ImportError:
        return None


def xpu_available() -> bool:
    return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def dml_available() -> bool:
    try:
        import torch_directml  # type: ignore

        return bool(torch_directml.device_count() > 0)
    except Exception:
        return False


def resolve_torch_device(device: str | None = None):
    """Return a torch.device / DirectML device handle for model.to(...)."""
    d = device or pick_device()
    if d in ("dml", "privateuseone"):
        import torch_directml  # type: ignore

        return torch_directml.device()
    return torch.device(d)


def pick_device() -> str:
    force = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if force in ("cpu", "cuda", "xpu", "dml", "privateuseone"):
        return "dml" if force == "privateuseone" else force
    helper = _xpu_helper()
    if helper is not None:
        try:
            info = helper.detect_device(prefer="xpu", profile_hint="255H")
            if info.is_xpu:
                return "xpu"
        except Exception:
            pass
    if xpu_available():
        return "xpu"
    if cuda_available():
        return "cuda"
    if dml_available():
        return "dml"
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
    if d == "dml":
        try:
            import torch_directml  # type: ignore

            return torch_directml.device_name(0)
        except Exception:
            return "DirectML"
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


def recommend_batch_size(default: int = 16, model=None) -> int:
    """Heuristic micro-batch for TrOCR on Arc / XPU.

    ``TROCR_BATCH_SIZE`` always wins. Otherwise prefer
    ``transformers-xpu-helper`` vision heuristics when installed.
    """
    override = os.environ.get("TROCR_BATCH_SIZE", "").strip()
    if override.isdigit():
        return max(1, int(override))

    helper = _xpu_helper()
    if helper is not None:
        try:
            from transformers_xpu_helper import ultra_255h_config
            from transformers_xpu_helper.memory import suggest_vision_batch_size
            from transformers_xpu_helper.trainer import recommend_for_model

            gc_on = os.environ.get("GRADIENT_CHECKPOINTING", "1").strip() not in (
                "0",
                "false",
                "no",
            )
            cfg = ultra_255h_config(
                torch_compile=False,
                gradient_checkpointing=gc_on,
            )
            if model is not None:
                return int(
                    recommend_for_model(
                        model,
                        task="vision",
                        image_hw=(384, 384),
                        config=cfg,
                    ).per_device_train_batch_size
                )
            # trocr-large-printed ≈ 558M params when no model handle yet
            return int(
                suggest_vision_batch_size(
                    558_000_000,
                    image_hw=(384, 384),
                    config=cfg,
                )
            )
        except Exception as exc:
            print(f"[WARN] xpu-helper batch recommend failed: {exc}", flush=True)

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
    Linux XPU: helper / bf16 unless TROCR_FORCE_FP32=1.
    CUDA: fp16 (Jobs recipe).
    """
    d = device or pick_device()
    if d == "xpu":
        if os.environ.get("TROCR_FORCE_FP32", "").strip() in ("1", "true", "yes"):
            return False, False
        if sys.platform == "win32":
            return False, False
        helper = _xpu_helper()
        if helper is not None:
            try:
                from transformers_xpu_helper import ultra_255h_config
                from transformers_xpu_helper.amp import resolve_amp
                from transformers_xpu_helper.hardware import detect_device

                info = detect_device(prefer="xpu", profile_hint="255H")
                amp = resolve_amp(ultra_255h_config(), info)
                if amp.dtype is not None and amp.dtype == torch.bfloat16:
                    return False, True
                if amp.dtype is not None and amp.dtype == torch.float16:
                    return True, False
            except Exception:
                pass
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
    """DataLoader settings for TrainingArguments (dataloader_* keys)."""
    d = device or pick_device()
    if sys.platform == "win32":
        return {"dataloader_num_workers": 0, "dataloader_pin_memory": False}

    helper = _xpu_helper()
    if helper is not None and d == "xpu":
        try:
            from transformers_xpu_helper import ultra_255h_config
            from transformers_xpu_helper.dataloader import dataloader_kwargs as helper_dl
            from transformers_xpu_helper.hardware import detect_device

            raw = helper_dl(ultra_255h_config(), detect_device(prefer="xpu", profile_hint="255H"))
            out = {
                "dataloader_num_workers": int(raw.get("num_workers", 4)),
                "dataloader_pin_memory": bool(raw.get("pin_memory", False)),
            }
            if raw.get("persistent_workers"):
                out["dataloader_persistent_workers"] = True
            return out
        except Exception as exc:
            print(f"[WARN] xpu-helper dataloader_kwargs failed: {exc}", flush=True)

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

    helper = _xpu_helper()
    if helper is not None:
        try:
            from transformers_xpu_helper import (
                apply_memory_fraction,
                apply_runtime_env,
                ultra_255h_config,
            )
            from transformers_xpu_helper.hardware import detect_device

            cfg = ultra_255h_config()
            frac_env = os.environ.get("TROCR_XPU_MEM_FRACTION", "").strip()
            if frac_env:
                cfg = cfg.with_updates(memory_fraction=float(frac_env))
            info = detect_device(prefer="xpu", profile_hint="255H")
            apply_runtime_env(cfg, info.profile)
            apply_memory_fraction(cfg, info)
            print(
                f"[INFO] transformers-xpu-helper v{getattr(helper, '__version__', '?')} "
                f"profile={cfg.profile_name} mem_frac={cfg.memory_fraction}",
                flush=True,
            )
            return
        except Exception as exc:
            print(f"[WARN] xpu-helper runtime setup failed: {exc}", flush=True)

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
    helper = _xpu_helper()
    print(f"[INFO] transformers-xpu-helper: {'yes' if helper else 'no'}")
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
