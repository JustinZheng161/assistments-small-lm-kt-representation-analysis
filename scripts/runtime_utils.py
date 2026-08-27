"""Shared runtime helpers for reproducible local model audits."""
from __future__ import annotations

import gc
from contextlib import contextmanager

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cuda/cpu/mps without changing the CPU default."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    """Use float32 by default; allow reduced precision only when explicitly requested."""
    if dtype == "auto":
        return torch.float32
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in values:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if device.type == "cpu" and dtype == "float16":
        raise ValueError("float16 is not enabled for CPU inference; use float32 or bfloat16")
    return values[dtype]


@contextmanager
def inference_runtime(device: torch.device):
    """Set deterministic inference context and release accelerator cache afterwards."""
    with torch.inference_mode():
        yield
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}
