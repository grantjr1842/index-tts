"""VRAM optimization utilities for IndexTTS2.

This module provides utilities for reducing VRAM usage during inference:
- INT8 quantization for static models
- Memory profiling helpers
"""

import torch
from typing import Optional
import gc


def quantize_model_int8(model: torch.nn.Module, dtype=torch.qint8) -> torch.nn.Module:
    """Apply dynamic INT8 quantization to a model.
    
    This reduces model memory by ~50% with minimal quality impact for
    inference-only models like the semantic encoder.
    
    Args:
        model: PyTorch model to quantize
        dtype: Quantization dtype (default: torch.qint8)
        
    Returns:
        Quantized model (on CPU, must be moved back to device after)
    """
    from torch.ao.quantization import quantize_dynamic
    
    # Must move entire model to CPU first - quantization only works on CPU
    # Use .to() to recursively move all submodules, parameters, and buffers
    model = model.cpu()
    
    # Ensure all parameters are on CPU (sanity check)
    for param in model.parameters():
        if param.device.type != 'cpu':
            param.data = param.data.cpu()
    
    # Ensure all buffers are on CPU
    for buffer in model.buffers():
        if buffer.device.type != 'cpu':
            buffer.data = buffer.data.cpu()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    quantized = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=dtype
    )
    
    return quantized


def get_vram_usage() -> dict:
    """Get current VRAM usage statistics.
    
    Returns:
        Dict with allocated, reserved, and max_allocated in GB
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "free_gb": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9,
    }


def print_vram_usage(prefix: str = "") -> None:
    """Print current VRAM usage to console."""
    usage = get_vram_usage()
    if not usage.get("available"):
        print(f"{prefix}CUDA not available")
        return
    
    print(f"{prefix}VRAM: {usage['allocated_gb']:.2f}GB allocated, "
          f"{usage['free_gb']:.2f}GB free, "
          f"{usage['max_allocated_gb']:.2f}GB peak")


def force_cleanup() -> None:
    """Force CUDA memory cleanup and garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class VRAMProfiler:
    """Context manager for profiling VRAM usage of a code block.
    
    Usage:
        with VRAMProfiler("Loading model"):
            model = load_model()
    """
    
    def __init__(self, name: str = "Block", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_allocated = 0
        self.start_reserved = 0
    
    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_allocated = torch.cuda.memory_allocated()
            self.start_reserved = torch.cuda.memory_reserved()
        return self
    
    def __exit__(self, *args):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            end_allocated = torch.cuda.memory_allocated()
            end_reserved = torch.cuda.memory_reserved()
            
            delta_alloc = (end_allocated - self.start_allocated) / 1e9
            delta_res = (end_reserved - self.start_reserved) / 1e9
            
            print(f"[VRAM] {self.name}: "
                  f"allocated {delta_alloc:+.3f}GB, "
                  f"reserved {delta_res:+.3f}GB, "
                  f"total {end_allocated/1e9:.2f}GB")


class VRAMTracker:
    """Track VRAM usage across model loading stages.
    
    Usage:
        tracker = VRAMTracker()
        # ... load model ...
        tracker.snapshot("after_gpt")
        # ... load more ...
        tracker.snapshot("after_semantic")
        tracker.print_summary()
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.snapshots: list[tuple[str, float, float]] = []
        if self.enabled:
            self.snapshots.append(("init", torch.cuda.memory_allocated() / 1e9, 0.0))
    
    def snapshot(self, name: str, model_name: Optional[str] = None) -> None:
        """Take a VRAM snapshot with optional model size tracking."""
        if not self.enabled:
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        delta = allocated - (self.snapshots[-1][1] if self.snapshots else 0)
        self.snapshots.append((name, allocated, delta))
        
        if model_name:
            print(f"[VRAM] {name}: {allocated:.2f}GB total (+{delta:.2f}GB for {model_name})")
        else:
            print(f"[VRAM] {name}: {allocated:.2f}GB total (+{delta:.2f}GB)")
    
    def get_model_vram(self, model: torch.nn.Module, name: str) -> float:
        """Estimate VRAM used by a model's parameters."""
        if not self.enabled:
            return 0.0
        total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        total_gb = total_bytes / 1e9
        print(f"[VRAM] {name} params: {total_gb:.2f}GB")
        return total_gb
    
    def print_summary(self) -> None:
        """Print summary of all snapshots."""
        if not self.enabled or not self.snapshots:
            return
        print("\n[VRAM Summary]")
        for name, allocated, delta in self.snapshots:
            print(f"  {name}: {allocated:.2f}GB (+{delta:.2f}GB)")
        print(f"  Final: {self.snapshots[-1][1]:.2f}GB")
