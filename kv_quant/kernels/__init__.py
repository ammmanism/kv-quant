"""Triton kernels for quantized KV cache operations."""

from kv_quant.kernels.dequant_naive import dequant_naive

try:
    from kv_quant.kernels.dequant_fused import dequant_fused
except ImportError:
    dequant_fused = None  # type: ignore

__all__ = ["dequant_naive", "dequant_fused"]
