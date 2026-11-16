"""VRAM estimation and tracking utilities."""

from dataclasses import dataclass

import torch

from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.layout import PageLayout, get_layout


@dataclass(frozen=True)
class VRAMBreakdown:
    """Detailed VRAM breakdown."""

    sink_fp16_bytes: int
    quant_pages_bytes: int
    ring_fp16_bytes: int
    metadata_bytes: int
    activation_bytes: int
    overhead_bytes: int
    total_bytes: int

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def compression_ratio(self) -> float:
        """Ratio vs full FP16 KV cache."""
        fp16_total = self.sink_fp16_bytes + self.quant_pages_bytes * 2 + self.ring_fp16_bytes
        return fp16_total / self.total_bytes if self.total_bytes > 0 else 1.0


def estimate_vram(
    quant_config: QuantConfig,
    cache_config: CacheConfig,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    layout: PageLayout | None = None,
) -> VRAMBreakdown:
    """Estimate VRAM usage for a model with quantized KV cache.

    Args:
        quant_config: Quantization configuration.
        cache_config: Cache configuration.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Head dimension.
        max_seq_len: Maximum sequence length.
        layout: Page layout (uses canonical if None).

    Returns:
        VRAMBreakdown with per-component byte counts.
    """
    layout = layout or get_layout()

    # Per-layer calculations
    tokens_per_layer = max_seq_len
    sink_tokens = quant_config.sink_tokens
    quantizable = tokens_per_layer - sink_tokens
    num_pages = (quantizable + layout.page_size - 1) // layout.page_size

    # Sink (FP16)
    sink_bytes = sink_tokens * num_heads * head_dim * 2 * num_layers

    # Quantized pages
    page_bytes = layout.total_bytes * num_heads * num_layers * num_pages

    # Ring buffer (worst case: group_size - 1 tokens)
    ring_tokens = quant_config.group_size - 1
    ring_bytes = ring_tokens * num_heads * head_dim * 2 * num_layers

    # Metadata (included in page_bytes via layout)
    meta_bytes = 0  # Already counted in page_bytes

    # Activations (rough estimate for attention scores, etc.)
    activation_bytes = max_seq_len * num_heads * head_dim * 2 * num_layers * 2

    # Overhead (CUDA context, allocator, etc.)
    overhead_bytes = int((sink_bytes + page_bytes + ring_bytes) * 0.05)

    total = sink_bytes + page_bytes + ring_bytes + activation_bytes + overhead_bytes

    return VRAMBreakdown(
        sink_fp16_bytes=sink_bytes,
        quant_pages_bytes=page_bytes,
        ring_fp16_bytes=ring_bytes,
        metadata_bytes=meta_bytes,
        activation_bytes=activation_bytes,
        overhead_bytes=overhead_bytes,
        total_bytes=total,
    )


def measure_actual_vram(device: str = "cuda") -> dict[str, int]:
    """Measure actual VRAM usage on device.

    Returns:
        Dict with 'allocated', 'reserved', 'free' in bytes.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}

    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free = total - reserved

    return {"allocated": allocated, "reserved": reserved, "free": free}
