"""VRAM estimation and tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.layout import PageLayout, get_layout

if TYPE_CHECKING:
    import pandas as pd

    from kv_quant.cache import HybridPagedKVCache


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


def kv_cache_calculator_bytes(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    k_bits: int,
    v_bits: int,
    sink_tokens: int = 4,
    group_size: int = 64,
) -> int:
    """Analytical KV-cache prediction (kv-cache-calculator style).

    Hybrid-paged model: only *sealed* (full-group) tokens live in quantized
    pages; sink tokens and the sub-group ring tail stay in FP16. The model
    matches the implemented packer: INT6 is currently stored as 1 byte per
    value (see kv_quant/quantizer.py; deviation from the layout.py 0.75
    B/elem contract is tracked in rfc/reviews/hardware-truth-review.md).

    Returns:
        Total bytes across all layers.
    """
    channels = num_kv_heads * head_dim
    quant_tokens = max(0, seq_len - sink_tokens)
    sealed = (quant_tokens // group_size) * group_size
    ring_tokens = quant_tokens - sealed

    def bits_to_bytes(bits: int) -> float:
        return 1.0 if bits == 6 else bits / 8

    payload_bytes = int(sealed * channels * (bits_to_bytes(k_bits) + bits_to_bytes(v_bits)))
    sink_fp16_bytes = min(sink_tokens, seq_len) * channels * 2 * 2
    ring_fp16_bytes = ring_tokens * channels * 2 * 2

    per_layer = payload_bytes + sink_fp16_bytes + ring_fp16_bytes
    return per_layer * num_layers


def get_tier_bytes(cache: HybridPagedKVCache) -> dict[str, int]:
    """Read the 3-tier byte breakdown from a live cache (public API only).

    Note: lives in ``vram.py`` (not ``cache.py``) to respect the
    one-core-file-per-task invariant; uses only public cache attributes.
    """
    sink = 0
    if cache.sink_k is not None:
        sink = cache.sink_k.numel() * 2 + cache.sink_v.numel() * 2  # type: ignore[union-attr]
    pages = sum(
        p.k_payload.numel() + p.v_payload.numel()
        for p in cache._pages.values()  # noqa: SLF001
    )
    ring = 0
    if cache.ring_k is not None:
        ring = cache.ring_k.numel() * 2 + cache.ring_v.numel() * 2  # type: ignore[union-attr]
    return {
        "sink_fp16_bytes": sink,
        "quant_pages_bytes": pages,
        "ring_fp16_bytes": ring,
        "total_bytes": sink + pages + ring,
    }


def cross_validate_vram(
    configs: list[tuple[int, int, int]],
    num_layers: int = 32,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    sink_tokens: int = 4,
    group_size: int = 64,
    chunk_tokens: int = 1024,
    seed: int = 42,
) -> pd.DataFrame:
    """Cross-validate measured cache bytes against the analytical calculator.

    Each config is ``(k_bits, v_bits, seq_len)``. A real
    ``HybridPagedKVCache`` is built and populated token-by-token through the
    public ``append`` path; measured bytes come from the live object (shared
    pages counted once). On a CUDA host the peak ``max_memory_allocated``
    delta is used instead; on CPU-only hosts the object byte accounting is
    the measured source of truth.
    """
    import pandas as pd

    from kv_quant.cache import HybridPagedKVCache
    from kv_quant.config import CacheConfig

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows: list[dict[str, float | int | str]] = []

    for k_bits, v_bits, seq_len in configs:
        qc = QuantConfig(
            k_bits=k_bits,  # type: ignore[arg-type]
            v_bits=v_bits,  # type: ignore[arg-type]
            sink_tokens=sink_tokens,
        )
        cc = CacheConfig(device="cpu" if not use_cuda else "cuda")
        if use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        cache = HybridPagedKVCache(qc, cc)

        remaining = seq_len
        while remaining > 0:
            take = min(chunk_tokens, remaining)
            k = torch.randn(take, num_kv_heads, head_dim, generator=gen).to(device)
            v = torch.randn(take, num_kv_heads, head_dim, generator=gen).to(device)
            cache.append(k, v)
            remaining -= take

        tiers = get_tier_bytes(cache)
        if use_cuda:
            measured_peak = int(torch.cuda.max_memory_allocated())
            method = "cuda_max_memory_allocated"
        else:
            measured_peak = tiers["total_bytes"]
            method = "cache_object_bytes"

        predicted = kv_cache_calculator_bytes(
            num_layers,
            num_kv_heads,
            head_dim,
            seq_len,
            k_bits,
            v_bits,
            sink_tokens,
            group_size,
        )
        measured_full = measured_peak * num_layers
        fp16_equiv = num_layers * seq_len * num_kv_heads * head_dim * 2 * 2

        residual = (measured_full - predicted) / predicted if predicted else 0.0
        rows.append(
            {
                "k_bits": k_bits,
                "v_bits": v_bits,
                "seq_len": seq_len,
                "sink_tokens": sink_tokens,
                "measured_peak_bytes": measured_full,
                "calculator_predicted_bytes": predicted,
                "kv_calc_fp16_equiv_bytes": fp16_equiv,
                "compression_ratio": round(fp16_equiv / measured_full, 4) if measured_full else 0.0,
                "residual_pct": round(residual * 100, 4),
                "measure_method": method,
                "sink_bytes": tiers["sink_fp16_bytes"] * num_layers,
                "quant_pages_bytes": tiers["quant_pages_bytes"] * num_layers,
                "ring_bytes": tiers["ring_fp16_bytes"] * num_layers,
            }
        )

    df = pd.DataFrame(rows)
    # Global linear regression fit: measured = a * predicted + b
    import numpy as np

    x = df["calculator_predicted_bytes"].to_numpy(dtype=float)
    y = df["measured_peak_bytes"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fit = slope * x + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    mape = float(np.mean(np.abs((y - x) / x)) * 100)
    df["r_squared"] = round(r_squared, 6)
    df["mape_pct"] = round(mape, 4)
    return df


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
