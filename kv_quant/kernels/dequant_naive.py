"""Naive reference dequantization kernel (Python/Torch)."""

from dataclasses import dataclass
from typing import Any

import torch

from kv_quant.layout import PageLayout, get_layout
from kv_quant.quantizer import QuantParams, dequantize_group


@dataclass(frozen=True)
class NaiveDequantStats:
    """Exact HBM read/write telemetry for the naive materialization path."""

    hbm_read_bytes: int
    hbm_write_bytes: int
    pages_dequantized: int
    sink_tokens: int
    ring_tokens: int


def dequant_naive(
    packed_page: torch.Tensor,
    params_list: list[QuantParams],
    layout: PageLayout | None = None,
) -> torch.Tensor:
    """Naive reference dequantization (pure Python/PyTorch).

    Used as ground truth for parity testing.

    Args:
        packed_page: Packed page tensor of shape (total_bytes,), dtype uint8.
        params_list: List of QuantParams per group.
        layout: Page layout (uses canonical if None).

    Returns:
        Dequantized page tensor of shape (page_size,), dtype fp16.
    """
    layout = layout or get_layout()
    page_size = layout.page_size
    sink_tokens = layout.sink_tokens
    group_size = layout.group_size

    # Output buffer
    out = torch.empty(page_size, dtype=torch.float16, device=packed_page.device)

    # 1. Copy sink tokens (first sink_tokens * 2 bytes = FP16)
    sink_bytes = sink_tokens * 2
    if sink_bytes > 0:
        sink_data = packed_page[:sink_bytes].view(torch.float16)
        out[:sink_tokens] = sink_data

    # 2. Dequantize each group
    quant_offset = sink_bytes
    groups_per_page = (page_size - sink_tokens) // group_size

    for g in range(groups_per_page):
        params = params_list[g]
        scale = params.scale.to(torch.float32)
        zero_point = params.zero_point.to(torch.float32)

        # Extract packed INT4 for this group
        group_quant_bytes = group_size // 2  # 2 values per byte
        group_packed = packed_page[quant_offset : quant_offset + group_quant_bytes]
        quant_offset += group_quant_bytes

        # Unpack nibbles
        low = (group_packed & 0x0F).to(torch.float32)
        high = (group_packed >> 4).to(torch.float32)

        # Interleave: low[0], high[0], low[1], high[1], ...
        dequant = torch.empty(group_size, dtype=torch.float32, device=packed_page.device)
        dequant[0::2] = low
        dequant[1::2] = high

        # Dequantize: (x - zero_point) * scale
        dequant = (dequant - zero_point) * scale

        # Write to output
        out_idx = sink_tokens + g * group_size
        out[out_idx : out_idx + group_size] = dequant.to(torch.float16)

    # 3. Skip metadata (scale + zero_point per group) - already consumed via params_list

    return out


def materialize_kv_naive(
    sink_k: torch.Tensor | None,
    sink_v: torch.Tensor | None,
    pages: dict[int, Any],  # PhysicalPage from kv_quant.cache (duck-typed)
    page_table: list[int],
    ring_k: torch.Tensor | None,
    ring_v: torch.Tensor | None,
    k_bits: int,
    v_bits: int,
    k_scheme: str,
    v_scheme: str,
    num_heads: int,
    head_dim: int,
    group_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, NaiveDequantStats]:
    """Materialize FP16 K/V from Tiers 1+2+3 before SDPA, logging exact HBM bytes.

    Args:
        sink_k/sink_v: Tier 1 FP16 sink tensors (sink_tokens, heads, dim).
        pages: Mapping page_id -> PhysicalPage (with k_payload/v_payload/params).
        page_table: Logical sequence of physical page ids.
        ring_k/ring_v: Tier 3 FP16 tail buffers.
        k_bits/v_bits, k_scheme/v_scheme: Quantization specs for dequantization.
        num_heads, head_dim: Shape metadata.
        group_size: Tokens per sealed page.

    Returns:
        (k_fp16, v_fp16, stats)
    """
    pieces_k: list[torch.Tensor] = []
    pieces_v: list[torch.Tensor] = []
    read_bytes = 0
    write_bytes = 0

    if sink_k is not None:
        pieces_k.append(sink_k)
        pieces_v.append(sink_v)  # type: ignore[arg-type]
        read_bytes += sink_k.numel() * 2 + sink_v.numel() * 2  # type: ignore[union-attr]

    for pid in page_table:
        page = pages[pid]
        k_ch, v_ch = [], []
        heads_dim = num_heads * head_dim
        for i in range(heads_dim):
            k_ch.append(dequantize_group(page.k_payload[i], page.k_params[i], k_bits, k_scheme))  # type: ignore[arg-type]
            v_ch.append(dequantize_group(page.v_payload[i], page.v_params[i], v_bits, v_scheme))  # type: ignore[arg-type]
        k_block = torch.stack(k_ch, 0).t().reshape(group_size, num_heads, head_dim)
        v_block = torch.stack(v_ch, 0).t().reshape(group_size, num_heads, head_dim)
        pieces_k.append(k_block.to(torch.float16))
        pieces_v.append(v_block.to(torch.float16))
        read_bytes += page.k_payload.numel() + page.v_payload.numel()
        write_bytes += k_block.numel() * 2 + v_block.numel() * 2

    ring_tokens = 0
    if ring_k is not None:
        pieces_k.append(ring_k)
        pieces_v.append(ring_v)  # type: ignore[arg-type]
        ring_tokens = ring_k.shape[0]
        read_bytes += ring_k.numel() * 2 + ring_v.numel() * 2  # type: ignore[union-attr]

    if not pieces_k:
        empty = torch.zeros(0, num_heads, head_dim, dtype=torch.float16)
        return empty, empty, NaiveDequantStats(0, 0, 0, 0, 0)

    k_fp16 = torch.cat(pieces_k, 0)
    v_fp16 = torch.cat(pieces_v, 0)
    stats = NaiveDequantStats(
        hbm_read_bytes=read_bytes,
        hbm_write_bytes=write_bytes,
        pages_dequantized=len(page_table),
        sink_tokens=sink_k.shape[0] if sink_k is not None else 0,
        ring_tokens=ring_tokens,
    )
    return k_fp16, v_fp16, stats
