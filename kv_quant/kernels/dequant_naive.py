"""Naive reference dequantization kernel (Python/Torch)."""

import torch

from kv_quant.layout import PageLayout, get_layout
from kv_quant.quantizer import QuantParams


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
        group_packed = packed_page[quant_offset:quant_offset + group_quant_bytes]
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
        out[out_idx:out_idx + group_size] = dequant.to(torch.float16)

    # 3. Skip metadata (scale + zero_point per group) - already consumed via params_list

    return out
