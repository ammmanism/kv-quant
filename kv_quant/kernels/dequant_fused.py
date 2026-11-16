"""Fused Triton dequantization kernel."""

import torch
import triton
import triton.language as tl

from kv_quant.layout import PageLayout, get_layout
from kv_quant.quantizer import QuantParams


@triton.jit
def _dequant_fused_kernel(
    packed_ptr,  # uint8* packed page data
    scales_ptr,  # fp16* scales (one per group)
    zps_ptr,  # uint8* zero_points (one per group)
    out_ptr,  # fp16* output
    page_size: tl.constexpr,
    sink_tokens: tl.constexpr,
    group_size: tl.constexpr,
    groups_per_page: tl.constexpr,
):
    """Triton kernel for fused dequantization."""
    pid = tl.program_id(0)

    # Each thread block handles one group
    if pid >= groups_per_page:
        return

    # Sink tokens are copied separately (host or separate kernel)
    # This kernel only handles quantized groups

    # Group offset in packed data (after sink)
    sink_bytes = sink_tokens * 2
    quant_bytes_per_group = group_size // 2
    group_offset = sink_bytes + pid * quant_bytes_per_group

    # Load scale and zero_point
    scale = tl.load(scales_ptr + pid).to(tl.float32)
    zp = tl.load(zps_ptr + pid).to(tl.float32)

    # Load packed INT4 for this group
    packed = tl.load(packed_ptr + group_offset)

    # Unpack and dequantize
    # We'll process 8 bytes (16 values) per thread
    for i in range(0, quant_bytes_per_group, 8):
        # Load 8 bytes = 16 INT4 values
        chunk = tl.load(packed_ptr + group_offset + i)

        # Unpack 16 values
        vals = tl.zeros((16,), dtype=tl.float32)
        for j in range(8):
            byte = (chunk >> (j * 8)) & 0xFF
            low = byte & 0x0F
            high = (byte >> 4) & 0x0F
            vals[j * 2] = low.to(tl.float32)
            vals[j * 2 + 1] = high.to(tl.float32)

        # Dequantize
        vals = (vals - zp) * scale

        # Store to output
        out_idx = sink_tokens + pid * group_size + i * 2
        for j in range(16):
            tl.store(out_ptr + out_idx + j, vals[j].to(tl.float16))


def dequant_fused(
    packed_page: torch.Tensor,
    params_list: list[QuantParams],
    layout: PageLayout | None = None,
) -> torch.Tensor:
    """Fused Triton dequantization.

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
    groups_per_page = (page_size - sink_tokens) // group_size

    device = packed_page.device

    # Output buffer
    out = torch.empty(page_size, dtype=torch.float16, device=device)

    # 1. Copy sink tokens (FP16)
    sink_bytes = sink_tokens * 2
    if sink_bytes > 0:
        out[:sink_tokens] = packed_page[:sink_bytes].view(torch.float16)

    # 2. Prepare params arrays for kernel
    scales = torch.stack([p.scale for p in params_list]).to(device)
    zps = torch.stack([p.zero_point for p in params_list]).to(device)

    # 3. Launch kernel
    grid = (groups_per_page,)
    _dequant_fused_kernel[grid](
        packed_page,
        scales,
        zps,
        out,
        page_size,
        sink_tokens,
        group_size,
        groups_per_page,
    )

    return out
