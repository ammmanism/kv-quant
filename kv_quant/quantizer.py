"""Pure functional quantizer: stateless, no in-place mutation."""

from dataclasses import dataclass

import torch

from kv_quant.layout import PageLayout


@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters for a single group."""
    scale: torch.Tensor      # fp16, shape ()
    zero_point: torch.Tensor # uint8, shape ()


def quantize_group_fp16_to_int4(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, QuantParams]:
    """Quantize a single group from FP16 to INT4 (pure functional).

    Args:
        x: Input tensor of shape (group_size,), dtype fp16.
        generator: Optional seeded generator for stochastic rounding.

    Returns:
        (packed_int4, params) where packed_int4 is uint8 tensor of shape (group_size // 2,)
        and params contains scale and zero_point.
    """
    raise NotImplementedError("quantize_group_fp16_to_int4 not implemented")


def dequantize_group_int4_to_fp16(
    packed: torch.Tensor,
    params: QuantParams,
) -> torch.Tensor:
    """Dequantize a single group from INT4 to FP16 (pure functional).

    Args:
        packed: Packed INT4 tensor of shape (group_size // 2,), dtype uint8.
        params: QuantParams with scale (fp16) and zero_point (uint8).

    Returns:
        Dequantized tensor of shape (group_size,), dtype fp16.
    """
    raise NotImplementedError("dequantize_group_int4_to_fp16 not implemented")


def quantize_page(
    page_fp16: torch.Tensor,
    layout: PageLayout | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[QuantParams]]:
    """Quantize a full page (sink + quantizable tokens).

    Args:
        page_fp16: Input page tensor of shape (page_size,), dtype fp16.
        layout: Page layout (uses canonical if None).
        generator: Optional seeded generator.

    Returns:
        (packed_page, params_list) where packed_page is uint8 tensor of layout.total_bytes,
        and params_list has one QuantParams per group.
    """
    raise NotImplementedError("quantize_page not implemented")


def dequantize_page(
    packed_page: torch.Tensor,
    params_list: list[QuantParams],
    layout: PageLayout | None = None,
) -> torch.Tensor:
    """Dequantize a full page back to FP16.

    Args:
        packed_page: Packed page tensor of shape (total_bytes,), dtype uint8.
        params_list: List of QuantParams per group.
        layout: Page layout (uses canonical if None).

    Returns:
        Dequantized page tensor of shape (page_size,), dtype fp16.
    """
    raise NotImplementedError("dequantize_page not implemented")
