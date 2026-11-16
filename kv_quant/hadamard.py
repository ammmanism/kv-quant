"""Pure functional Hadamard transforms (stateless, no in-place mutation)."""

import torch

from kv_quant.layout import PageLayout


def hadamard_transform(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply Walsh-Hadamard transform (pure functional).

    Args:
        x: Input tensor of shape (..., n) where n is a power of 2.
        generator: Optional seeded generator (unused, for API consistency).

    Returns:
        Hadamard-transformed tensor of same shape and dtype.
    """
    raise NotImplementedError("hadamard_transform not implemented")


def inverse_hadamard_transform(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply inverse Walsh-Hadamard transform (pure functional).

    For normalized Hadamard, inverse is the same as forward up to scaling.

    Args:
        x: Input tensor of shape (..., n) where n is a power of 2.
        generator: Optional seeded generator (unused, for API consistency).

    Returns:
        Inverse-transformed tensor of same shape and dtype.
    """
    raise NotImplementedError("inverse_hadamard_transform not implemented")


def hadamard_quantize(
    x: torch.Tensor,
    layout: PageLayout | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hadamard transform + quantize (pure functional pipeline).

    Args:
        x: Input tensor of shape (group_size,), dtype fp16.
        layout: Page layout (uses canonical if None).
        generator: Optional seeded generator.

    Returns:
        Quantized output (packed INT4).
    """
    raise NotImplementedError("hadamard_quantize not implemented")


def hadamard_dequantize(
    packed: torch.Tensor,
    layout: PageLayout | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Dequantize + inverse Hadamard transform (pure functional pipeline).

    Args:
        packed: Packed INT4 tensor.
        layout: Page layout (uses canonical if None).
        generator: Optional seeded generator.

    Returns:
        Dequantized FP16 tensor.
    """
    raise NotImplementedError("hadamard_dequantize not implemented")
