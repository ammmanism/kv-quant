"""Pure functional QJL (Quantized Johnson-Lindenstrauss) sketch."""

import torch

from kv_quant.layout import QJL_SEED, QJL_SKETCH_WORDS


def generate_qjl_projection(
    d_in: int,
    d_out: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate QJL projection matrix (pure functional).

    Args:
        d_in: Input dimension.
        d_out: Output dimension (sketch size).
        generator: Seeded generator for reproducibility.

    Returns:
        Projection matrix of shape (d_out, d_in), dtype fp16.
    """
    raise NotImplementedError("generate_qjl_projection not implemented")


def qjl_sketch(
    x: torch.Tensor,
    projection: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply QJL sketch to input (pure functional).

    Args:
        x: Input tensor of shape (..., d_in).
        projection: Optional precomputed projection matrix (d_out, d_in).
        generator: Seeded generator (used if projection is None).

    Returns:
        Sketch tensor of shape (..., d_out).
    """
    raise NotImplementedError("qjl_sketch not implemented")


def qjl_reconstruct(
    sketch: torch.Tensor,
    projection: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Reconstruct approximation from QJL sketch (pure functional).

    Args:
        sketch: Sketch tensor of shape (..., d_out).
        projection: Projection matrix of shape (d_out, d_in).
        generator: Optional seeded generator.

    Returns:
        Reconstructed tensor of shape (..., d_in).
    """
    raise NotImplementedError("qjl_reconstruct not implemented")


def get_default_projection(
    d_in: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Get default QJL projection for given input dimension.

    Uses fixed seed QJL_SEED for reproducibility.

    Args:
        d_in: Input dimension.
        generator: Optional generator (overrides fixed seed if provided).

    Returns:
        Projection matrix of shape (QJL_SKETCH_WORDS, d_in).
    """
    if QJL_SKETCH_WORDS == 0:
        raise ValueError("QJL sketch disabled (QJL_SKETCH_WORDS == 0)")
    gen = generator or torch.Generator().manual_seed(QJL_SEED)
    return generate_qjl_projection(d_in, QJL_SKETCH_WORDS, gen)
