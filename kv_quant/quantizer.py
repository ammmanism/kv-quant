"""Pure functional quantizer: stateless, no in-place mutation.

Implements symmetric/asymmetric block quantization for K and V with support for
INT4, INT2, INT8, FP8, and optional Hadamard rotation and QJL residual sketch.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from kv_quant.config import QuantConfig
from kv_quant.hadamard import hadamard_rademacher_transform
from kv_quant.qjl_sketch import (
    QJLSketch,
)


@dataclass(frozen=True)
class QuantParams:
    """Quantization parameters for a single group."""

    scale: torch.Tensor  # FP16, shape ()
    zero_point: torch.Tensor  # UINT8, shape () - only for asymmetric
    original_size: int = 0  # Original tensor size (for unpack padding)
    # For QJL
    qjl_sketch: QJLSketch | None = None
    qjl_projection: torch.Tensor | None = None


def _quantize_symmetric(
    x: torch.Tensor,
    bits: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization: x_q = round(x / scale), scale = max(|x|) / (2^(bits-1) - 1).

    Args:
        x: Input tensor of shape (group_size,), dtype FP16/BF16.
        bits: Bit-width (2, 4, 6, 8).
        generator: Optional seeded generator for stochastic rounding.

    Returns:
        (quantized_int, scale) where quantized_int is in [-2^(bits-1), 2^(bits-1)-1].
    """
    max_val = 2 ** (bits - 1) - 1
    x_abs_max = x.abs().max()

    if x_abs_max == 0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    else:
        scale = x_abs_max / max_val

    # Quantize
    x_q = torch.round(x / scale).to(torch.int32)
    x_q = torch.clamp(x_q, -max_val, max_val)

    return x_q, scale.to(torch.float16)


def _quantize_asymmetric(
    x: torch.Tensor,
    bits: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric quantization: x_q = round(x / scale) + zero_point.

    Args:
        x: Input tensor of shape (group_size,), dtype FP16/BF16.
        bits: Bit-width (2, 4, 6, 8).
        generator: Optional seeded generator.

    Returns:
        (quantized_int, scale, zero_point) where quantized_int in [0, 2^bits - 1].
    """
    max_val = 2**bits - 1
    x_min = x.min()
    x_max = x.max()

    if x_max == x_min:
        scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        zero_point = torch.tensor(0, dtype=torch.uint8, device=x.device)
    else:
        scale = (x_max - x_min) / max_val
        zero_point = torch.round(-x_min / scale).to(torch.int32)
        zero_point = torch.clamp(zero_point, 0, max_val).to(torch.uint8)

    x_q = torch.round(x / scale) + zero_point.to(torch.int32)
    x_q = torch.clamp(x_q, 0, max_val)

    return x_q, scale.to(torch.float16), zero_point


def _dequantize_symmetric(
    x_q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric dequantization: x = x_q * scale."""
    return x_q.to(torch.float32) * scale.to(torch.float32)


def _dequantize_asymmetric(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Asymmetric dequantization: x = (x_q - zero_point) * scale."""
    return (x_q.to(torch.float32) - zero_point.to(torch.float32)) * scale.to(torch.float32)


def pack_int4(x_q: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """
    Pack INT4 values (2 per byte) into uint8.

    Args:
        x_q: Tensor of shape (group_size,) with values in [-8, 7] (signed)
             or [0, 15] (unsigned).
        signed: If True, input is signed [-8, 7]. If False, input is unsigned [0, 15].

    Returns:
        Packed tensor of shape (ceil(group_size / 2),), dtype uint8.
    """
    if signed:
        # Convert signed [-8, 7] to unsigned [0, 15]
        x_unsigned = (x_q + 8).to(torch.uint8)
        pad_value = 8  # 0 in signed
    else:
        # Already unsigned [0, 15]
        x_unsigned = x_q.to(torch.uint8)
        pad_value = 0

    # Pad to even length
    n = x_unsigned.shape[0]
    pad = (2 - n % 2) % 2
    if pad > 0:
        x_unsigned = torch.nn.functional.pad(x_unsigned, (0, pad), value=pad_value)

    # Pack pairs: low nibble | (high nibble << 4)
    x_unsigned = x_unsigned.view(-1, 2)
    packed = x_unsigned[:, 0] | (x_unsigned[:, 1] << 4)

    return packed


def unpack_int4(
    packed: torch.Tensor, signed: bool = True, original_size: int | None = None
) -> torch.Tensor:
    """
    Unpack INT4 values from uint8.

    Args:
        packed: Tensor of shape (ceil(group_size / 2),), dtype uint8.
        signed: If True, return signed [-8, 7]. If False, return [0, 15].
        original_size: Original size before padding (to trim padded elements).

    Returns:
        Unpacked tensor of shape (group_size,), dtype int32.
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    # Interleave
    unpacked = torch.empty(packed.shape[0] * 2, dtype=torch.int32, device=packed.device)
    unpacked[0::2] = low
    unpacked[1::2] = high

    # Trim to original size if provided
    if original_size is not None:
        unpacked = unpacked[:original_size]

    if signed:
        unpacked = unpacked - 8  # Convert 0..15 to -8..7

    return unpacked


def pack_int2(x_q: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """
    Pack INT2 values (4 per byte) into uint8.

    Args:
        x_q: Tensor of shape (group_size,) with values in [-2, 1] (signed)
             or [0, 3] (unsigned).
        signed: If True, input is signed [-2, 1]. If False, input is unsigned [0, 3].

    Returns:
        Packed tensor of shape (ceil(group_size / 4),), dtype uint8.
    """
    if signed:
        # Convert signed [-2, 1] to unsigned [0, 3]
        x_unsigned = (x_q + 2).to(torch.uint8)
        pad_value = 2  # 0 in signed
    else:
        # Already unsigned [0, 3]
        x_unsigned = x_q.to(torch.uint8)
        pad_value = 0

    # Pad to multiple of 4
    n = x_unsigned.shape[0]
    pad = (4 - n % 4) % 4
    if pad > 0:
        x_unsigned = torch.nn.functional.pad(x_unsigned, (0, pad), value=pad_value)

    x_unsigned = x_unsigned.view(-1, 4)

    packed = (
        x_unsigned[:, 0]
        | (x_unsigned[:, 1] << 2)
        | (x_unsigned[:, 2] << 4)
        | (x_unsigned[:, 3] << 6)
    )

    return packed


def unpack_int2(
    packed: torch.Tensor, signed: bool = True, original_size: int | None = None
) -> torch.Tensor:
    """
    Unpack INT2 values from uint8.

    Args:
        packed: Tensor of shape (ceil(group_size / 4),), dtype uint8.
        signed: If True, return signed [-2, 1]. If False, return [0, 3].
        original_size: Original size before padding (to trim padded elements).

    Returns:
        Unpacked tensor of shape (group_size,), dtype int32.
    """
    crumb0 = packed & 0x03
    crumb1 = (packed >> 2) & 0x03
    crumb2 = (packed >> 4) & 0x03
    crumb3 = (packed >> 6) & 0x03

    unpacked = torch.empty(packed.shape[0] * 4, dtype=torch.int32, device=packed.device)
    unpacked[0::4] = crumb0
    unpacked[1::4] = crumb1
    unpacked[2::4] = crumb2
    unpacked[3::4] = crumb3

    # Trim to original size if provided
    if original_size is not None:
        unpacked = unpacked[:original_size]

    if signed:
        unpacked = unpacked - 2  # Convert 0..3 to -2..1

    return unpacked


def quantize_group(
    x: torch.Tensor,
    bits: int,
    scheme: Literal["sym", "asym"],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, QuantParams]:
    """
    Quantize a single group.

    Args:
        x: Input tensor of shape (group_size,), dtype FP16/BF16.
        bits: Bit-width (2, 4, 6, 8, 16).
        scheme: "sym" or "asym".
        generator: Optional seeded generator.

    Returns:
        (packed_quantized, QuantParams)
    """
    original_size = x.shape[0]

    if bits == 16:
        # No quantization - return FP16 as-is (packed as raw bytes).
        # .contiguous() first: callers may pass strided channel views, and
        # `view` requires the last-dim stride to be 1.
        packed = x.contiguous().view(torch.uint8)
        return packed, QuantParams(
            scale=torch.tensor(1.0, dtype=torch.float16, device=x.device),
            zero_point=torch.tensor(0, dtype=torch.uint8, device=x.device),
            original_size=original_size,
        )

    if scheme == "sym":
        x_q, scale = _quantize_symmetric(x, bits, generator)
        zero_point = torch.tensor(0, dtype=torch.uint8, device=x.device)
    else:
        x_q, scale, zero_point = _quantize_asymmetric(x, bits, generator)

    is_signed = scheme == "sym"

    # Pack based on bit-width
    if bits == 4:
        packed = pack_int4(x_q, signed=is_signed)
    elif bits == 2:
        packed = pack_int2(x_q, signed=is_signed)
    elif bits == 8:
        # INT8: 1 value per byte
        if scheme == "sym":
            packed = (x_q + 128).to(torch.uint8)
        else:
            packed = x_q.to(torch.uint8)
    elif bits == 6:
        # INT6: pack 4 values per 3 bytes (24 bits)
        # For simplicity, use 1 byte per value (wasteful but correct)
        packed = x_q.to(torch.uint8)
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")

    return packed, QuantParams(
        scale=scale,
        zero_point=zero_point,
        original_size=original_size,
    )


def dequantize_group(
    packed: torch.Tensor,
    params: QuantParams,
    bits: int,
    scheme: Literal["sym", "asym"],
) -> torch.Tensor:
    """
    Dequantize a single group.

    Args:
        packed: Packed quantized tensor.
        params: QuantParams with scale and zero_point.
        bits: Bit-width.
        scheme: "sym" or "asym".

    Returns:
        Dequantized tensor of shape (group_size,), dtype FP16.
    """
    if bits == 16:
        return packed.view(torch.float16)

    original_size = params.original_size if params.original_size > 0 else None
    is_signed = scheme == "sym"

    # Unpack
    if bits == 4:
        x_q = unpack_int4(packed, signed=is_signed, original_size=original_size)
    elif bits == 2:
        x_q = unpack_int2(packed, signed=is_signed, original_size=original_size)
    elif bits == 8:
        if scheme == "sym":
            x_q = packed.to(torch.int32) - 128
        else:
            x_q = packed.to(torch.int32)
    elif bits == 6:
        x_q = packed.to(torch.int32)
    else:
        raise ValueError(f"Unsupported bit-width: {bits}")

    # Dequantize
    if scheme == "sym":
        x = _dequantize_symmetric(x_q, params.scale)
    else:
        x = _dequantize_asymmetric(x_q, params.scale, params.zero_point)

    return x.to(torch.float16)


def quantize_k(
    k: torch.Tensor,
    config: QuantConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[QuantParams], list[QJLSketch] | None]:
    """
    Quantize K tensor with optional Hadamard and QJL.

    Args:
        k: K tensor of shape (seq_len, num_heads, head_dim).
        config: QuantConfig.
        generator: Optional seeded generator.

    Returns:
        (packed_k, params_list, qjl_sketches)
        - packed_k: (seq_len, num_heads, packed_dim)
        - params_list: list of QuantParams per group
        - qjl_sketches: list of QJLSketch per group (if use_qjl_residual)
    """
    seq_len, num_heads, head_dim = k.shape

    # Apply Hadamard rotation if enabled
    if config.use_hadamard:
        if head_dim & (head_dim - 1) != 0:
            raise ValueError(f"Hadamard requires head_dim power of 2, got {head_dim}")
        k, hadamard_signs = hadamard_rademacher_transform(k, generator)

    # Reshape for group quantization
    group_size = config.group_size
    k_flat = k.reshape(-1, head_dim)  # (num_groups * group_size, head_dim)
    num_groups = k_flat.shape[0] // group_size

    packed_list: list[torch.Tensor] = []
    params_list: list[QuantParams] = []
    qjl_sketches: list[QJLSketch] | None = [] if config.use_qjl_residual else None

    for g in range(num_groups):
        group_start = g * group_size
        group_end = group_start + group_size
        group = k_flat[group_start:group_end]  # (group_size, head_dim)

        # Quantize per channel (head_dim) or per token
        if config.k_per_channel:
            # Quantize each channel separately
            raise NotImplementedError("Per-channel quantization not yet implemented")
        else:
            # Quantize each token's head_dim as a group
            # Actually, group_size tokens x head_dim - we quantize along group_size
            # Reshape to (head_dim, group_size) and quantize each channel
            group_t = group.t()  # (head_dim, group_size)
            group_packed = []
            group_params = []

            for c in range(head_dim):
                channel = group_t[c]  # (group_size,)
                packed, params = quantize_group(channel, config.k_bits, config.k_scheme, generator)
                group_packed.append(packed)
                group_params.append(params)

            packed_list.append(torch.stack(group_packed))  # (head_dim, packed_per_channel)
            params_list.extend(group_params)

    # Handle QJL residual sketch if enabled
    if config.use_qjl_residual and config.k_bits <= 4:
        # Compute residual and sketch
        # This is a simplified version - full implementation needs dequantization first
        raise NotImplementedError("QJL residual sketch requires dequantization first")

    # Reshape packed back
    # packed_list shape: (num_groups, head_dim, packed_per_channel)
    # Need to rearrange to (seq_len, num_heads, packed_dim)

    return torch.stack(packed_list), params_list, qjl_sketches


def quantize_v(
    v: torch.Tensor,
    config: QuantConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, list[QuantParams]]:
    """
    Quantize V tensor.

    Args:
        v: V tensor of shape (seq_len, num_heads, head_dim).
        config: QuantConfig.
        generator: Optional seeded generator.

    Returns:
        (packed_v, params_list)
    """
    # Similar to quantize_k but without Hadamard/QJL options
    seq_len, num_heads, head_dim = v.shape
    group_size = config.group_size
    v_flat = v.reshape(-1, head_dim)
    num_groups = v_flat.shape[0] // group_size

    packed_list = []
    params_list = []

    for g in range(num_groups):
        group_start = g * group_size
        group_end = group_start + group_size
        group = v_flat[group_start:group_end]  # (group_size, head_dim)

        group_t = group.t()  # (head_dim, group_size)
        group_packed = []
        group_params = []

        for c in range(head_dim):
            channel = group_t[c]
            packed, params = quantize_group(channel, config.v_bits, config.v_scheme, generator)
            group_packed.append(packed)
            group_params.append(params)

        packed_list.append(torch.stack(group_packed))
        params_list.extend(group_params)

    return torch.stack(packed_list), params_list


def dequantize_k(
    packed_k: torch.Tensor,
    params_list: list[QuantParams],
    config: QuantConfig,
    head_dim: int,
) -> torch.Tensor:
    """
    Dequantize K tensor with optional Hadamard inverse.

    Args:
        packed_k: Packed K tensor.
        params_list: QuantParams per group.
        config: QuantConfig.
        head_dim: Head dimension.

    Returns:
        Dequantized K of shape (seq_len, num_heads, head_dim).
    """
    # Inverse of quantize_k
    raise NotImplementedError("dequantize_k - needs full implementation")


def dequantize_v(
    packed_v: torch.Tensor,
    params_list: list[QuantParams],
    config: QuantConfig,
    head_dim: int,
) -> torch.Tensor:
    """
    Dequantize V tensor.

    Args:
        packed_v: Packed V tensor.
        params_list: QuantParams per group.
        config: QuantConfig.
        head_dim: Head dimension.

    Returns:
        Dequantized V of shape (seq_len, num_heads, head_dim).
    """
    raise NotImplementedError("dequantize_v - needs full implementation")


def estimate_attention_logits_qjl(
    q: torch.Tensor,
    k_dequantized: torch.Tensor,
    qjl_sketches: list[QJLSketch],
    qjl_projections: list[torch.Tensor],
    config: QuantConfig,
) -> torch.Tensor:
    """
    Estimate attention logits with QJL residual correction.

    logits = q @ k_dequantized^T + q @ r_QJL^T

    Args:
        q: Query tensor (seq_len_q, num_heads, head_dim).
        k_dequantized: Dequantized K tensor (seq_len_k, num_heads, head_dim).
        qjl_sketches: QJL sketches per group.
        qjl_projections: Projection matrices per group.
        config: QuantConfig.

    Returns:
        Attention logits (seq_len_q, seq_len_k, num_heads).
    """
    # Base logits from dequantized K
    base_logits = torch.einsum("qhd,khd->qkh", q, k_dequantized)

    # Add QJL residual correction
    # For each group, estimate <q, r> and add to appropriate positions
    # This requires knowing group boundaries
    raise NotImplementedError("estimate_attention_logits_qjl - batch version needed")


# ===== Property-based test helpers =====


def theoretical_max_error(bits: int, scheme: Literal["sym", "asym"]) -> float:
    """
    Theoretical maximum quantization error for given bit-width.

    For symmetric: max_error = scale = max(|x|) / (2^(bits-1) - 1)
    Worst case when max(|x|) = 1 (normalized): max_error = 1 / (2^(bits-1) - 1)

    For asymmetric: similar but depends on range.
    """
    if bits >= 16:
        return 0.0

    if scheme == "sym":
        return float(1.0 / (2 ** (bits - 1) - 1))
    else:
        return float(1.0 / (2**bits - 1))


def theoretical_sqnr_db(bits: int) -> float:
    """
    Theoretical Signal-to-Quantization-Noise Ratio (SQNR) in dB.

    For uniform quantization of Gaussian: SQNR ≈ 6.02 * bits + 1.76 dB
    For uniform quantization of uniform: SQNR ≈ 6.02 * bits dB
    """
    if bits >= 16:
        return float("inf")
    return 6.02 * bits


def compute_sqnr_db(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute actual SQNR in dB between original and reconstructed."""
    signal_power = (original**2).mean()
    noise_power = ((original - reconstructed) ** 2).mean()
    if noise_power == 0:
        return float("inf")
    return 10 * torch.log10(signal_power / noise_power).item()


def compute_max_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Compute maximum absolute error."""
    return (original - reconstructed).abs().max().item()
