"""1-Bit Quantized Johnson-Lindenstrauss (QJL) Sketch.

For k_bits=2 (or 4), compute quantization residual r = K - K_hat,
project with seeded Rademacher/Gaussian sketch z_k = sign(S r) ∈ {-1, +1}^D
packed into uint32 words alongside FP16 residual norm ||r||_2,
and provide the unbiased estimator:
    <q, r>_QJL = sqrt(π/2) / D * ||r||_2 * <S q, z_k>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:  # pragma: no cover - avoids a runtime import cycle
    from kv_quant.config import QuantConfig
    from kv_quant.quantizer import QuantParams


@dataclass(frozen=True)
class QJLSketch:
    """Packed QJL sketch for a quantization group."""

    # Sign bits packed into uint32 words (32 signs per word)
    signs: torch.Tensor  # shape (num_words,), dtype=torch.uint32
    # Residual L2 norm (FP16)
    norm: torch.Tensor  # shape (), dtype=torch.float16
    # Number of dimensions (for unpacking)
    dim: int


def generate_rademacher_projection(
    dim: int,
    num_words: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate Rademacher projection matrix S ∈ {-1, +1}^(num_words*32, dim).

    Each row is a random Rademacher vector (±1 with equal probability).

    Args:
        dim: Input dimension.
        num_words: Number of uint32 output words (each holds 32 signs).
        generator: Seeded generator for reproducibility.
        device: Target device.

    Returns:
        Projection matrix of shape (num_words * 32, dim), dtype=int8.
    """
    if generator is None:
        generator = torch.Generator(device=device)

    total_bits = num_words * 32
    # Generate ±1 matrix
    proj = torch.randint(
        0, 2, (total_bits, dim), generator=generator, device=device, dtype=torch.int8
    )
    proj = proj * 2 - 1  # Convert 0/1 to -1/+1

    return proj


def generate_gaussian_projection(
    dim: int,
    num_words: int,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate Gaussian projection matrix S ∈ R^(num_words*32, dim).

    Args:
        dim: Input dimension.
        num_words: Number of uint32 output words.
        generator: Seeded generator.
        device: Target device.

    Returns:
        Projection matrix of shape (num_words * 32, dim), dtype=float32.
    """
    if generator is None:
        generator = torch.Generator(device=device)

    total_bits = num_words * 32
    proj = torch.randn(total_bits, dim, generator=generator, device=device, dtype=torch.float32)

    return proj


def compute_qjl_sketch(
    residual: torch.Tensor,
    projection: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    use_rademacher: bool = True,
) -> QJLSketch:
    """
    Compute 1-bit QJL sketch of quantization residual.

    Args:
        residual: Residual tensor of shape (..., D) where D is the head dimension.
        projection: Optional precomputed projection matrix (num_bits, D).
        generator: Seeded generator.
        use_rademacher: If True, use Rademacher (±1) projection. If False, use Gaussian.

    Returns:
        QJLSketch with packed signs and residual norm.
    """
    # Flatten all but last dimension
    orig_shape = residual.shape
    d = orig_shape[-1]
    residual_flat = residual.reshape(-1, d)

    # Compute L2 norm per vector
    norms = torch.norm(residual_flat, p=2, dim=-1, keepdim=True)  # (N, 1)

    # Normalize residual for sign computation
    # Avoid division by zero
    normalized = residual_flat / (norms + 1e-8)

    # Generate projection if not provided
    if projection is not None:
        # Derive num_words from the provided projection shape
        total_bits = projection.shape[0]
        num_words = (total_bits + 31) // 32
    else:
        num_words = (d + 31) // 32  # Round up to 32 bits per word
        total_bits = num_words * 32
        if use_rademacher:
            projection = generate_rademacher_projection(d, num_words, generator, residual.device)
        else:
            projection = generate_gaussian_projection(d, num_words, generator, residual.device)

    # Project: (N, D) @ (D, num_bits) -> (N, num_bits)
    # For Rademacher: signs = sign(S @ r)
    projected = normalized @ projection.T.to(residual.dtype)  # (N, num_bits)

    # Get signs
    signs = (projected > 0).to(torch.uint8)  # 0 or 1

    # Pack into uint32 words
    # Reshape to (N, num_words, 32)
    signs = signs.view(-1, num_words, 32)

    # Pack 32 bits per word - use int64 for bitwise operations, then cast to uint32
    packed = torch.zeros(signs.shape[:2], dtype=torch.int64, device=residual.device)
    for i in range(32):
        packed |= signs[:, :, i].to(torch.int64) << i
    packed = packed.to(torch.uint32)

    # Return first vector's sketch (assuming batch size 1 for now)
    # In practice, we'd return one sketch per group
    return QJLSketch(
        signs=packed[0],  # (num_words,)
        norm=norms[0].to(torch.float16).squeeze(),  # ()
        dim=d,
    )


def estimate_qjl_inner_product(
    q: torch.Tensor,
    sketch: QJLSketch,
    projection: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    use_rademacher: bool = True,
) -> torch.Tensor:
    """
    Unbiased estimator for <q, r> using 1-bit QJL sketch.

    Formula:
        <q, r>_QJL = sqrt(π/2) / D * ||r||_2 * <S q, z_k>

    where z_k = sign(S r) ∈ {-1, +1}^D are the sketch signs.

    Args:
        q: Query vector of shape (D,).
        sketch: QJLSketch of the residual r.
        projection: Optional projection matrix S (must match the one used for sketch).
        generator: Seeded generator.
        use_rademacher: Whether Rademacher projection was used.

    Returns:
        Estimated inner product <q, r>.
    """
    d = sketch.dim

    # Generate or use projection - must match the one used in compute_qjl_sketch
    num_words = sketch.signs.shape[0]
    total_bits = num_words * 32
    if projection is None:
        if use_rademacher:
            projection = generate_rademacher_projection(d, num_words, generator, q.device)
        else:
            projection = generate_gaussian_projection(d, num_words, generator, q.device)

    # Project q: S q  -> shape (total_bits,)
    proj_q = q.to(torch.float32).reshape(-1) @ projection.T.to(torch.float32)  # (total_bits,)

    # Unpack sketch signs - use int64 for bitwise operations
    # Convert packed uint32 back to ±1
    signs_int = sketch.signs.to(torch.int64)
    signs = torch.zeros(total_bits, dtype=torch.float32, device=q.device)
    for i in range(32):
        bit = (signs_int >> i) & 1
        signs[i::32] = bit.to(torch.float32) * 2 - 1  # Convert 0/1 to -1/+1

    # Compute <S q, z_k> over all m sketch rows
    ip = (proj_q * signs).sum()

    # Unbiased estimator: <q,r> ≈ sqrt(π/2)/m * ||r|| * <S q, z_k>
    # For Gaussian S: E[sign(g·r)(g·q)] = sqrt(2/π) * <q,r>/||r||
    # Note: Rademacher projection is a common practical approximation.
    m = total_bits
    scale = math.sqrt(math.pi / 2) / m * sketch.norm.to(torch.float32)

    return scale * ip


def estimate_attention_logits_qjl(
    q: torch.Tensor,
    k_quantized: torch.Tensor,
    k_dequantized: torch.Tensor,
    sketch: QJLSketch,
    projection: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Estimate attention logits with QJL residual correction.

    logits = q @ k_dequantized^T + q @ r_QJL^T

    where r = k - k_dequantized is the quantization residual.

    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        k_quantized: Quantized key tensor (for reference).
        k_dequantized: Dequantized key tensor of shape (seq_len_k, num_heads, head_dim).
        sketch: QJLSketch of the key residual (per head/group).
        projection: Optional projection matrix.
        generator: Seeded generator.

    Returns:
        Attention logits with QJL correction: (seq_len_q, seq_len_k, num_heads).
    """
    # This is a simplified version - in practice we'd have per-group sketches
    # For now, assume single sketch per head
    raise NotImplementedError("estimate_attention_logits_qjl - batch version needed")


def pack_qjl_signs(signs: torch.Tensor) -> torch.Tensor:
    """
    Pack boolean signs tensor into uint32 words.

    Args:
        signs: Tensor of shape (..., D) with values in {0, 1} or {False, True}.

    Returns:
        Packed tensor of shape (..., ceil(D/32)), dtype=uint32.
    """
    d = signs.shape[-1]
    num_words = (d + 31) // 32

    # Pad to multiple of 32
    pad = num_words * 32 - d
    if pad > 0:
        signs = torch.nn.functional.pad(signs, (0, pad), value=0)

    # Reshape and pack - use int64 for bitwise operations
    signs = signs.reshape(*signs.shape[:-1], num_words, 32)
    packed = torch.zeros(signs.shape[:-1], dtype=torch.int64, device=signs.device)

    for i in range(32):
        packed |= signs[..., i].to(torch.int64) << i
    packed = packed.to(torch.uint32)

    return packed


def unpack_qjl_signs(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Unpack uint32 words back to boolean signs.

    Args:
        packed: Tensor of shape (..., num_words), dtype=uint32.
        dim: Original dimension D.

    Returns:
        Signs tensor of shape (..., D) with values in {-1, +1}.
    """
    num_words = packed.shape[-1]
    # Use int64 for bitwise operations
    packed_int = packed.to(torch.int64)
    signs = torch.zeros(
        *packed.shape[:-1], num_words * 32, dtype=torch.float32, device=packed.device
    )

    for i in range(32):
        bit = (packed_int >> i) & 1
        signs[..., i::32] = bit.to(torch.float32) * 2 - 1

    return signs[..., :dim]


# Constants for QJL
QJL_SCALE_FACTOR = math.sqrt(math.pi / 2)  # sqrt(π/2) ≈ 1.2533


# ---------------------------------------------------------------------------
# v1.3.0 Gaussian QJL reference prototype (ADR-010)
#
# The functions below implement the EXACT Gaussian identity. The legacy
# Rademacher helpers above are retained byte-stable for p1 artifact
# reproduction; new code must use the Gaussian contract.
# ---------------------------------------------------------------------------

# Prototype packing contract: M = 64 sign bits = two uint32 words.
QJL_M_BITS = 64


@dataclass(frozen=True)
class Int2QJLPayload:
    """Self-contained INT2 + Gaussian-QJL sketch for one key vector.

    Contents (per the ADR-010 packing contract):
      - ``packed_int2``: the key quantized to symmetric INT2, 4 crumbs per
        byte, via the pinned ``quantizer`` layout (Invariant 3).
      - ``int2_params``: scale/zero-point of the INT2 base grid.
      - ``signs``: M packed sign bits of the Gaussian projection of the
        residual (two uint32 words at the default M=64).
      - ``residual_norm``: FP16 ||k - k_hat||_2.
    """

    packed_int2: torch.Tensor  # (ceil(D/4),) uint8
    int2_params: QuantParams
    signs: torch.Tensor  # (ceil(M/32),) uint32
    residual_norm: torch.Tensor  # () float16
    dim: int
    m_bits: int
    scheme: Literal["sym", "asym"] = "sym"


def generate_gaussian_proj(
    dim: int, m_bits: int, seed: int, device: torch.device | None = None
) -> torch.Tensor:
    """Seeded i.i.d. standard-normal projection S of shape (m_bits, dim).

    Gaussian entries are rotationally invariant: for any unit vector u the
    projection S u is exactly N(0, I_m) in law. That is the property the
    arcsine/half-normal proof needs and the reason this module refuses
    Rademacher S for the unbiased contract (see
    ``docs/HADAMARD_AND_QJL_MATH.md``).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(m_bits, dim, generator=generator, dtype=torch.float32)
    return proj.to(device) if device is not None else proj


def quantize_k_int2_qjl(
    k: torch.Tensor,
    cfg: QuantConfig | None = None,
    gaussian_proj: torch.Tensor | None = None,
    seed: int = 42,
    m_bits: int = QJL_M_BITS,
) -> Int2QJLPayload:
    """Quantize one key vector to INT2 + Gaussian QJL residual sketch.

    Contract: r = k - dequant(quantize(k, INT2, sym)) is sketched as
    sign(S r) with S i.i.d. N(0,1) of shape (M, D), packed M bits per the
    layout helper, plus the FP16 residual norm. The estimator
    ``estimate_qjl_logits_ref`` is then UNBIASED for <q, k>:

        E[ sqrt(pi/2)/M * ||r|| * <S q, sign(S r)> ] = <q, r>

    Why Gaussian and not Rademacher: the identity relies on rotational
    invariance -- for s ~ N(0, I_D) and fixed r, sign(<s, r>) * <s, q>
    has expectation sqrt(2/pi) * <q, r> / ||r|| EXACTLY. A Rademacher row
    is invariant only under coordinate permutations and sign flips, not
    general rotations, so the half-normal factor does not apply and the
    estimator acquires a direction-dependent bias.

    Args:
        k: Key vector of shape (D,), FP16/BF16/FP32.
        cfg: Optional ``QuantConfig``; the prototype supports symmetric
            INT2 only (``cfg`` accepted for interface compatibility).
        gaussian_proj: Optional precomputed (M, D) Gaussian projection;
            generated from ``seed`` when omitted.
        seed: Projection seed (used only when ``gaussian_proj`` is None).
        m_bits: Sketch width M; default 64 (two uint32 words).

    Returns:
        ``Int2QJLPayload`` holding packed INT2 crumbs, the base-grid
        parameters, the packed signs and the FP16 residual norm.
    """
    from kv_quant.quantizer import dequantize_group, quantize_group

    if cfg is not None and cfg.k_scheme != "sym":
        raise ValueError("the INT2+QJL prototype supports only the symmetric scheme")
    if k.dim() != 1:
        raise ValueError(f"expected a single key vector of shape (D,), got {tuple(k.shape)}")
    dim = k.shape[-1]
    if gaussian_proj is None:
        gaussian_proj = generate_gaussian_proj(dim, m_bits, seed)
    if tuple(gaussian_proj.shape) != (m_bits, dim):
        raise ValueError(f"gaussian_proj shape {tuple(gaussian_proj.shape)} != ({m_bits}, {dim})")

    k32 = k.to(torch.float32)
    packed, params = quantize_group(k32, 2, "sym")
    k_hat = dequantize_group(packed, params, 2, "sym")
    residual = k32 - k_hat.to(torch.float32)
    norm = residual.norm()

    projected = gaussian_proj.to(torch.float32) @ residual
    sign_bits = (projected > 0).to(torch.uint8)
    sign_bits = (
        torch.nn.functional.pad(sign_bits, (0, m_bits - sign_bits.shape[0]), value=0)
        if sign_bits.shape[0] < m_bits
        else sign_bits
    )
    packed_signs = pack_qjl_signs(sign_bits)

    return Int2QJLPayload(
        packed_int2=packed,
        int2_params=params,
        signs=packed_signs,
        residual_norm=norm.to(torch.float16),
        dim=dim,
        m_bits=sign_bits.shape[0],
        scheme="sym",
    )


def estimate_qjl_logits_ref(
    q: torch.Tensor,
    qb_qjl: Int2QJLPayload,
    gaussian_proj: torch.Tensor | None = None,
    seed: int = 42,
) -> float:
    """Unbiased estimate of the attention logit <q, k> from the payload.

    ``hat_ell = <q, k_hat> + sqrt(pi/2)/M * ||r|| * <S q, sign(S r)>`` where
    the first term is the deterministic INT2 base and the second is the
    Gaussian-QJL correction of the quantization residual. The estimator is
    unbiased for ``<q, k>`` up to the FP16 rounding of the stored norm.

    Args:
        q: Query vector of shape (D,).
        qb_qjl: Payload produced by ``quantize_k_int2_qjl``.
        gaussian_proj: The SAME projection used at quantization time;
            regenerated from ``seed`` when omitted.
        seed: Projection seed (must match the quantization-side seed).

    Returns:
        The estimated logit as a Python float.
    """
    from kv_quant.quantizer import dequantize_group

    if q.shape[-1] != qb_qjl.dim:
        raise ValueError(f"q dim {q.shape[-1]} != payload dim {qb_qjl.dim}")
    proj = gaussian_proj
    if proj is None:
        proj = generate_gaussian_proj(qb_qjl.dim, qb_qjl.m_bits, seed)

    k_hat = dequantize_group(qb_qjl.packed_int2, qb_qjl.int2_params, 2, qb_qjl.scheme)
    base = float((q.to(torch.float32) @ k_hat.to(torch.float32)).item())

    signs = unpack_qjl_signs(qb_qjl.signs, qb_qjl.m_bits).to(torch.float32)
    proj_q = proj.to(torch.float32) @ q.to(torch.float32)
    correction = (
        QJL_SCALE_FACTOR
        / qb_qjl.m_bits
        * float(qb_qjl.residual_norm.to(torch.float32).item())
        * float((proj_q * signs).sum().item())
    )
    return base + correction


def naive_int2_logit(q: torch.Tensor, qb_qjl: Int2QJLPayload) -> float:
    """The naive INT2 logit <q, k_hat> WITHOUT the QJL residual correction."""
    from kv_quant.quantizer import dequantize_group

    k_hat = dequantize_group(qb_qjl.packed_int2, qb_qjl.int2_params, 2, qb_qjl.scheme)
    return float((q.to(torch.float32) @ k_hat.to(torch.float32)).item())
