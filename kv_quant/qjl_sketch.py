"""1-Bit Quantized Johnson-Lindenstrauss (QJL) Sketch.

For k_bits=2 (or 4), compute quantization residual r = K - K_hat,
project with seeded Rademacher/Gaussian sketch z_k = sign(S r) ∈ {-1, +1}^D
packed into uint32 words alongside FP16 residual norm ||r||_2,
and provide the unbiased estimator:
    <q, r>_QJL = sqrt(π/2) / D * ||r||_2 * <S q, z_k>
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QJLSketch:
    """Packed QJL sketch for a quantization group."""
    # Sign bits packed into uint32 words (32 signs per word)
    signs: torch.Tensor      # shape (num_words,), dtype=torch.uint32
    # Residual L2 norm (FP16)
    norm: torch.Tensor       # shape (), dtype=torch.float16
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
    proj = torch.randint(0, 2, (total_bits, dim), generator=generator, device=device, dtype=torch.int8)
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
        packed |= (signs[:, :, i].to(torch.int64) << i)
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
        packed |= (signs[..., i].to(torch.int64) << i)
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
    signs = torch.zeros(*packed.shape[:-1], num_words * 32, dtype=torch.float32, device=packed.device)

    for i in range(32):
        bit = (packed_int >> i) & 1
        signs[..., i::32] = bit.to(torch.float32) * 2 - 1

    return signs[..., :dim]


# Constants for QJL
QJL_SCALE_FACTOR = math.sqrt(math.pi / 2)  # sqrt(π/2) ≈ 1.2533
