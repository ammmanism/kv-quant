"""Fast Walsh-Hadamard Transform (FWHT) with Rademacher rotation.

Implements orthogonal Sylvester-Hadamard transform H_D / sqrt(D) in O(D log D)
butterfly stages combined with seeded Rademacher sign flips S = diag(±1).

Rotating post-RoPE Q' = Q S H_D and K' = K S H_D preserves exact inner products
<Q', K'> = <Q, K> while reducing channel kurtosis from κ > 25 to κ ≈ 3.0.
"""

import torch


def hadamard_transform(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Apply Fast Walsh-Hadamard Transform (FWHT) to the last dimension.

    The transform is orthogonal: H_D @ H_D^T = D * I, so H_D / sqrt(D) is unitary.

    Args:
        x: Input tensor of shape (..., D) where D is a power of 2.
        generator: Optional seeded generator for reproducibility (unused, for API consistency).
        normalize: If True, return H_D / sqrt(D) @ x (orthonormal). If False, return H_D @ x.

    Returns:
        Hadamard-transformed tensor of same shape and dtype as x.
    """
    # Validate last dimension is power of 2
    d = x.shape[-1]
    if d & (d - 1) != 0:
        raise ValueError(f"Last dimension must be power of 2, got {d}")

    # Work in float32 for numerical stability
    dtype = x.dtype
    x = x.to(torch.float32)

    # FWHT butterfly algorithm - use standard iterative approach
    # Must clone slices to avoid view aliasing issues during in-place updates
    h = x.clone()

    step = 1
    while step < d:
        for i in range(0, d, 2 * step):
            # Clone to avoid view aliasing: first and second must be independent of h
            first = h[..., i : i + step].clone()
            second = h[..., i + step : i + 2 * step].clone()
            # Butterfly: [u, v] -> [u+v, u-v]
            h[..., i : i + step] = first + second
            h[..., i + step : i + 2 * step] = first - second
        step *= 2

    if normalize:
        h = h / (d**0.5)

    return h.to(dtype)


def inverse_hadamard_transform(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Apply inverse Walsh-Hadamard Transform.

    For normalized FWHT, H_D / sqrt(D) is self-inverse (orthogonal).
    For unnormalized, inverse is H_D / D.

    Args:
        x: Input tensor of shape (..., D) where D is a power of 2.
        generator: Optional seeded generator (unused).
        normalize: If True, assumes forward used normalize=True.

    Returns:
        Inverse-transformed tensor of same shape and dtype.
    """
    # For orthonormal FWHT, forward == inverse
    return hadamard_transform(x, generator, normalize)


def rademacher_rotation(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random Rademacher sign flips S = diag(±1) to the last dimension.

    Args:
        x: Input tensor of shape (..., D).
        generator: Seeded generator for reproducibility.

    Returns:
        (rotated_x, signs) where rotated_x = x * signs and signs is the Rademacher vector.
    """
    d = x.shape[-1]
    if generator is None:
        generator = torch.Generator(device=x.device)

    # Generate Rademacher signs: ±1 with equal probability
    signs = torch.randint(0, 2, (d,), generator=generator, device=x.device, dtype=torch.int8)
    signs = signs.to(x.dtype) * 2 - 1  # Convert 0/1 to -1/+1

    # Apply sign flips
    rotated = x * signs

    return rotated, signs


def hadamard_rademacher_transform(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rademacher rotation followed by FWHT: x -> H_D @ (S @ x) / sqrt(D).

    This is the full outlier-eradicating transform:
    Q' = Q @ S @ H_D / sqrt(D)
    K' = K @ S @ H_D / sqrt(D)

    Args:
        x: Input tensor of shape (..., D) where D is a power of 2.
        generator: Seeded generator for Rademacher signs.
        normalize: If True, apply 1/sqrt(D) normalization.

    Returns:
        (transformed_x, signs) where transformed_x = H_D @ (S @ x) / sqrt(D)
    """
    # Apply Rademacher sign flips
    rotated, signs = rademacher_rotation(x, generator)

    # Apply FWHT
    transformed = hadamard_transform(rotated, generator, normalize)

    return transformed, signs


def hadamard_rademacher_inverse(
    x: torch.Tensor,
    signs: torch.Tensor,
    generator: torch.Generator | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Inverse of hadamard_rademacher_transform.

    Since H_D is self-inverse (orthonormal) and S is diagonal with ±1 (self-inverse):
    x = S @ H_D @ x' / sqrt(D)

    Args:
        x: Transformed tensor of shape (..., D).
        signs: Rademacher signs from forward transform.
        generator: Optional generator (unused).
        normalize: If True, assumes forward used normalize=True.

    Returns:
        Original tensor before transformation.
    """
    # Apply inverse FWHT (same as forward for orthonormal)
    h_inv = inverse_hadamard_transform(x, generator, normalize)

    # Apply inverse Rademacher (same as forward since S = S^-1)
    original = h_inv * signs

    return original


def compute_kurtosis(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute excess kurtosis along a dimension.

    Kurtosis = E[(X - μ)^4] / σ^4 - 3
    For Gaussian: kurtosis = 0 (excess) or 3 (Fisher)
    We return excess kurtosis (0 for Gaussian).

    Args:
        x: Input tensor.
        dim: Dimension along which to compute kurtosis.

    Returns:
        Excess kurtosis tensor (scalar if dim reduces all).
    """
    x_float = x.to(torch.float32)
    mean = x_float.mean(dim=dim, keepdim=True)
    centered = x_float - mean
    var = (centered**2).mean(dim=dim, keepdim=True)
    std = torch.sqrt(var + 1e-8)
    fourth_moment = (centered**4).mean(dim=dim, keepdim=True)
    kurtosis = fourth_moment / (std**4) - 3.0
    return kurtosis.squeeze(dim)


def verify_orthogonality(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    atol: float = 1e-5,
) -> bool:
    """
    Verify that the Hadamard-Rademacher transform preserves inner products.

    For any vectors q, k: <Q', K'> = <Q, K> where Q' = H_D @ S @ Q / sqrt(D).

    Args:
        x: Test tensor of shape (..., D).
        generator: Seeded generator.
        atol: Absolute tolerance for verification.

    Returns:
        True if inner products are preserved within tolerance.
    """
    # Test with random vectors
    d = x.shape[-1]
    q = torch.randn_like(x[..., :1])  # (..., 1, D)
    k = torch.randn_like(x[..., :1])

    # Original inner product
    orig_ip = (q @ k.transpose(-2, -1)).squeeze()

    # Transform both
    q_prime, signs = hadamard_rademacher_transform(q, generator)
    k_prime, _ = hadamard_rademacher_transform(k, generator)

    # Transformed inner product
    trans_ip = (q_prime @ k_prime.transpose(-2, -1)).squeeze()

    return torch.allclose(orig_ip, trans_ip, atol=atol)


def _rademacher_signs(n: int, seed: int) -> torch.Tensor:
    """Seeded Rademacher sign vector D = diag(+-1) of length n on CPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    bits = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int8)
    return bits.to(torch.float32) * 2.0 - 1.0


def _require_pow2(n: int, what: str) -> None:
    if n <= 0 or n & (n - 1) != 0:
        raise ValueError(f"{what} must be a positive power of 2, got {n}")


def fast_walsh_hadamard(
    x: torch.Tensor,
    scope: str = "full",
    rotary_dim: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Orthogonal seeded Walsh-Hadamard rotation on the last dimension.

    Both scopes define an orthogonal map R, so rotating Q and K preserves
    inner products exactly: <x R, y R> = <x, y> (inner-product invariance).

    ``scope="full"`` (Full-RoPE models, e.g. Llama):
        x' = H_D (D_seed x) / sqrt(D), with D = head_dim, a power of 2.

    ``scope="block_diag"`` (Partial-RoPE models, e.g. Qwen/Mistral variants):
        x' = blkdiag(D1 H_r / sqrt(r), D2 H_p / sqrt(p)) x
        where r = rotary_dim and p = D - r. Independent seeded sign vectors
        D1 (seed) and D2 (seed + 1) act on the two subspaces. When p is not a
        power of 2 the pass-through block is the identity I_p (documented
        contract), which keeps the map orthogonal. Rotating only the rotary
        subspace avoids contaminating position-phase channels with
        pass-through content.

    Args:
        x: Input tensor of shape (..., D).
        scope: "full" or "block_diag".
        rotary_dim: Rotary width for block_diag; defaults to D (full width).
        seed: Seed for the Rademacher sign flips.

    Returns:
        Rotated tensor with the same shape and dtype as x. The input is never
        mutated. The map is orthogonal (R^T R = I); its inverse is the
        transpose D H, provided by :func:`inverse_fast_walsh_hadamard`.

    Raises:
        ValueError: For an unknown scope, non-power-of-2 full width, invalid
            rotary width, or non-power-of-2 rotary width.
    """
    d = x.shape[-1]
    if scope == "full":
        _require_pow2(d, "head_dim for scope='full'")
        signs = _rademacher_signs(d, seed).to(x.device)
        return hadamard_transform(x * signs)
    if scope != "block_diag":
        raise ValueError(f"unknown hadamard scope: {scope!r} (expected 'full' or 'block_diag')")

    r = d if rotary_dim is None else rotary_dim
    if not 0 < r <= d:
        raise ValueError(f"rotary_dim={r} must be in (0, {d}]")
    _require_pow2(r, "rotary_dim")

    signs_rot = _rademacher_signs(r, seed).to(x.device)
    rotated = hadamard_transform(x[..., :r] * signs_rot)

    p = d - r
    if p == 0:
        return rotated
    if p & (p - 1) == 0:
        signs_pass = _rademacher_signs(p, seed + 1).to(x.device)
        passthrough = hadamard_transform(x[..., r:] * signs_pass)
    else:
        passthrough = x[..., r:]
    return torch.cat([rotated, passthrough], dim=-1)


def hadamard_matrix(
    head_dim: int,
    scope: str = "full",
    rotary_dim: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Materialize the rotation matrix R (head_dim x head_dim) for inspection.

    R is orthogonal: R^T R = I. Useful for exact invariance proofs in tests;
    the production path applies the transform without materializing R.
    """
    eye = torch.eye(head_dim, dtype=torch.float32)
    rotated = fast_walsh_hadamard(eye, scope=scope, rotary_dim=rotary_dim, seed=seed)
    return rotated.reshape(head_dim, head_dim)


def inverse_fast_walsh_hadamard(
    x: torch.Tensor,
    scope: str = "full",
    rotary_dim: int | None = None,
    seed: int = 42,
) -> torch.Tensor:
    """Exact inverse of :func:`fast_walsh_hadamard` (the transpose map D H).

    The forward map is R = H D (signs first, then butterfly); orthogonal maps
    satisfy R^-1 = R^T = D H, so the inverse applies the butterfly first and
    the seeded signs afterwards. Same seed and scope as the forward call.
    """
    d = x.shape[-1]
    if scope == "full":
        _require_pow2(d, "head_dim for scope='full'")
        signs = _rademacher_signs(d, seed).to(x.device)
        return hadamard_transform(x) * signs
    if scope != "block_diag":
        raise ValueError(f"unknown hadamard scope: {scope!r} (expected 'full' or 'block_diag')")

    r = d if rotary_dim is None else rotary_dim
    if not 0 < r <= d:
        raise ValueError(f"rotary_dim={r} must be in (0, {d}]")
    _require_pow2(r, "rotary_dim")

    signs_rot = _rademacher_signs(r, seed).to(x.device)
    restored_rot = hadamard_transform(x[..., :r]) * signs_rot

    p = d - r
    if p == 0:
        return restored_rot
    if p & (p - 1) == 0:
        signs_pass = _rademacher_signs(p, seed + 1).to(x.device)
        restored_pass = hadamard_transform(x[..., r:]) * signs_pass
    else:
        restored_pass = x[..., r:]
    return torch.cat([restored_rot, restored_pass], dim=-1)
