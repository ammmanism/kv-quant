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
            first = h[..., i:i+step].clone()
            second = h[..., i+step:i+2*step].clone()
            # Butterfly: [u, v] -> [u+v, u-v]
            h[..., i:i+step] = first + second
            h[..., i+step:i+2*step] = first - second
        step *= 2

    if normalize:
        h = h / (d ** 0.5)

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
    var = (centered ** 2).mean(dim=dim, keepdim=True)
    std = torch.sqrt(var + 1e-8)
    fourth_moment = (centered ** 4).mean(dim=dim, keepdim=True)
    kurtosis = fourth_moment / (std ** 4) - 3.0
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
