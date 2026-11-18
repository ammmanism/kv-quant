"""Fused Triton Walsh-Hadamard Q/K rotation with a PyTorch reference path.

Rotates post-RoPE ``Q`` and ``K`` into the Hadamard basis *before* they enter
``HybridPagedKVCache.append()``. Because both operands are rotated by the same
orthogonal map ``R``, ``dequant_fused`` computes ``<Q', K'> = <Q R, K R> =
<Q, K>`` directly in that basis, so the memory-bound KV read loop gains **zero
extra ops**.

Verification status (ADR-006, ADR-008): Triton ships no win32 wheel and the
current host has no CUDA device, so the code below is GPU-only. The
correctness reference is :func:`kv_quant.hadamard.fast_walsh_hadamard`; the
parity gate is ``tests/test_hadamard.py::test_gpu_triton_rotation_matches_
reference``, which runs on a CUDA host and skips elsewhere. Until that gate has
run on a GPU, treat the device path as **unverified**.
"""

from __future__ import annotations

import torch

from kv_quant.hadamard import _rademacher_signs, fast_walsh_hadamard

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - Windows has no triton wheel
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

HAS_TRITON = _HAS_TRITON
HAS_CUDA = torch.cuda.is_available()
GPU_KERNEL_AVAILABLE = HAS_TRITON and HAS_CUDA


# ---------------------------------------------------------------------------
# Triton kernels (compiled lazily; only launched when CUDA is available)
# ---------------------------------------------------------------------------

if _HAS_TRITON:

    @triton.jit
    def _fwht_region(x, cols, START: tl.constexpr, WIDTH: tl.constexpr, LOG2_WIDTH: tl.constexpr):
        """In-register butterfly over the column window [START, START + WIDTH).

        Columns outside the window are passed through untouched, which is what
        keeps rotary and pass-through subspaces contamination-free. Stride-h
        butterfly: x[i] <- x[i] + (-1)^(b & h) * x[i ^ h], O(WIDTH log WIDTH).
        """
        local = cols - START
        inside = (local >= 0) & (local < WIDTH)
        for stage in tl.static_range(LOG2_WIDTH):
            stride: tl.constexpr = 1 << stage
            partner = tl.where(inside, START + (local ^ stride), cols)
            gathered = tl.gather(x, tl.broadcast_to(partner[None, :], x.shape), 1)
            signed = tl.where((local & stride) == 0, gathered, -gathered)
            x = tl.where(inside[None, :], x + signed, x)
        return x

    @triton.jit
    def _triton_fwht_rotate_kernel(
        X,  # (rows, HEAD_DIM) fp16, contiguous
        SIGNS,  # (HEAD_DIM,) fp32 Rademacher signs
        OUT,  # (rows, HEAD_DIM) fp16
        ROWS,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        LOG2_ROTARY: tl.constexpr,
        LOG2_PASS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Rotate BLOCK_ROWS Q or K vectors into the Hadamard basis."""
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, HEAD_DIM)
        mask = rows[:, None] < ROWS
        x = tl.load(X + rows[:, None] * HEAD_DIM + cols[None, :], mask=mask, other=0.0)
        x = x.to(tl.float32) * tl.load(SIGNS + cols)[None, :]

        x = _fwht_region(x, cols, 0, ROTARY_DIM, LOG2_ROTARY)
        rot_scale: tl.constexpr = 1.0 / tl.sqrt(float(ROTARY_DIM))
        x = tl.where(cols < ROTARY_DIM, x * rot_scale, x)

        if LOG2_PASS > 0:
            x = _fwht_region(x, cols, ROTARY_DIM, HEAD_DIM - ROTARY_DIM, LOG2_PASS)
            pass_scale: tl.constexpr = 1.0 / tl.sqrt(float(HEAD_DIM - ROTARY_DIM))
            x = tl.where(cols >= ROTARY_DIM, x * pass_scale, x)

        tl.store(
            OUT + rows[:, None] * HEAD_DIM + cols[None, :],
            x.to(OUT.dtype.element_ty),
            mask=mask,
        )


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _signs_for(head_dim: int, rotary_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    """Concatenated Rademacher signs: seed for the rotary block, seed+1 for pass."""
    rot = _rademacher_signs(rotary_dim, seed)
    if rotary_dim == head_dim:
        return rot.to(device)
    return torch.cat([rot, _rademacher_signs(head_dim - rotary_dim, seed + 1)]).to(device)


def rotate_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    scope: str = "block_diag",
    rotary_dim: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K into the Hadamard basis (GPU Triton, CPU PyTorch).

    Applies the same orthogonal map to both operands, so attention logits are
    unchanged while channel kurtosis collapses toward the Gaussian value.

    Args:
        q: Query tensor of shape (..., head_dim).
        k: Key tensor of shape (..., head_dim).
        scope: ``"full"`` or ``"block_diag"`` (see ``fast_walsh_hadamard``).
        rotary_dim: Rotary width for ``"block_diag"``; defaults to full width.
        seed: Rademacher seed shared with the dequantization-side reference.

    Returns:
        ``(q_rotated, k_rotated)`` with the input shapes and dtypes.

    Raises:
        ValueError: If Q and K disagree on head width, or an unsupported scope
            is requested.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k head_dim mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    head_dim = q.shape[-1]
    rot = head_dim if (rotary_dim is None or scope == "full") else rotary_dim
    pass_dim = head_dim - rot

    gpu_ok = (
        GPU_KERNEL_AVAILABLE
        and q.is_cuda
        and k.is_cuda
        and q.dtype == k.dtype
        and q.is_contiguous()
        and k.is_contiguous()
        and _is_pow2(rot)
        and (pass_dim == 0 or _is_pow2(pass_dim))
    )
    if not gpu_ok:
        # PyTorch reference path (also the parity oracle for the GPU kernel).
        return (
            fast_walsh_hadamard(q, scope=scope, rotary_dim=rotary_dim, seed=seed),
            fast_walsh_hadamard(k, scope=scope, rotary_dim=rotary_dim, seed=seed),
        )

    signs = _signs_for(head_dim, rot, seed, q.device)
    out = []
    for x in (q, k):
        flat = x.reshape(-1, head_dim)
        dst = torch.empty_like(flat)
        rows = flat.shape[0]
        block_rows = 64
        if triton is None:  # pragma: no cover - unreachable when GPU_KERNEL_AVAILABLE
            raise RuntimeError("triton is unavailable; GPU_KERNEL_AVAILABLE must be False")
        _triton_fwht_rotate_kernel[(triton.cdiv(rows, block_rows),)](
            flat,
            signs,
            dst,
            rows,
            HEAD_DIM=head_dim,
            ROTARY_DIM=rot,
            LOG2_ROTARY=rot.bit_length() - 1,
            LOG2_PASS=pass_dim.bit_length() - 1 if pass_dim else 0,
            BLOCK_ROWS=block_rows,
        )
        out.append(dst.reshape(x.shape))
    return out[0], out[1]
