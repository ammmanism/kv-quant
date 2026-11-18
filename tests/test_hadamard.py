"""Property tests for orthogonal Walsh-Hadamard Q/K rotation (ADR-008)."""

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from kv_quant.config import QuantConfig
from kv_quant.hadamard import (
    fast_walsh_hadamard,
    hadamard_matrix,
    hadamard_transform,
    inverse_fast_walsh_hadamard,
)

SCOPES = [("full", None), ("block_diag", 32), ("block_diag", 64), ("block_diag", 128)]
HEAD_DIM = 128

vector = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=HEAD_DIM,
    max_size=HEAD_DIM,
).map(lambda v: torch.tensor(v, dtype=torch.float32))


def _as_batch(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(1, HEAD_DIM)


@given(q=vector, k=vector)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_inner_product_invariance_full(q: torch.Tensor, k: torch.Tensor) -> None:
    """<Q R, K R> == <Q, K> exactly for the full rotation (float32 rounding only)."""
    qp = fast_walsh_hadamard(_as_batch(q), scope="full")
    kp = fast_walsh_hadamard(_as_batch(k), scope="full")
    orig = (q * k).sum().item()
    rot = (qp * kp).sum().item()
    scale = max(abs(orig), 1.0)
    assert abs(rot - orig) <= 1e-4 * scale


@pytest.mark.parametrize("rotary_dim", [32, 64, 128])
@given(q=vector, k=vector)
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_inner_product_invariance_block_diag(
    rotary_dim: int, q: torch.Tensor, k: torch.Tensor
) -> None:
    """<Q R, K R> == <Q, K> for block-diagonal rotation on partial-RoPE widths."""
    qp = fast_walsh_hadamard(_as_batch(q), scope="block_diag", rotary_dim=rotary_dim)
    kp = fast_walsh_hadamard(_as_batch(k), scope="block_diag", rotary_dim=rotary_dim)
    orig = (q * k).sum().item()
    rot = (qp * kp).sum().item()
    scale = max(abs(orig), 1.0)
    assert abs(rot - orig) <= 1e-4 * scale


@pytest.mark.parametrize("scope,rotary_dim", SCOPES)
def test_rotation_matrix_is_orthogonal(scope: str, rotary_dim: int | None) -> None:
    r = hadamard_matrix(HEAD_DIM, scope, rotary_dim)
    eye = torch.eye(HEAD_DIM)
    assert torch.allclose(r.T @ r, eye, atol=1e-5)


@pytest.mark.parametrize("rotary_dim", [32, 64])
def test_block_diag_zero_subspace_contamination(rotary_dim: int) -> None:
    """A vector supported only on the pass-through subspace must not leak into
    the rotary block (and vice versa)."""
    x = torch.zeros(1, HEAD_DIM)
    x[0, rotary_dim:] = torch.randn(HEAD_DIM - rotary_dim)
    y = fast_walsh_hadamard(x, scope="block_diag", rotary_dim=rotary_dim)
    assert torch.count_nonzero(y[0, :rotary_dim]) == 0

    z = torch.zeros(1, HEAD_DIM)
    z[0, :rotary_dim] = torch.randn(rotary_dim)
    w = fast_walsh_hadamard(z, scope="block_diag", rotary_dim=rotary_dim)
    assert torch.count_nonzero(w[0, rotary_dim:]) == 0


def test_block_diag_with_full_rotary_matches_full_scope() -> None:
    x = torch.randn(4, HEAD_DIM)
    full = fast_walsh_hadamard(x, scope="full", seed=7)
    block = fast_walsh_hadamard(x, scope="block_diag", rotary_dim=HEAD_DIM, seed=7)
    assert torch.equal(full, block)


def test_rotation_is_seeded_and_pure() -> None:
    x = torch.randn(2, HEAD_DIM)
    snapshot = x.clone()
    a = fast_walsh_hadamard(x, scope="full", seed=42)
    b = fast_walsh_hadamard(x, scope="full", seed=42)
    c = fast_walsh_hadamard(x, scope="full", seed=43)
    assert torch.equal(x, snapshot), "input must not be mutated (pure functional)"
    assert torch.equal(a, b), "same seed must give identical rotation"
    assert not torch.equal(a, c), "different seed must give a different rotation"


def test_rotation_inverse_roundtrip() -> None:
    x = torch.randn(8, HEAD_DIM)
    for scope, rotary_dim in [("full", None), ("block_diag", 64), ("block_diag", 32)]:
        y = fast_walsh_hadamard(x, scope=scope, rotary_dim=rotary_dim)
        z = inverse_fast_walsh_hadamard(y, scope=scope, rotary_dim=rotary_dim)
        assert torch.allclose(z, x, atol=1e-4), (scope, rotary_dim)


def test_rotation_equals_materialized_matrix() -> None:
    x = torch.randn(2, HEAD_DIM)
    r = hadamard_matrix(HEAD_DIM, "block_diag", 64, seed=11)
    assert torch.allclose(fast_walsh_hadamard(x, "block_diag", 64, seed=11), x @ r, atol=1e-5)


def test_invalid_scope_and_widths_raise() -> None:
    x = torch.randn(1, HEAD_DIM)
    with pytest.raises(ValueError):
        fast_walsh_hadamard(x, scope="bogus")
    with pytest.raises(ValueError):
        fast_walsh_hadamard(x, scope="block_diag", rotary_dim=100)  # not power of 2
    with pytest.raises(ValueError):
        fast_walsh_hadamard(x, scope="block_diag", rotary_dim=0)
    with pytest.raises(ValueError):
        fast_walsh_hadamard(torch.randn(1, 100), scope="full")  # not power of 2


def test_block_diag_non_pow2_passthrough_is_identity() -> None:
    """D=40, rotary_dim=16 -> pass width 24 (not a power of 2) is left as I_24."""
    x = torch.zeros(1, 40)
    x[0, 16:] = torch.arange(24, dtype=torch.float32)
    y = fast_walsh_hadamard(x, scope="block_diag", rotary_dim=16)
    assert torch.equal(y[0, 16:], x[0, 16:]), "non-power-of-2 pass block must be identity"
    assert torch.count_nonzero(y[0, :16]) == 0


class TestResolveHadamardScope:
    def test_disabled_is_none(self) -> None:
        assert QuantConfig().resolve_hadamard_scope(128, 128) == "none"

    def test_partial_rope_resolves_block_diag(self) -> None:
        cfg = QuantConfig(use_hadamard=True)
        assert cfg.resolve_hadamard_scope(128, 64) == "block_diag"

    def test_full_rope_resolves_full(self) -> None:
        cfg = QuantConfig(use_hadamard=True, rotary_dim=128)
        assert cfg.resolve_hadamard_scope(128) == "full"

    def test_explicit_scope_wins(self) -> None:
        cfg = QuantConfig(use_hadamard=True, hadamard_scope="full")
        assert cfg.resolve_hadamard_scope(128, 64) == "full"
        cfg = QuantConfig(use_hadamard=True, hadamard_scope="block_diag")
        assert cfg.resolve_hadamard_scope(128, 128) == "block_diag"

    def test_config_rotary_dim_fallback(self) -> None:
        cfg = QuantConfig(use_hadamard=True, rotary_dim=32)
        assert cfg.resolve_hadamard_scope(128) == "block_diag"
        # explicit argument takes precedence over the config field
        assert cfg.resolve_hadamard_scope(128, 128) == "full"

    def test_unknown_rotary_dim_raises(self) -> None:
        cfg = QuantConfig(use_hadamard=True)
        with pytest.raises(ValueError):
            cfg.resolve_hadamard_scope(128)

    def test_out_of_range_rotary_dim_raises(self) -> None:
        cfg = QuantConfig(use_hadamard=True)
        with pytest.raises(ValueError):
            cfg.resolve_hadamard_scope(128, 256)


def test_fast_walsh_matches_legacy_transform_with_signs() -> None:
    """The rotation is exactly the committed Phase-1 butterfly with seeded signs."""
    from kv_quant.hadamard import _rademacher_signs

    x = torch.randn(3, HEAD_DIM)
    signs = _rademacher_signs(HEAD_DIM, seed=1).reshape(1, HEAD_DIM)
    assert torch.allclose(
        fast_walsh_hadamard(x, "full", seed=1),
        hadamard_transform(x * signs),
        atol=1e-5,
    )


class TestRotateQkKernelDispatch:
    """The kernel wrapper must agree with the PyTorch reference on every path."""

    def test_cpu_fallback_matches_reference(self) -> None:
        from kv_quant.kernels.triton_hadamard import rotate_qk

        q = torch.randn(4, 8, HEAD_DIM)
        k = torch.randn(4, 8, HEAD_DIM)
        for scope, rot in [("full", None), ("block_diag", 32), ("block_diag", 64)]:
            qk, kk = rotate_qk(q, k, scope=scope, rotary_dim=rot, seed=5)
            assert torch.allclose(qk, fast_walsh_hadamard(q, scope, rot, 5))
            assert torch.allclose(kk, fast_walsh_hadamard(k, scope, rot, 5))

    def test_head_dim_mismatch_raises(self) -> None:
        from kv_quant.kernels.triton_hadamard import rotate_qk

        with pytest.raises(ValueError):
            rotate_qk(torch.randn(2, 64), torch.randn(2, 128))

    @pytest.mark.gpu
    def test_gpu_triton_rotation_matches_reference(self) -> None:
        """Parity gate for `_triton_fwht_rotate_kernel` (CUDA hosts only).

        Unverified until this runs on a GPU; see ADR-006/ADR-008.
        """
        from kv_quant.kernels.triton_hadamard import GPU_KERNEL_AVAILABLE, rotate_qk

        if not GPU_KERNEL_AVAILABLE:
            pytest.skip("Triton + CUDA required")
        q = torch.randn(32, 8, HEAD_DIM, device="cuda", dtype=torch.float16)
        k = torch.randn(32, 8, HEAD_DIM, device="cuda", dtype=torch.float16)
        for scope, rot in [("full", None), ("block_diag", 32), ("block_diag", 64)]:
            qk, kk = rotate_qk(q, k, scope=scope, rotary_dim=rot, seed=5)
            assert torch.allclose(qk.float(), fast_walsh_hadamard(q, scope, rot, 5), atol=2e-3)
            assert torch.allclose(kk.float(), fast_walsh_hadamard(k, scope, rot, 5), atol=2e-3)
            # inner-product invariance holds on the device path too
            ref = (q.float() * k.float()).sum(-1)
            got = (qk.float() * kk.float()).sum(-1)
            assert torch.allclose(got, ref, atol=1e-2)
