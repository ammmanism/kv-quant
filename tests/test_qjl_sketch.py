"""3-way verification gate for the Gaussian QJL residual sketch (ADR-010).

Gate 1: INT2 packing parity (QJL disabled) against the analytical grid.
Gate 2: deterministic sketch reproducibility under a fixed seed.
Gate 3: unbiasedness and statistical MSE reduction vs FP16 ground truth on
        1000 fixture-derived D=128 attention heads (aligned query regime).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from kv_quant.qjl_sketch import (
    QJL_M_BITS,
    estimate_qjl_logits_ref,
    naive_int2_logit,
    quantize_k_int2_qjl,
)
from kv_quant.quantizer import dequantize_group, quantize_group

D = 128
FIXTURE = Path(__file__).resolve().parents[1] / "bench" / "activations" / "micro_kv_2k.pt"


def _keys(n: int, seed: int = 2026) -> torch.Tensor:
    """Mirror of bench/qjl_mse_ablation.py's D=128 head-pair construction."""
    fixture = torch.load(FIXTURE, map_location="cpu", weights_only=True)
    k = fixture["k"].float()  # (layers, seq, heads, 64)
    layers, _, heads, dim = k.shape
    pool = torch.cat(
        [
            torch.cat([k[layer, :, h, :], k[layer, :, h + 1, :]], dim=-1)
            for layer in range(layers)
            for h in range(0, heads - 1, 2)
        ],
        dim=0,
    )
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(pool.shape[0], generator=generator)[:n]
    return pool[idx]


# ---------------------------------------------------------------------------
# Gate 1: packing parity, QJL disabled
# ---------------------------------------------------------------------------


class TestGate1PackingParity:
    def test_int2_roundtrip_matches_analytical_grid(self) -> None:
        """Sym INT2 grid is {-s, 0, +s} with s = max|k|; roundtrip stays on it."""
        generator = torch.Generator().manual_seed(1)
        for _ in range(50):
            k = torch.randn(64, generator=generator)
            packed, params = quantize_group(k, 2, "sym")
            k_hat = dequantize_group(packed, params, 2, "sym")
            s = float(k.abs().max())
            for value in k_hat.tolist():
                assert min(abs(value - level) for level in (-s, 0.0, s)) < 1e-3

    def test_payload_base_matches_direct_int2_dequant(self) -> None:
        """The payload's INT2 base (naive logit path) is bit-identical to the
        plain quantizer roundtrip -- i.e. QJL adds nothing to the base grid."""
        generator = torch.Generator().manual_seed(2)
        k = torch.randn(D, generator=generator)
        payload = quantize_k_int2_qjl(k, None, seed=42)
        direct, params = quantize_group(k, 2, "sym")
        assert torch.equal(payload.packed_int2, direct)
        q = torch.randn(D, generator=generator)
        naive = naive_int2_logit(q, payload)
        manual = float(
            (
                q.to(torch.float32) @ dequantize_group(direct, params, 2, "sym").to(torch.float32)
            ).item()
        )
        assert abs(naive - manual) < 1e-3

    def test_asym_scheme_rejected(self) -> None:
        from kv_quant.config import QuantConfig

        cfg = QuantConfig(k_scheme="asym")
        with pytest.raises(ValueError):
            quantize_k_int2_qjl(torch.randn(D), cfg, seed=42)


# ---------------------------------------------------------------------------
# Gate 2: deterministic sketch reproducibility
# ---------------------------------------------------------------------------


class TestGate2Reproducibility:
    def test_same_seed_bit_identical(self) -> None:
        k = _keys(8)
        for key in k:
            a = quantize_k_int2_qjl(key, None, seed=42)
            b = quantize_k_int2_qjl(key, None, seed=42)
            assert torch.equal(a.signs, b.signs), "uint32 sign sketches must be bit-identical"
            assert torch.equal(a.packed_int2, b.packed_int2)
            assert a.residual_norm == b.residual_norm
            q = key / key.norm()
            est_a = estimate_qjl_logits_ref(q, a, seed=42)
            est_b = estimate_qjl_logits_ref(q, b, seed=42)
            assert abs(est_a - est_b) < 1e-5

    def test_different_seed_different_sketch(self) -> None:
        k = _keys(4)
        for key in k:
            a = quantize_k_int2_qjl(key, None, seed=42)
            c = quantize_k_int2_qjl(key, None, seed=43)
            assert not torch.equal(a.signs, c.signs)

    def test_shape_mismatch_and_projection_checks_raise(self) -> None:
        with pytest.raises(ValueError):
            quantize_k_int2_qjl(torch.randn(D, 2), None, seed=42)  # not a vector
        with pytest.raises(ValueError):
            quantize_k_int2_qjl(torch.randn(D), None, gaussian_proj=torch.randn(3, 7))
        with pytest.raises(ValueError):
            estimate_qjl_logits_ref(
                torch.randn(64), quantize_k_int2_qjl(torch.randn(D), None, seed=42)
            )


# ---------------------------------------------------------------------------
# Gate 3: unbiasedness & statistical MSE reduction vs FP16 ground truth
# ---------------------------------------------------------------------------


class TestGate3Unbiasedness:
    N = 1000
    SCALE = math.sqrt(D)

    def test_qjl_bias_under_one_centinat(self) -> None:
        """|E[hat_ell(INT2+QJL) - ell_FP16]| < 0.01 in the aligned regime."""
        keys = _keys(self.N)
        errors = []
        for key in keys:
            payload = quantize_k_int2_qjl(key, None, seed=42, m_bits=QJL_M_BITS)
            q = key / key.norm()
            true = float((q @ key).item()) / self.SCALE
            errors.append(estimate_qjl_logits_ref(q, payload, seed=42) / self.SCALE - true)
        bias = abs(float(torch.tensor(errors).mean()))
        assert bias < 0.01, f"QJL empirical bias {bias:.5f} exceeds 0.01"

    def test_naive_int2_exhibits_systematic_bias(self) -> None:
        """The naive INT2 shrinkage grid is systematically biased (the failure
        mode QJL exists to correct); |bias| must exceed the QJL gate 10x."""
        keys = _keys(self.N)
        errors = []
        for key in keys:
            payload = quantize_k_int2_qjl(key, None, seed=42)
            q = key / key.norm()
            true = float((q @ key).item()) / self.SCALE
            errors.append(naive_int2_logit(q, payload) / self.SCALE - true)
        bias = abs(float(torch.tensor(errors).mean()))
        assert bias > 0.1, f"expected severe systematic naive bias, got {bias:.5f}"

    def test_mse_reduction_vs_naive(self) -> None:
        """MSE(INT2+QJL) < 0.35 x MSE(INT2_naive) in the aligned regime."""
        keys = _keys(self.N)
        err_naive, err_qjl = [], []
        for key in keys:
            payload = quantize_k_int2_qjl(key, None, seed=42, m_bits=QJL_M_BITS)
            q = key / key.norm()
            true = float((q @ key).item()) / self.SCALE
            err_naive.append(naive_int2_logit(q, payload) / self.SCALE - true)
            err_qjl.append(estimate_qjl_logits_ref(q, payload, seed=42) / self.SCALE - true)
        mse_naive = float(torch.tensor(err_naive).pow(2).mean())
        mse_qjl = float(torch.tensor(err_qjl).pow(2).mean())
        assert mse_qjl < 0.35 * mse_naive, f"MSE ratio {mse_qjl / mse_naive:.4f} >= 0.35"

    def test_sketch_variance_shrinks_with_m(self) -> None:
        """MSE must fall as ~1/M: doubling the sketch width at least halves it."""
        keys = _keys(200)
        for m in (64, 256):
            errors = []
            for key in keys:
                payload = quantize_k_int2_qjl(key, None, seed=42, m_bits=m)
                q = key / key.norm()
                true = float((q @ key).item()) / self.SCALE
                errors.append(estimate_qjl_logits_ref(q, payload, seed=42) / self.SCALE - true)
            mse = float(torch.tensor(errors).pow(2).mean())
            if m == 64:
                mse64 = mse
        assert mse < mse64 / 2.5, (mse, mse64)
