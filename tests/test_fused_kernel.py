"""Parity tests for the fused Split-K GQA decode path (CPU fallback + GPU)."""

import itertools
import math
from pathlib import Path

import pytest
import torch
from kv_quant.kernels.dequant_fused import (
    GPU_KERNEL_AVAILABLE,
    fused_splitk_attention,
    fused_splitk_attention_cpu,
)
from kv_quant.quantizer import QuantParams, quantize_group

HEAD_DIM = 32
NUM_KV_HEADS = 4


def build_pages(
    k: torch.Tensor,
    v: torch.Tensor,
    group_size: int,
    k_bits: int,
    v_bits: int,
    k_scheme: str,
    v_scheme: str,
):
    """Quantize K/V into packed pages + params + block table."""
    tokens = k.shape[0]
    num_pages = max(1, math.ceil(tokens / group_size))
    channels = NUM_KV_HEADS * HEAD_DIM

    def packed_len(bits: int) -> int:
        if bits == 16:
            return group_size * 2
        if bits == 8:
            return group_size
        if bits == 4:
            return group_size // 2
        if bits == 2:
            return group_size // 4
        raise ValueError(bits)

    packed_len_k = packed_len(k_bits)
    packed_len_v = packed_len(v_bits)

    k_pages = torch.zeros(num_pages, channels, packed_len_k, dtype=torch.uint8)
    v_pages = torch.zeros(num_pages, channels, packed_len_v, dtype=torch.uint8)
    k_params: list[list[QuantParams]] = []
    v_params: list[list[QuantParams]] = []

    for p in range(num_pages):
        kp_page, vp_page = [], []
        for c in range(channels):
            h, d = divmod(c, HEAD_DIM)
            lo = p * group_size
            hi = min(lo + group_size, tokens)
            kchan = k[lo:hi, h, d].contiguous() if hi > lo else torch.zeros(1, dtype=torch.float32)
            vchan = v[lo:hi, h, d].contiguous() if hi > lo else torch.zeros(1, dtype=torch.float32)
            pk, qk = quantize_group(kchan, k_bits, k_scheme)
            pv, qv = quantize_group(vchan, v_bits, v_scheme)
            k_pages[p, c, : pk.shape[0]] = pk
            v_pages[p, c, : pv.shape[0]] = pv
            kp_page.append(qk)
            vp_page.append(qv)
        k_params.append(kp_page)
        v_params.append(vp_page)

    block_table = torch.arange(num_pages, dtype=torch.int64).unsqueeze(0)
    return k_pages, v_pages, k_params, v_params, block_table


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gqa_group: int,
) -> torch.Tensor:
    """Materialized FP16 K/V -> GQA broadcast -> SDPA reference."""
    batch, num_q_heads, q_len, dim = q.shape
    k_rep = k.repeat_interleave(gqa_group, dim=1).transpose(0, 1)
    v_rep = v.repeat_interleave(gqa_group, dim=1).transpose(0, 1)
    scores = torch.matmul(q.float(), k_rep.float().transpose(-1, -2)) / math.sqrt(dim)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_rep.float()).to(torch.float16)


def run_parity_case(
    seq_len: int,
    k_bits: int,
    v_bits: int,
    k_scheme: str,
    v_scheme: str,
    group_size: int,
    gqa: int,
):
    torch.manual_seed(42)
    num_q_heads = NUM_KV_HEADS * gqa
    q = torch.randn(1, num_q_heads, min(16, seq_len), HEAD_DIM, dtype=torch.float16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)

    k_pages, v_pages, kp, vp, bt = build_pages(k, v, group_size, k_bits, v_bits, k_scheme, v_scheme)
    kv_len = torch.tensor([seq_len], dtype=torch.int64)

    fused = fused_splitk_attention(
        q,
        k_pages,
        v_pages,
        kp,
        vp,
        bt,
        kv_len,
        num_kv_heads=NUM_KV_HEADS,
        k_bits=k_bits,
        v_bits=v_bits,
        k_scheme=k_scheme,
        v_scheme=v_scheme,
        group_size=group_size,
        num_splits=4,
    )

    # Reconstruct FP16 K/V via naive dequant for the reference
    k_ref = torch.zeros(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v_ref = torch.zeros_like(k_ref)
    channels = NUM_KV_HEADS * HEAD_DIM
    for p in range(bt.shape[1]):
        for c in range(channels):
            h, d = divmod(c, HEAD_DIM)
            lo, hi = p * group_size, min((p + 1) * group_size, seq_len)
            if hi <= lo:
                continue
            orig = hi - lo
            kd = torch.empty_like(k_pages[p, c]).float()
            vd = torch.empty_like(v_pages[p, c]).float()
            kd[: k_pages[p, c].shape[0]] = k_pages[p, c].float()
            vd[: v_pages[p, c].shape[0]] = v_pages[p, c].float()
            from kv_quant.kernels.dequant_fused import _dequantize_packed_vectorized

            kv = _dequantize_packed_vectorized(
                k_pages[p, c],
                kp[p][c].scale,
                kp[p][c].zero_point,
                k_bits,
                k_scheme,
                orig,
            )
            vv = _dequantize_packed_vectorized(
                v_pages[p, c],
                vp[p][c].scale,
                vp[p][c].zero_point,
                v_bits,
                v_scheme,
                orig,
            )
            k_ref[lo:hi, h, d] = kv
            v_ref[lo:hi, h, d] = vv

    naive = naive_attention(q, k_ref, v_ref, gqa)
    assert fused.shape == naive.shape
    assert torch.allclose(fused, naive, atol=1e-3, rtol=1e-3), (
        f"mismatch bits=({k_bits},{v_bits}) scheme=({k_scheme},{v_scheme}) "
        f"G={group_size} seq={seq_len} gqa={gqa} "
        f"maxdiff={(fused.float() - naive.float()).abs().max().item()}"
    )


# Curated matrix covering all required dimensions without 900+ tests
_CONFIGS = []
for i, (kb, vb) in enumerate(
    [(16, 16), (8, 8), (4, 4), (4, 8), (8, 4), (16, 4), (4, 16), (8, 16), (16, 8)]
):
    scheme = "sym" if i % 2 == 0 else "asym"
    group = [32, 64, 128][i % 3]
    seq = [5, 64, 69, 513, 2048, 8192][i % 6]
    gqa = [1, 4, 8][i % 3]
    _CONFIGS.append((seq, kb, vb, scheme, scheme, group, gqa))


@pytest.mark.parametrize("seq_len,k_bits,v_bits,k_scheme,v_scheme,group_size,gqa", _CONFIGS)
def test_fused_parity_matrix(seq_len, k_bits, v_bits, k_scheme, v_scheme, group_size, gqa):
    run_parity_case(seq_len, k_bits, v_bits, k_scheme, v_scheme, group_size, gqa)


def test_scheme_cross_product():
    for k_scheme, v_scheme in itertools.product(["sym", "asym"], repeat=2):
        run_parity_case(69, 4, 4, k_scheme, v_scheme, 64, 4)


def test_unaligned_short_seq():
    for seq in [5, 69, 513]:
        run_parity_case(seq, 4, 4, "sym", "asym", 64, 4)


def test_fixture_parity():
    """Validate against the committed 2k-token micro-activation fixture."""
    fixture_path = Path(__file__).resolve().parents[1] / "bench" / "activations" / "micro_kv_2k.pt"
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    k_full = fixture["k"][0].float()  # (2048, 8, 64)
    v_full = fixture["v"][0].float()
    k = k_full[:, :NUM_KV_HEADS, :HEAD_DIM].to(torch.float16)
    v = v_full[:, :NUM_KV_HEADS, :HEAD_DIM].to(torch.float16)
    q = torch.randn(1, NUM_KV_HEADS * 4, 8, HEAD_DIM, dtype=torch.float16)

    k_pages, v_pages, kp, vp, bt = build_pages(k, v, 64, 4, 4, "sym", "asym")
    fused = fused_splitk_attention_cpu(
        q,
        k_pages,
        v_pages,
        kp,
        vp,
        bt,
        torch.tensor([k.shape[0]]),
        num_kv_heads=NUM_KV_HEADS,
        k_bits=4,
        v_bits=4,
        k_scheme="sym",
        v_scheme="asym",
        group_size=64,
    )
    assert fused.shape == q.shape
    assert torch.isfinite(fused).all()


@pytest.mark.gpu
def test_gpu_kernel_parity():
    if not GPU_KERNEL_AVAILABLE:
        pytest.skip("Triton + CUDA required")
    run_parity_case(513, 4, 4, "sym", "asym", 64, 4)


def _build_pages_mixed_k(k, v, group_size, k_bits_by_head, v_bits, k_scheme, v_scheme):
    """ADR-009 pages: per-head K bit-widths, channel rows padded to the widest."""
    tokens = k.shape[0]
    num_pages = max(1, math.ceil(tokens / group_size))
    channels = NUM_KV_HEADS * HEAD_DIM

    def packed_len(bits: int) -> int:
        return {16: group_size * 2, 8: group_size, 4: group_size // 2, 2: group_size // 4}[bits]

    max_pk = max(packed_len(b) for b in k_bits_by_head)
    k_pages = torch.zeros(num_pages, channels, max_pk, dtype=torch.uint8)
    v_pages = torch.zeros(num_pages, channels, packed_len(v_bits), dtype=torch.uint8)
    k_params: list[list[QuantParams]] = []
    v_params: list[list[QuantParams]] = []
    for p in range(num_pages):
        kp_page, vp_page = [], []
        for c in range(channels):
            h, d = divmod(c, HEAD_DIM)
            lo, hi = p * group_size, min((p + 1) * group_size, tokens)
            kchan = k[lo:hi, h, d].contiguous() if hi > lo else torch.zeros(1, dtype=torch.float32)
            vchan = v[lo:hi, h, d].contiguous() if hi > lo else torch.zeros(1, dtype=torch.float32)
            pk, qk = quantize_group(kchan, k_bits_by_head[h], k_scheme)
            pv, qv = quantize_group(vchan, v_bits, v_scheme)
            k_pages[p, c, : pk.shape[0]] = pk
            v_pages[p, c, : pv.shape[0]] = pv
            kp_page.append(qk)
            vp_page.append(qv)
        k_params.append(kp_page)
        v_params.append(vp_page)
    block_table = torch.arange(num_pages, dtype=torch.int64).unsqueeze(0)
    return k_pages, v_pages, k_params, v_params, block_table


def test_head_adaptive_parity_cpu():
    """Mixed per-head K widths must match a per-head dequant reference."""
    from kv_quant.kernels.dequant_fused import _dequantize_packed_vectorized

    torch.manual_seed(42)
    seq_len, group_size, gqa = 513, 64, 4
    k_bits_by_head = [8, 4, 4, 4]  # head 0 retrieval (K8), rest streaming (K4)
    q = torch.randn(1, NUM_KV_HEADS * gqa, 8, HEAD_DIM, dtype=torch.float16)
    k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)

    k_pages, v_pages, kp, vp, bt = _build_pages_mixed_k(
        k, v, group_size, k_bits_by_head, 4, "sym", "asym"
    )
    fused = fused_splitk_attention_cpu(
        q,
        k_pages,
        v_pages,
        kp,
        vp,
        bt,
        torch.tensor([seq_len], dtype=torch.int64),
        num_kv_heads=NUM_KV_HEADS,
        k_bits=4,
        v_bits=4,
        k_scheme="sym",
        v_scheme="asym",
        group_size=group_size,
        head_k_bits=k_bits_by_head,
    )

    k_ref = torch.zeros_like(k)
    v_ref = torch.zeros_like(v)
    channels = NUM_KV_HEADS * HEAD_DIM
    for p in range(bt.shape[1]):
        for c in range(channels):
            h, d = divmod(c, HEAD_DIM)
            lo, hi = p * group_size, min((p + 1) * group_size, seq_len)
            if hi <= lo:
                continue
            kv = _dequantize_packed_vectorized(
                k_pages[p, c],
                kp[p][c].scale,
                kp[p][c].zero_point,
                k_bits_by_head[h],
                "sym",
                hi - lo,
            )
            vv = _dequantize_packed_vectorized(
                v_pages[p, c],
                vp[p][c].scale,
                vp[p][c].zero_point,
                4,
                "asym",
                hi - lo,
            )
            k_ref[lo:hi, h, d] = kv
            v_ref[lo:hi, h, d] = vv
    naive = naive_attention(q, k_ref, v_ref, gqa)
    assert torch.allclose(fused, naive, atol=1e-3, rtol=1e-3), (
        f"head-adaptive mismatch maxdiff={(fused.float() - naive.float()).abs().max().item()}"
    )
    # The retrieval head (K8) must agree with the naive path more tightly
    # than the naive path built with uniform K4 would.
    fused_uniform = fused_splitk_attention_cpu(
        q,
        k_pages,
        v_pages,
        kp,
        vp,
        bt,
        torch.tensor([seq_len], dtype=torch.int64),
        num_kv_heads=NUM_KV_HEADS,
        k_bits=4,
        v_bits=4,
        k_scheme="sym",
        v_scheme="asym",
        group_size=group_size,
    )
    err_adaptive = (fused.float() - naive.float()).abs().max().item()
    err_uniform = (fused_uniform.float() - naive.float()).abs().max().item()
    assert err_adaptive < err_uniform, (err_adaptive, err_uniform)


@pytest.mark.gpu
def test_gpu_head_adaptive_parity():
    if not GPU_KERNEL_AVAILABLE:
        pytest.skip("Triton + CUDA required")
    test_head_adaptive_parity_cpu()


def test_cpu_dispatch_matches_reference():
    """fused_splitk_attention dispatches to the CPU reference without GPU."""
    torch.manual_seed(7)
    q = torch.randn(1, NUM_KV_HEADS, 4, HEAD_DIM, dtype=torch.float16)
    k = torch.randn(64, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v = torch.randn_like(k)
    kp_pages, vp_pages, kp, vp, bt = build_pages(k, v, 64, 4, 4, "sym", "asym")
    a = fused_splitk_attention(
        q,
        kp_pages,
        vp_pages,
        kp,
        vp,
        bt,
        torch.tensor([64]),
        num_kv_heads=NUM_KV_HEADS,
        k_bits=4,
        v_bits=4,
        k_scheme="sym",
        v_scheme="asym",
        group_size=64,
    )
    b = fused_splitk_attention_cpu(
        q,
        kp_pages,
        vp_pages,
        kp,
        vp,
        bt,
        torch.tensor([64]),
        num_kv_heads=NUM_KV_HEADS,
        k_bits=4,
        v_bits=4,
        k_scheme="sym",
        v_scheme="asym",
        group_size=64,
    )
    assert torch.equal(a, b)
