# Adversarial Review: Fused Split-K GQA Triton Kernel

**Scope**: `fused-kernel`
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-16
**Branch**: `feat/fused-splitk-triton-kernel`
**Target Tag**: `v0.3.0-fused-kernel`

---

## Findings

### Finding 1: INT4 packed-length math divided by zero for 16-bit
- [x] **Finding**: `packed_len = group_size // (8 // bits)` raised `ZeroDivisionError` for `bits=16`.
- **Resolution**: RESOLVED — Added an explicit `packed_len()` helper handling 16/8/4/2 bits. All 9 bit-width combinations in the matrix now pass.

### Finding 2: Non-contiguous channel slices broke 16-bit view
- [x] **Finding**: `k[lo:hi, h, d]` has stride `head_dim`, so `view(torch.uint8)` failed for 16-bit.
- **Resolution**: RESOLVED — Added `.contiguous()` on channel slices in `build_pages`.

### Finding 3: Triton import on CPU-only CI
- [x] **Finding**: Importing `triton` at module scope could break environments without the wheel.
- **Resolution**: RESOLVED — Wrapped Triton import in `try/except ImportError`, exposing `HAS_TRITON` / `GPU_KERNEL_AVAILABLE` flags. `fused_splitk_attention` dispatches to the CPU reference path when either is false.

### Finding 4: Fused CPU path and naive reference must match to 1e-3
- [x] **Finding**: A fused path that silently diverges from naive dequantization would defeat Invariant 6.
- **Resolution**: RESOLVED — `run_parity_case` reconstructs FP16 K/V via the same vectorized dequantizer, applies GQA broadcast, and asserts `torch.allclose(..., atol=1e-3, rtol=1e-3)` across bits, schemes, group sizes, unaligned lengths, and GQA ratios. 13 CPU cases pass; 1 GPU test skips on CPU-only CI.

### Finding 5: 8.11× HBM derivation must be reproducible
- [x] **Finding**: A claimed traffic reduction without a committed derivation would violate Invariant 1.
- **Resolution**: RESOLVED — `bench/bandwidth.py --bits 4 --group-size 64` emits `hbm_traffic_reduction: 8.111`; the derivation is documented in `docs/KERNEL_NOTES.md` and committed in `bench/data/p3_hbm_traffic_model.csv` with a manifest.

### Finding 6: FP8 tensor-core support misattributed to Ampere
- [x] **Finding**: Claiming FP8 MMA on sm80 would overstate compatibility.
- **Resolution**: RESOLVED — `arch_compat_matrix.csv` marks `fp8_e4m3_tensorcore` as unsupported on sm80/sm86 and supported on sm89+/sm90/sm100, with a note explaining the sm89+ requirement.

### Finding 7: Legacy `dequant_fused` parity API must remain intact
- [x] **Finding**: Rewriting `dequant_fused.py` could break `tests/test_parity.py`, which imports `dequant_fused`.
- **Resolution**: RESOLVED — The legacy `dequant_fused(packed_page, params_list, layout)` signature is retained and delegates to the validated reference path; `tests/test_parity.py` and all prior gates still pass.

---

## Summary

All 7 findings resolved. The fused Split-K GQA kernel module (with Triton kernels and CPU reference), comprehensive parity tests, bandwidth probe, two committed CSVs with manifests, and `docs/KERNEL_NOTES.md` are in place. All 8 local gates pass.

**Status**: APPROVED — ready for merge
