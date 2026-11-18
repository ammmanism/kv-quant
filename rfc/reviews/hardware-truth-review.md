# Adversarial Review: Hardware Truth (ncu, TTFT/ITL, VRAM cross-validation)

**Scope**: `hardware-truth`
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-18
**Branch**: `bench/hardware-roofline-and-vram`
**Target Tag**: `v0.7.0-hardware-truth`

---

## Findings

### Finding 1: Latent fp16 strided-view crash in the quantizer (severity: HIGH)
- [x] **Finding**: `quantize_group(x, bits=16)` calls `x.view(torch.uint8)`, which
  raises `RuntimeError: stride(-1) must be 1` on strided channel views. The
  VRAM cross-validation — the first code path to exercise the fp16 branch
  through `HybridPagedKVCache.append` — crashed on every `(16,16)` config.
  Phase-2 tests only ever exercised `bits in {4, 8}` through the cache.
- **Resolution**: Fixed with user approval (second core file for this task):
  `packed = x.contiguous().view(torch.uint8)`.
  Regression test: `tests/test_hardware_truth.py::test_quantize_group_bits16_strided_channel`.

### Finding 2: INT6 packing contradicts the layout contract (severity: MEDIUM)
- [x] **Finding**: `quantizer.py` stores INT6 as 1 byte/value ("wasteful but
  correct"), while `layout.bytes_per_element` claims 0.75 B/elem. The VRAM
  cross-validation exposed a +13.2% residual on 6-bit configs until the
  calculator modeled implemented truth.
- **Resolution**: Won't do (this PR) — re-packing INT6 as 4-values-per-3-bytes
  changes the packing/parity surface and belongs in a dedicated, isolated
  change. Documented in `docs/KERNEL_NOTES.md §4`; the calculator models the
  implementation so the comparison stays honest. Flagged as follow-up.

### Finding 3: p6 sweep VRAM estimates inherit the layout 0.75 B/elem for 6-bit
- [x] **Finding**: `p6_v1_frontier_sweep.csv:theoretical_vram_mib` uses
  `estimate_vram` (layout contract), so its 6-bit rows are ~25% optimistic
  vs. the implemented INT6 packing.
- **Resolution**: Won't do — p6 rows are explicitly *layout-contract* estimates
  and the sweep CSV is hash-pinned; re-derivation belongs with Finding 2's
  packing upgrade. The p7 cross-validation is the implementation-truth record.

### Finding 4: `get_tier_bytes` lives in `vram.py`, not on `HybridPagedKVCache`
- [x] **Finding**: The milestone text names `HybridPagedKVCache.get_tier_bytes()`.
- **Resolution**: Won't do — the one-core-file-per-task invariant forbids
  editing `cache.py` and `vram.py` in the same pass. Implemented as
  `kv_quant.vram.get_tier_bytes(cache)` using only public cache attributes;
  moving it into the class is a trivial follow-up.

### Finding 5: ncu counters are modeled, not profiled (severity: process)
- [x] **Finding**: No CUDA host / `ncu` binary is available; real
  `dram__bytes_read.sum` profiles are impossible here.
- **Resolution**: Won't do — every row carries `mode="ncu-modeled"`, the
  manifest records `gpu_name="cpu-only-host"`, and `docs/KERNEL_NOTES.md`
  states the upgrade path. Counters derive from the *exact* byte equations
  (naive matches theory to the byte) plus a documented L2 residency model.

### Finding 6: R²=1.0000 / MAPE=0.0000% looks fabricated
- [x] **Finding**: A perfect fit is suspicious and must not be asserted blindly.
- **Resolution**: Investigated — the fit is exact *by construction of honest
  accounting*: measured bytes come from real `HybridPagedKVCache` objects
  (shared pages once, per-channel byte packing, fp16 ring/sink), and the
  calculator independently re-derives the same equations from layout
  constants (`sealed-token payload`, `bits_to_bytes` incl. the INT6
  exception). The match is the *point* of cross-validation: it verifies page
  rounding, ring accounting, and packer behavior. Two residuals were found
  and fixed en route (calculator double-counted unsealed ring tokens as
  payload; INT6 byte width). `test_committed_crossval_regression_from_csv`
  recomputes R²/MAPE from the committed CSV and asserts the fit (R²>0.999,
  MAPE<1%), so the claim is re-derivable from the artifact, not the narrative.
  The manifest records the method (`cache_object_bytes`, CPU) honestly.

### Finding 7: Latency figures are a model, not wall-clock
- [x] **Finding**: TTFT/ITL numbers could be mistaken for measurements.
- **Resolution**: Won't do — rows carry `mode="modeled"`; the model is a pure
  function of the committed byte equations (same HBM constants as the
  bandwidth CSV) plus seeded lognormal jitter, so P50/P90/P99 are
  reproducible. The derived claim (`amortize_decode_steps < 1`) is asserted
  for all 20 rows by `test_latency_crossover_amortizes_fast`.

### Finding 8: Stale background artifact risk
- [x] **Finding**: The first crossval run started before the calculator fix;
  committing its output would have embedded stale math.
- **Resolution**: Fixed — the stale run was killed and re-run after the fix;
  the committed CSV was produced entirely by the fixed code (verified by the
  recomputation test above).

---

## Summary

All 8 findings resolved: 3 fixed in this PR (fp16 strided-view crash,
calculator double-count, stale artifact), 5 resolved as documented
won't-dos with concrete follow-ups. 12 new tests, 3 new CSVs + manifests,
Flagship Figure 3 with provenance, KERNEL_NOTES theory-vs-measured table,
and 14 new CSV-backed README claims. All 8 local gates green (85 passed,
2 skipped GPU).

**Status**: APPROVED — ready for merge
