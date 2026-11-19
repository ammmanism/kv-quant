# Adversarial Review: Head-Sensitivity Profiler & Flagship Figure 4 (v1.2.0)

**Scope**: `head-profiler` — retrieval-head profiler, per-head mixed-precision
dispatch (cache + kernel), p11 artifacts and `fig4_hadamard_and_head_rescue`.
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-19
**Branch**: `feat/head-sensitivity-profiler`
**Target Tag**: `v1.2.0-head-profiler`

**Artifacts audited**: `kv_quant/profiler.py` (`profile_calibration`,
`compile_head_schedule`), `kv_quant/cache.py` (per-head schedule, page bit
records), `kv_quant/kernels/dequant_fused.py` (`HEAD_K_BITS` warp-uniform
lookup, CPU `head_k_bits`), `tests/test_profiler.py`, `tests/test_cache.py`,
`tests/test_fused_kernel.py`,
`scripts/generate_phase11_artifacts.py`,
`bench/data/p11_head_sensitivity_schedule.csv` (+ manifest),
`bench/data/p11_head_rescue_curve.csv` (+ manifest),
`docs/figures/fig4_hadamard_and_head_rescue.png` (+ meta), ADR-009.

**Method**: line-level read of all new code, independent recomputation of the
schedule statistics from the committed CSV, a determinism re-run with hash
comparison, execution of the full local gate suite, and specific attention to
the two claims most likely to be fiction: the "top 25%" assignment and the
"matches FP16 within 0.3% at 128K" rescue headline.

---

## Findings

### Finding 1: The mandated 0.3% rescue claim is not reproduced (severity: HIGH)
- [x] **Finding**: The milestone requires Panel B to show the 4.5-bit
  head-adaptive curve "matching FP16 within 0.3% at 128K". The committed
  projection gives `head_adaptive_4p5_hadamard = 0.9530` vs
  `fp16 = 0.9788` at 128K — a **2.6% gap** — and the adaptive curve does not
  even dominate pure `k4v4_hadamard` (0.9576). The bound is structural, not a
  modeling accident: the replay adapter encodes K8V4 as a flat ~0.97
  baseline (an irreducible ~2% FP16 gap), and any blend containing a K8V4
  error term inherits it. I checked whether a friendlier but defensible
  model exists (e.g. treating retrieval heads as fully protected,
  `err = 0.75 * err(K4V4) * factor`) — it would land closer to FP16, but
  choosing it *after* seeing the first result would be exactly the post-hoc
  model selection this repository forbids.
- **Resolution**: RESOLVED — reported as unmet in ADR-009 with the structural
  reason; the projection math is fixed a priori in the manifest
  (`retrieval_fraction * err(K8V4) + (1 - retrieval_fraction) * err(K4V4) *
  10**(-delta_db/10)`, positive parts only) and committed before the ADR was
  written. What the schedule demonstrably buys is stated instead: the lowest
  `delta_logp` of any quantized config at every context (0.1358–0.2614 vs
  1.1766 for the v1.0 cliff at 128K), and the whole family holds the 32K
  cliff (0.8455) at ≥ 0.953.

### Finding 2: The distant-mass metric is context-length dependent (severity: MEDIUM)
- [x] **Finding**: The null-control rows (Gaussian `micro_kv_2k.pt` fixture at
  2K) show `distant_attn_mass` 0.933–0.942 for *uniform* attention: at 2K the
  sink+local exclusions leave 94% of the key range inside the "distant"
  window, so the metric is inflated by geometry, not head behavior. A reader
  comparing across calibration sets — or worse, running the profiler on a
  short-context fixture to nominate retrieval heads — would classify every
  head as retrieval.
- **Resolution**: RESOLVED — the null control is committed with an explicit
  `calibration_set` label and ADR-009 states the metric "is only comparable
  within one calibration geometry"; the design schedule uses a single
  geometry (8K) for all ranked heads.

### Finding 3: A merge bug briefly inflated projected trial counts 25× (severity: MEDIUM)
- [x] **Finding**: The first run of the rescue-curve projection merged the
  K8V4 baseline on `(depth_pct, needle_format)` without `context_len`, so
  every cell matched all five contexts' K8V4 rows: `trials = 16500` per
  projected row (25× the real 660) with absurdly tight Wilson intervals — a
  fabricated-precision failure mode that would have shipped had I not
  sanity-checked the trial count against the committed v1.0 baselines.
- **Resolution**: RESOLVED — the merge now keys on
  `(context_len, depth_pct, needle_format)`; regenerated output reports
  `trials = 660` and Wilson intervals consistent with the replay baseline
  rows beside it. Recorded here because `trials` columns in projection CSVs
  deserve exactly this sanity check in review.

### Finding 4: The "4.5-bit" budget is not realized as CPU-path bytes (severity: MEDIUM)
- [x] **Finding**: `HybridPagedKVCache._quantize_block` pads every channel of
  a mixed-width page to the widest packed length, so a page mixing K8 and K4
  heads occupies K8-width bytes for all heads on the CPU path; `vram_bytes`
  will not drop by the schedule's theoretical 4.5/8 ratio. A reader could
  take the ADR's "4.5-bit" as a measured VRAM saving.
- **Resolution**: RESOLVED — disclosed verbatim in ADR-009's consequences:
  the 4.5-bit figure is theoretical byte accounting over
  `assigned_k_bits`/`assigned_v_bits`, realized only on the per-head-width GPU
  unpack path; the CPU representation is a storage artifact. No VRAM claim
  is made from the CPU path.

### Finding 5: The Triton kernel change has never compiled or run (severity: MEDIUM)
- [x] **Finding**: `_splitk_paged_quant_attn_kernel` now takes a runtime
  `HEAD_K_BITS` pointer and a scalar `if k_bits == 4` unpack branch where it
  previously had a `K_BITS` constexpr. Triton has no win32 wheel and this
  host has no CUDA device, so a syntax error in that branch would be
  invisible to every local gate.
- **Resolution**: Won't do — cannot be executed here (ADR-006 two-tier
  economics). Mitigations: the warp-uniformity argument is structural (one
  scalar load per program; the grid maps one program to one kv_head) and
  stated in the kernel docstring; the *CPU* path implements the identical
  per-head semantics and is parity-tested against a per-head dequant oracle
  in `test_head_adaptive_parity_cpu`, which also asserts the adaptive output
  is strictly closer to the oracle than a uniform-bits decode; the
  `gpu`-marked test covers the device path on dispatch.

### Finding 6: Two schedule compilers now coexist (severity: LOW)
- [x] **Finding**: `compile_head_bit_schedule` (phase-2, three-tier,
  20/50/30) and `compile_head_schedule` (ADR-009, two-tier, budget-derived
  fraction) both live in `profiler.py`. A new caller grabbing the first
  symbol alphabetically gets streaming heads at K2 — far more aggressive than
  anything ADR-009 licenses.
- **Resolution**: RESOLVED — ADR-009 names `compile_head_schedule` as the
  sanctioned compiler and keeps the legacy one pinned for byte-stable
  reproduction of the committed `p2_*` artifacts; both compilers have
  dedicated contract tests (`test_compile_head_bit_schedule_unchanged`,
  `TestCompileHeadSchedule`).

### Finding 7: `profile_heads` mutates global RNG state (severity: LOW, pre-existing)
- [x] **Finding**: The phase-2 KL sensitivity path calls
  `torch.manual_seed(layer_idx * 1000 + h)` inside the profiler — hidden
  global state, in tension with Invariant 4 (pure functions, explicit
  generators). It also makes the KL noise identical across calibration
  sequences of the same head (an artifact masquerading as a property).
- **Resolution**: Won't do in this pass — replacing it with a
  `torch.Generator` argument changes the committed `p2_*` numbers and breaks
  their byte-stable reproduction. Recorded as v2.0 debt together with a
  one-time re-pin of the p2 artifacts when the generator refactor lands.

### Finding 8: Per-head V bits are recorded but not dispatched (severity: LOW)
- [x] **Finding**: `PhysicalPage` stores `v_bits` per head and
  `set_head_schedule` accepts per-head V widths, but the kernel reads only
  `HEAD_K_BITS` and the CPU fused path dispatches only `head_k_bits`.
- **Resolution**: RESOLVED by scoping — the milestone specifies the per-head
  *K* lookup (V is the value payload whose error the replay model already
  treats as benign); the page's V record is kept so a future V dispatch needs
  no migration. ADR-009 states "per-head V bits are recorded on the page but
  not yet dispatched".

---

## Empirical checks performed

1. **Schedule quality recomputed from the committed CSV**: ground truth is 20
   retrieval-biased heads of 96; the compiler assigned K8V4 to 24 — all 20
   true retrieval heads (recall 20/20) plus 4 borderline streaming heads
   (precision 20/24). Class distant-mass ranges: retrieval 0.596–1.000 vs
   streaming 0.557–0.594 — separable, though narrower than the heatmap
   suggests; the four misclassifications sit exactly in that gap's noise
   floor.
2. **Budget arithmetic**: mean of (K+V)/2 over the compiled schedule equals
   4.500 exactly; the derived retrieval fraction is 24/96 = 0.25 as mandated.
3. **Determinism**: regenerated both p11 CSVs; line-ending-normalized SHA256
   identical before and after (`2a712857…`, `55691ae8…`).
4. **Figure provenance**: `check_figure_provenance.py` passes for
   `fig4_hadamard_and_head_rescue.png` (4469×1618, 300 DPI, generating
   script + three source CSV hashes recorded, including the p10 rotation CSV
   the Hadamard factor derives from).
5. **Claims gate**: 94 claims, all resolving; the five new ADR-009 citations
   point at `p11_*` columns that exist.
6. **Test suite**: full local gates green — profiler (12 new tests), cache
   (per-head precision ordering, schedule validation, fork/CoW bit
   survival), fused kernel (mixed-width parity + adaptive-closer-than-uniform
   assertion), plus all prior suites.

## Verdict

The profiler, schedule compiler, and per-head dispatch are correct and
property-tested; the schedule recovers the declared ground truth with
perfect recall at the mandated budget. The milestone's two headline numbers
diverge from measurement in opposite directions and are both reported: the
top-25% assignment is real and exact, while the "within 0.3% of FP16"
rescue is not (2.6% under an a-priori-fixed projection, bounded by the
replay's K8V4 pessimism). Approved for landing at `v1.2.0-head-profiler`
with the deviation, the CPU-path VRAM caveat, and the GPU-unverified kernel
status stated in ADR-009.
