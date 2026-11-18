# Changelog

All notable changes to kv-quant. Releases are tagged `vMAJOR.MINOR.PATCH[-label]`
and published on GitHub Releases with their CSV/figure artifacts auto-attached
by `release.yml`.

## [1.0.0] — 2026-11-18

**The coherence release.** No behavior change vs `v0.8.0-cliff-hero`; v1.0.0
locks the evidence base: ADRs, a reproducibility contract, citation metadata,
and a full-repo adversarial release audit.

### Added
- `docs/DECISIONS.md` — 7 Architectural Decision Records (3-tier hybrid
  cache, fused split-K kernel, post- vs pre-RoPE, nibble packing contract,
  ΔlogP + Wilson sensor, two-tier CI economics, v2.0 rescue roadmap), all
  CSV-cited.
- `REPRODUCE.md` — exact regeneration commands per artifact, host-context
  honesty notes (modeled vs measured), and SHA256 checksums for all 14 CSVs
  and 6 figures at this tag.
- `CITATION.cff` (v1.2.0 format) and Apache-2.0 `LICENSE`.
- `make verify-gpu` — GPU-host tier: GPU-marked parity tests + layout
  drift-check (auto-skips on CPU CI per ADR-006).

### Verification at tag
- 94 tests passed, 2 GPU-marked skipped, 2 deselected; 8 local gates green.
- Layout golden hash unchanged (`913cd248…`); claims checker fully resolved.

## [0.8.0-cliff-hero] — 2026-11-17

Flagship Figure 0 hero cliff chart (`fig0_hero_cliff.png`), CSV-backed CLI
Pareto advisor (`python -m kv_quant.advisor`), and `docs/THE_TRADE.md` with
the K/V asymmetry law and scoped production decision matrix. The cliff
callout is data-true (config-level, −20% vs FP16, onset ≤4K, flat through
128K: [#p6_v1_frontier_sweep.csv:acc_mean]); position-resolved dynamics are
presented as the Phase II hypothesis.
Notable fixes: advisor weight-budget arithmetic (bf16 bytes, not GiB/param),
like-for-like FP16 baseline pooling, INT6 1-byte packing deviation documented.

## [0.7.0-hardware-truth] — 2026-11-17

Phase 7: latency/TTFT/ITL suite, ncu-modeled DRAM bandwidth proof for the
fused split-K kernel (up to 9.09× traffic reduction at k4v4:
[#p7_ncu_bandwidth.csv:hbm_reduction_vs_naive]), VRAM byte calculator
cross-validated against the real allocator (R²=1.0000, MAPE=0.0000%:
[#p7_vram_crossval.csv:r_squared], [#p7_vram_crossval.csv:mape_pct]),
honest `modeled`/`ncu-modeled` labeling in manifests.

## [0.6.0-sweep] — 2026-11-16

Resumable sweep orchestrator (`kv_quant/sweep.py` + `sweeps/v1-frontier.yaml`),
the full 2,596-row frontier sweep (`p6_v1_frontier_sweep.csv`), and the
quality-golden baseline for golden-compare CI.

## [0.5.0-ecological] — 2026-11-16

Ecological evaluation baseline: longsum / multihop / longcode tasks,
concurrency sweep, and `bench/data/p5_ecological_baseline.csv`.

## [0.4.0-needle] — 2026-11-16

Position-resolved needle task with the continuous ΔlogP phase sensor and
Wilson 95% CIs; `p4_needle_depth_matrix.csv` (1,155 cells × 20 trials each:
[#p4_needle_depth_matrix.csv:ci95_low]) and cliff-coordinate detection.

## [0.3.0-fused-kernel] — 2026-11-15

Fused split-K GQA Triton decode kernel reading quantized pages directly
(zero FP16 materialization), kernel parity gate ≤1e-3, and the HBM traffic
model.

## [0.2.0-cache] — 2026-11-15

`HybridPagedKVCache`: FP16 sinks + quantized pages with Copy-on-Write prefix
hashing + FP16 ring tail; head-sensitivity profiling; CoW memory amortization
artifacts (up to 2.4× on forked sessions:
[#p2_cow_prefix_memory.csv:compression_factor]).

## [0.1.0-quantizer] — 2026-11-15

Layout contract (`kv_quant/layout.py`, golden-hash pinned), pure functional
INT4/INT6/INT8 quantizers with little-endian nibble packing, Hadamard
transform and QJL sketch ablations (post-Hadamard kurtosis collapse:
[#p1_hadamard_qjl_ablation.csv:post_hadamard_kurtosis]; SQNR bounds:
[#p1_sqnr_error_bounds.csv:sqnr_db]).

## [0.0.2-ci-hardened] — 2026-11-14

CI hardening: 8 required gate jobs, protected main (squash-only), PR
governance and adversary-gate checks, claims and provenance scripts.

## [0.0.1-init] — 2026-11-14

Repository skeleton, layout.py single source of truth, CI scaffold, golden
hash pinning.

[1.0.0]: https://github.com/ammmanism/kv-quant/releases/tag/v1.0.0
[0.8.0-cliff-hero]: https://github.com/ammmanism/kv-quant/releases/tag/v0.8.0-cliff-hero
[0.7.0-hardware-truth]: https://github.com/ammmanism/kv-quant/releases/tag/v0.7.0-hardware-truth
[0.6.0-sweep]: https://github.com/ammmanism/kv-quant/releases/tag/v0.6.0-sweep
[0.5.0-ecological]: https://github.com/ammmanism/kv-quant/releases/tag/v0.5.0-ecological
[0.4.0-needle]: https://github.com/ammmanism/kv-quant/releases/tag/v0.4.0-needle
[0.3.0-fused-kernel]: https://github.com/ammmanism/kv-quant/releases/tag/v0.3.0-fused-kernel
[0.2.0-cache]: https://github.com/ammmanism/kv-quant/releases/tag/v0.2.0-cache
[0.1.0-quantizer]: https://github.com/ammmanism/kv-quant/releases/tag/v0.1.0-quantizer
[0.0.2-ci-hardened]: https://github.com/ammmanism/kv-quant/releases/tag/v0.0.2-ci-hardened
[0.0.1-init]: https://github.com/ammmanism/kv-quant/releases/tag/v0.0.1-init
