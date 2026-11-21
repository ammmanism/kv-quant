# Changelog

All notable changes to kv-quant. Releases are tagged `vMAJOR.MINOR.PATCH[-label]`
and published on GitHub Releases with their CSV/figure artifacts auto-attached
by `release.yml`.

## [2.0.0] - 2026-11-21

**The flagship release.** Phase II closes with the full two-act story and a
zero-dependency interactive evidence explorer.

### Added
- `docs/explorer.html` + `scripts/build_interactive_explorer.py` -
  self-contained HTML5/SVG dashboard compiled directly from the
  SHA256-verified `bench/data/*.csv` manifests (19 artifacts, dual manifest
  schema, LF-normalized hashing). Three live views: position-resolved cliff
  & ΔlogP explorer, ragged-batch fleet cost calculator, and a one-click
  `QuantConfig` snippet generator that is compile-checked against the real
  dataclass in CI ([#p4_needle_depth_matrix.csv:acc_mean],
  [#p13_ragged_concurrency.csv:max_concurrent_sessions],
  [#p14_agent_trajectory_drift.csv:v_schema_incident]).
- `make verify-explorer` / CI step `scripts/check_explorer_drift.py` -
  rebuilds the explorer in memory and byte-compares it against the committed
  file, so it can never drift from the committed evidence.
- `scripts/explorer_smoke.js` - headless DOM drive of every task x phase x
  context x budget combination (111 combos, zero runtime errors).
- `docs/LAUNCH_POST.md` - Show HN / X / LinkedIn technical launch post.
- README two-act restructure (Act I: the cliff; Act II: the rescue), THE_TRADE
  agent/fleet sections, REPRODUCE v2.0.0 checksum table.

### Fixed
- `scripts/check_adversarial_review.py` (the strict Invariant-9 gate behind
  `make verify`) crashed on cp1252 consoles and rejected ~40 already-resolved
  findings because it only recognized `Fixed in <sha>` markers and read just
  the first line of multi-line resolutions. It now emits UTF-8, recognizes
  RESOLVED / Won't do / Fixed markers, and joins the full resolution block.
  All 20 committed review files pass; unchecked boxes still fail hard.
- Explorer snippet initially emitted `concurrency=` / `device=` kwargs that
  belong to `BenchConfig` / `CacheConfig`, not `QuantConfig` - caught by the
  new compile-check test and corrected to teach the real API split.

### Verification at tag
- 166 tests passed (160 existing + 6 explorer), 9 local gates green
  (explorer drift gate added); all 19 tracked CSV manifests verified.

## [1.5.0-agent-bench] - 2026-11-21

**The agentic turn.** A 20-turn simulated ReAct harness (2 contexts x 5
configs x 32 seeds = 6,400 turn-records) separates routing failure from
payload corruption.

- K-amnesia is a **measured null**: routing flipped in `6/6,400`
  turn-records, all onto files read within `12,073` tokens (inside the 20K
  forgotten-span window), so no configuration re-reads a forgotten span
  ([#p14_agent_trajectory_drift.csv:k_amnesia_incident],
  [#p14_agent_trajectory_drift.csv:retrieval_margin]).
- V-schema corruption is the failure channel: INT2 values corrupt `15.94%`
  of payloads at 64K and collapse `78.1%` of runs at turn `8`; K8V4 `1.87%` /
  `9.4%`; K4V4+Hadamard `0.16%` / `0%`
  ([#p14_agent_trajectory_drift.csv:v_schema_incident],
  [#p14_agent_trajectory_drift.csv:loop_collapse_turn]).
- Equal-budget contrast: K8V4 vs K4V8 resolve `71.9%` vs `100%` at 64K -
  spending the four bits on V loses sessions, spending them on K does not
  ([#p14_agent_trajectory_drift.csv:resolved]).
- Figure 5 (300 DPI + provenance), ADR-012, adversarial review with 15
  findings, and a Colab reproduction notebook with a parallel `--jobs` path
  (bit-identical to serial, thread-pinning fix documented).

## [1.4.0-ragged-serving] - 2026-11-20

**Concurrent agent serving.** Ragged-batch paged runner with copy-on-write
prefix hashing for shared system prompts.

- CoW alone fits `3.73x` more concurrent sessions than unshared FP16
  (41 vs 11 @ 24 GiB / 32K); quant + CoW reaches `14.64x` (161 vs 11);
  at 64K the gains are `1.60x` / `6.80x`
  ([#p13_ragged_concurrency.csv:max_concurrent_sessions]).
- Modeled decode throughput rises with session count up to `9,000 tok/s`
  fleet-wide at 32K ([#p13_ragged_concurrency.csv:modeled_decode_tok_s]).

## [1.3.0-qjl-prototype] - 2026-11-19

**Query JL residual sketch.** Gaussian 1-bit QJL sign sketch on the K
residual as a cheap importance estimator, with a Johnson-Lindenstrauss math
proof, three-way statistical tests, and the MSE ablation
([#p12_qjl_mse_ablation.csv:mse_ratio_vs_naive]). Reference prototype only - not
wired into the serving path.

## [1.2.2-docs] - 2026-11-19

Whitepaper synchronization: THE_TRADE, KERNEL_NOTES, and ADRs 001-009
aligned with the v1.2.0 rescue numbers; no code or artifact changes.

## [1.2.0-head-profiler] - 2026-11-19

**Retrieval-head profiling.** Distant-attention-mass bimodality across
layer x KV-head; the 4.5-bit compiler recovers all `20` ground-truth
retrieval heads with precision `20/24`
([#p11_head_sensitivity_schedule.csv:distant_attn_mass],
[#p11_head_sensitivity_schedule.csv:assigned_k_bits]). Head-adaptive
schedule: `0.9530` vs FP16 `0.9788` at 128K (projected, replay adapter -
never quoted as a measurement; [#p11_head_rescue_curve.csv:acc_mean]).

## [1.1.0-hadamard] - 2026-11-18

**The cliff rescue.** Block-diagonal Walsh-Hadamard rotation on the rotary
subspace before K quantization: pre-Hadamard kurtosis `14.842` drops to
`2.011` and SQNR gains `+7.349 dB` on the outlier fixture
([#p10_hadamard_rotation.csv:post_kurtosis],
[#p10_hadamard_rotation.csv:sqnr_delta_db]). The v1.0 INT4-K cliff
(`0.8455` at 32K) recovers to `0.9652` - FP16-parity on the projected curve
([#p10_hadamard_rescue.csv:acc_mean]). Projections from declared synthetic
fixtures, honestly labeled.

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

## [0.8.0-cliff-hero] - 2026-11-17

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

[2.0.0]: https://github.com/ammmanism/kv-quant/releases/tag/v2.0.0
[1.5.0-agent-bench]: https://github.com/ammmanism/kv-quant/releases/tag/v1.5.0-agent-bench
[1.4.0-ragged-serving]: https://github.com/ammmanism/kv-quant/releases/tag/v1.4.0-ragged-serving
[1.3.0-qjl-prototype]: https://github.com/ammmanism/kv-quant/releases/tag/v1.3.0-qjl-prototype
[1.2.2-docs]: https://github.com/ammmanism/kv-quant/releases/tag/v1.2.2-docs
[1.2.0-head-profiler]: https://github.com/ammmanism/kv-quant/releases/tag/v1.2.0-head-profiler
[1.1.0-hadamard]: https://github.com/ammmanism/kv-quant/releases/tag/v1.1.0-hadamard
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
