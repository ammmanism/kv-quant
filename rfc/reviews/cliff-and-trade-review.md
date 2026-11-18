# Adversarial Review: Hero Cliff Chart, CLI Pareto Advisor & THE_TRADE

**Scope**: `cliff-and-trade`
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-18
**Branch**: `docs/hero-cliff-and-trade`
**Target Tag**: `v0.8.0-cliff-hero`

---

## Findings

### Finding 1: The mandated "Late-Context INT4-K Cliff at 32K–64K" is not in the data (severity: HIGH, research integrity)
- [x] **Finding**: The milestone text mandates a callout at "32K–64K", but the
  committed `p6_v1_frontier_sweep.csv` shows the K4 family sitting ~20% below
  FP16 from 4K onward, flat through 128K — a *config-level* cliff, not a
  context-resolved one. Drawing the mandated callout as-written would have
  been a fabricated claim (Invariant 1 violation).
- **Resolution**: Escalated to the repo owner; **user-approved data-true
  resolution applied**. The callout reads "INT4-K Cliff: −20.0% vs FP16 /
  onset ≤ 4K, flat through 128K (config-level, not context-resolved)" and the
  drop value is computed live from the CSV in `scripts/plot_hero_cliff.py`.
  THE_TRADE §2 presents the RoPE position-resolution mechanism as an explicit
  *hypothesis* for the cliff's magnitude, to be validated in Phase II.

### Finding 2: Pre-RoPE per-channel "remedy" actually hurts in the artifact
- [x] **Finding**: THE_TRADE's §2 narrative ("pre_rope helps but has friction")
  contradicts the committed sweep, where pre-RoPE per-channel scores ~1.9pp
  *worse* than post-RoPE at 128K.
- **Resolution**: Fixed in `docs/THE_TRADE.md` §2.3 — the section now states
  the data result plainly, explains why the hypothesis is defeated by the
  implementation friction, and cites the CSV. The figure legend/curve shows
  the same. No claim contradicts the artifact.

### Finding 3: Wilson CI bands are very wide and statistically unusual
- [x] **Finding**: Bernoulli-discretizing continuous cell means (n=40) produces
  bands ±5–12pp that visually swamp small differences (e.g. K8V4 vs FP16).
- **Resolution**: Won't do (re-sampling) — the derivation is documented in the
  figure meta (`wilson_ci_derivation`) and panel title ("conservative because
  scores are continuous"). The K8V4-within-0.4% finding is carried by exact
  annotated values (Panel B, THE_TRADE §1) rather than band separation. The
  cliff itself remains unambiguous: K4 curves lie below every FP16/K8 band.

### Finding 4: K4V8 ITL speedup uses a proxy row
- [x] **Finding**: `p7_latency_ttft_itl.csv` has no k4v8 rows; the advisor maps
  K4V8 → the k8v4 row.
- **Resolution**: RESOLVED with explicit disclosure — the proxy is commented in
  `kv_quant/advisor.py` (identical effective bits/element → identical modeled
  traffic) and footnoted in THE_TRADE §3 ("mapped to the k8v4 row"). Recorded
  in the CANDIDATES table, not silently.

### Finding 5: Advisor VRAM budget semantics must be explicit
- [x] **Finding**: `--vram-budget-gb 24` could plausibly mean "KV budget" or
  "total GPU memory"; the original implementation subtracted weights at
  1 GiB/B-param (wrong unit).
- **Resolution**: RESOLVED — budget means *total GPU memory*; the advisor
  subtracts bf16 weights (`params × 2 bytes`) and documents the arithmetic.
  Regression tests pin both semantics (`test_recommend_respects_vram_budget`
 proves FP16 KV at 7.0 GiB is excluded at a 17 GiB total budget but admitted
 at 24 GiB) and the negative-budget edge case (`test_recommend_empty_returns_nothing`).

### Finding 6: Advisor baseline comparability
- [x] **Finding**: The initial FP16 baseline pooled sink-0 and sink-4 FP16
  rows, producing negative "drops" vs candidates that only use sink-4 —
  a like-for-like violation.
- **Resolution**: RESOLVED — `fp16_baseline` now filters to the same variant
  family as candidates (`post_rope`, `sink_tokens=4`, `int` encoding,
  `k_per_channel=False`); FP16's own drop is now exactly 0.00%.

### Finding 7: Figure caption mandate vs model identity
- [x] **Finding**: The canvas must print "Model: Qwen2.5-7B-Instruct (7.6B, GQA
  28Q/4KV, rotary_dim=128)"; the sweep manifest records
  `model_id="Qwen/Qwen2.5-7B-Instruct"`, matching the mandated string's model.
- **Resolution**: RESOLVED — the mandated line is printed on the canvas
  verbatim (with the data-source attribution appended), and the meta.json
  records `model_id` from the manifest.

### Finding 8: Pareto frontier uses calculator (payload-only) compression
- [x] **Finding**: Compression ratios (1.00×/2.00×/2.66×/3.98×) exclude
  per-group scale metadata, so they are slightly optimistic vs the layout
  contract's page-level accounting (which would yield ~3.6× at k4v4).
- **Resolution**: Won't do (this PR) — the ratio is computed by the
  implementation-truth calculator validated at R²=1.0000/MAPE=0.0000%
  ([#p7_vram_crossval.csv:r_squared]) and the axis label states the geometry.
  Aligning with the layout-contract page accounting is deferred with the INT6
  packing fix tracked in `rfc/reviews/hardware-truth-review.md`.

### Finding 9: Quality retained can exceed 100%
- [x] **Finding**: K8V8 shows retained >100% at some contexts (seeded jitter
  makes it marginally exceed FP16).
- **Resolution**: RESOLVED — left data-true (no clipping); the advisor's
  like-for-like fix (Finding 6) removes the negative-drop artifact from the
  decision table, and THE_TRADE quotes the sink-4 like-for-like values.

---

## Summary

All 9 findings resolved: 5 fixed (data-true cliff callout with user approval,
pre-RoPE honesty, budget semantics, baseline comparability, proxy disclosure),
4 resolved as documented won't-dos with derivations. 9 new advisor tests
(94 total passing), Flagship Figure 0 with provenance, THE_TRADE.md with 16
CSV-backed claims, README hero section. All 8 local gates green.

**Status**: APPROVED — ready for merge
