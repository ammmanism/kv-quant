# Architectural Decision Records — v1.0.0

Status: Accepted for `v1.0.0`. Each ADR records context, decision,
alternatives rejected, and consequences. Numeric evidence cites committed CSVs
via `[#file.csv:column]` (Invariant 1); modeled-artifact caveats are stated
inline.

---

## ADR-001: 3-Tier Hybrid Cache (FP16 Sinks + Quantized Pages + FP16 Ring Tail)

**Status**: Accepted · **Decided**: v0.2.0-cache · **Supersedes**: uniform
whole-cache quantization prototype.

**Context.** Uniform KV quantization must either quantize the first tokens
(attention sinks, which receive disproportionate attention mass) or keep
special machinery for them; and streaming decode produces sub-group tails
that cannot be packed without stalling.

**Decision.** `HybridPagedKVCache` keeps three tiers: (1) `sink_tokens` in
FP16, never quantized; (2) full `group_size` blocks sealed into quantized
pages with Copy-on-Write prefix hashing; (3) a trailing `< group_size` tail in
an FP16 ring buffer (Invariant 5).

**Alternatives rejected.**
- *Uniform quantization including sinks*: loses the FP16 anchor that absorbs
  residual attention mass; the sweep prices sink removal at +1.1pp accuracy on
  K4V4: [#p6_v1_frontier_sweep.csv:acc_mean].
- *Quantize-everything-with-stalling*: sub-group tails would force per-token
  re-quantization of the whole group each decode step.
- *No ring (quantize partial groups immediately)*: breaks the exact packed
  byte layout that the fused kernel decodes in registers.

**Consequences.** CoW prefix sharing gives up to 2.4× memory amortization on
forked agent sessions: [#p2_cow_prefix_memory.csv:compression_factor]. The
three tiers recombine inside the fused kernel's single online-softmax pass
(ADR-002). Cost: three code paths to keep byte-accounted — enforced by the
layout golden hash (Invariant 3) and the R²=1.0000 VRAM cross-validation:
[#p7_vram_crossval.csv:r_squared].

---

## ADR-002: Fused Split-K GQA Triton Decode Kernel vs. Materialized Dequant-before-FA2

**Status**: Accepted · **Decided**: v0.3.0-fused-kernel.

**Context.** The naive path materializes dequantized FP16 K/V to HBM and then
lets SDPA re-read it — three HBM round-trips per decode step.

**Decision.** A Triton split-K decode kernel reads quantized pages directly,
unpacks INT4 in registers, dequantizes against L2-resident scale vectors, and
fuses the GQA broadcast dot with online softmax across all three cache tiers;
partial K-shards are reduced by a separate split-K reduction kernel.

**Alternatives rejected.**
- *Materialize + FA2*: pays the write + re-read round-trip forever; traffic
  scales with context length (the naive row matches byte-theory exactly:
  [#p7_ncu_bandwidth.csv:dram_total_bytes]).
- *CUDA-only kernel*: Triton gives portable SM-level control with tolerable
  compile times; CUDA C++ would fork maintenance for marginal gain.

**Consequences.** Measured-modeled DRAM traffic reduction up to 9.09× at
k4v4/64K vs the 8.11× pure-byte theory (the gap is L2 residency of GQA scale
vectors and pinned sinks): [#p7_ncu_bandwidth.csv:hbm_reduction_vs_naive].
Decode ITL speedup 17.4× at k4v4/128K with TTFT overhead amortizing in <1
decode step: [#p7_latency_ttft_itl.csv:itl_speedup_p50],
[#p7_latency_ttft_itl.csv:amortize_decode_steps]. Correctness is gated by the
kernel parity test (≤1e-3 vs the naive reference, Invariant 6) on synthetic
and captured-activation fixtures; GPU execution is verified on GPU hosts
(`make verify-gpu`) since CI is CPU-only.

---

## ADR-003: Post-RoPE Group Key Quantization vs. Pre-RoPE Per-Channel

**Status**: Accepted (provisionally; revisit in v2.0 per ADR-007).

**Context.** Keys are the routing-critical operand (ADR-002's asymmetry law):
K4V8 drops >20% while K8V4 stays within 0.4% of FP16:
[#p6_v1_frontier_sweep.csv:acc_mean]. Pre-RoPE quantization is attractive in
principle because the pre-rotation key distribution is position-stationary.

**Decision.** Default to **post-RoPE group quantization** (`k_quant_stage="post_rope"`,
per-`group_size` scales) in v1.0.

**Alternatives rejected.**
- *Pre-RoPE per-channel*: measured ~1.9pp **worse** than post-RoPE at 128K in
  the committed sweep: [#p6_v1_frontier_sweep.csv:acc_mean]. The mechanism
  hypothesis (position-stationary pre-rotation distribution) is defeated by
  implementation friction: RoPE must then be applied *after* dequantization
  inside the kernel, forcing a transpose/buffering stage that breaks the
  zero-reload cache contract. Full analysis: `docs/THE_TRADE.md` §2.
- *FP8-E4M3 key encoding*: kept as a sweep ablation axis, not a default.

**Consequences.** The INT4-K cliff stands unmitigated in v1.0 and is the
headline open problem; the v2.0 rescue is a rotation-invariant K path
(block-diagonal Walsh-Hadamard, ADR-007), motivated by the Phase-1 evidence
that Hadamard rotation collapses channel kurtosis:
[#p1_hadamard_qjl_ablation.csv:post_hadamard_kurtosis].

---

## ADR-004: Little-Endian Nibble Packing Contract (`layout.py`) vs. Prototype `torch.int4`

**Status**: Accepted · **Decided**: v0.1.0-quantizer.

**Context.** Prototyping used native half-byte tensor dtypes; production
needs a stable, kernel-decodable byte layout.

**Decision.** All sub-byte quantized payloads pack two 4-bit values per byte,
low nibble first (`INT4_LOW_MASK=0x0F`, `INT4_HIGH_SHIFT=4`), with per-channel
scale (FP16) + zero-point (uint8) metadata per `group_size` tokens.
`kv_quant/layout.py` is the single source of truth for masks, shifts, and
byte-accounting equations, pinned against a golden SHA256 hash
(`913cd248…`, Invariant 3).

**Alternatives rejected.**
- *`torch.int4` prototype dtype*: not universally available, no control over
  nibble order, and incompatible with in-register Triton unpacking.
- *Big-endian nibble order*: arbitrary, but must never silently flip — the
  golden-hash drift gate makes any change a deliberate, reviewed event.
- *High-nibble-first "matching someone else's kernel"*: portability guess;
  our contract is pinned and self-documented instead.

**Consequences.** Any layout edit fails CI before kernels can silently
mis-decode (drift-check gate). The contract is validated end-to-end by the
VRAM cross-validation (measured cache bytes match the layout-derived
calculator at R²=1.0000: [#p7_vram_crossval.csv:r_squared]) and by kernel
parity fixtures. Known deviation tracked separately: INT6 is currently stored
1 byte/value ("wasteful but correct"), pending the v2.0 packing upgrade
(`docs/KERNEL_NOTES.md` §4).

---

## ADR-005: Continuous ΔlogP Phase Sensor + Wilson 95% CIs vs. Scalar Perplexity

**Status**: Accepted · **Decided**: v0.4.0-needle.

**Context.** Perplexity is a scalar aggregate that hides *where* in the
context comprehension collapses; needle-style quality metrics need
distributionally honest uncertainty at tiny trial counts.

**Decision.** The evaluation spine (`tasks/base.py`) records a continuous
per-position `delta_logp` degradation signal alongside task metrics, and every
reported proportion carries a Wilson 95% confidence interval
(`wilson_ci`, Invariant-compliant manifests with 10 mandatory fields).
Needle-depth results are committed with per-cell `ci95_low`/`ci95_high`:
[#p4_needle_depth_matrix.csv:ci95_low],
[#p4_needle_depth_matrix.csv:ci95_high].

**Alternatives rejected.**
- *Scalar perplexity*: single number per run; cannot resolve depth- or
  position-resolved cliffs, and its aggregate washes out the bimodal
  per-head/per-position structure the repo's flagship figures expose.
- *Bootstrap CIs everywhere*: heavier and unstable at n≈20 per cell; Wilson
  intervals are closed-form, bounded in [0,1], and honest at small n. Where
  scores are continuous rather than binomial, the discretization is documented
  as conservative (see `docs/figures/fig0_hero_cliff.png.meta.json`).

**Consequences.** All downstream figures inherit uncertainty bands; claims
gates (Invariant 1) can verify not just means but that uncertainty columns
exist. Cost: manifests carry more columns; accepted.

---

## ADR-006: Two-Tier CI Economics — Fast CPU Provenance Gates per PR, GPU Parity Scheduled

**Status**: Accepted · **Decided**: v0.0.2-ci-hardened, extended v0.7.0.

**Context.** Full GPU verification (Triton compile, kernel parity on device,
ncu profiles) is slow, scarce, and unaffordable per-PR; but merging unverified
layout or claim drift is worse.

**Decision.** Two tiers. **Tier 1 (every PR, CPU, minutes)**: the 8 required
gates — unit tests, lint, strict typecheck, layout golden-hash drift, figure
provenance, kernel parity vs the CPU reference path, golden-curve regression,
and the claims+adversary governance gate. **Tier 2 (GPU hosts)**:
`make verify-gpu` / the `gpu-gate` workflow — Triton kernel parity on device
and, when hardware with `ncu` is attached, real DRAM counters replacing the
modeled ones. GPU-marked tests auto-skip on CPU CI.

**Alternatives rejected.**
- *GPU-required per PR*: blocks the repo on hardware availability; the
  one-core-file + evidence gates already catch the drift classes GPU runs
  would catch (layout hash, parity vs reference).
- *No GPU tier*: would let the Triton path rot unverified; v2.0 must not
  inherit an unverified kernel.

**Consequences.** Every GPU-dependent number shipped in v1.0 is labeled
`ncu-modeled`/`modeled` with the execution context in its manifest
([#p7_ncu_bandwidth.csv:dram_total_bytes] et al.); the upgrade path to real
profiles is documented in `docs/KERNEL_NOTES.md` §3 and requires no code
changes. Three-strike revert (Invariant 8) and the adversary gate
(Invariant 9) keep Tier 1 trustworthy.

---

## ADR-007: The v2.0 Rescue Roadmap

**Status**: Accepted (forward-looking) · **Supersedes**: ad-hoc "future work"
notes scattered across phase docs.

**Context.** v1.0 quantifies the problem precisely: INT4-K is catastrophic
([#p6_v1_frontier_sweep.csv:acc_mean]), pre-RoPE reordering fails on friction
(ADR-003), and the remaining quality gap concentrates in key-channel outliers
under rotary position coupling.

**Decision.** v2.0 attacks four fronts, in dependency order:

1. **Block-Diagonal Walsh-Hadamard K Rotation.** Apply a fixed orthogonal
   Hadamard to key channels before quantization: outliers spread across
   channels, the INT4 grid sees a near-Gaussian shape. Phase-1 evidence:
   post-Hadamard kurtosis collapse
   ([#p1_hadamard_qjl_ablation.csv:post_hadamard_kurtosis]) and SQNR bounds
   ([#p1_sqnr_error_bounds.csv:sqnr_db]). Must be cache-resident
   (rotation folded into the fused kernel's register pipeline), not a
   pre-write pass, to avoid ADR-003's transpose friction.
2. **Retrieval-Head Mixed Precision.** The profiler already identifies
   attention-sink / retrieval heads with concentrated distant attention mass
   ([#p2_head_sensitivity_map.csv:distant_attn_mass]); route higher bit-widths
   to them via the existing per-head config switch (`wrapper.set_kv_config`).
3. **Gaussian QJL Sketching.** Replace the 1-bit Rademacher QJL residual
   sketch with a Gaussian projection variant for better inner-product MSE at
   equal bits (Phase-1 ablation shows the QJL gap:
   [#p1_hadamard_qjl_ablation.csv:inner_prod_mse_no_qjl]).
4. **Multi-Turn Agent Amnesia Benchmarks.** Extend the task suite
   (`tasks/`) with multi-turn benchmarks where CoW prefix sharing (ADR-001) is
   the performance mechanism, closing the loop between the cache design and
   the evaluation ecology.

**Consequences.** Each item lands as its own milestone with its own adversary
review; the layout golden hash will be re-pinned exactly once, at the Hadamard
rotation landing, to keep the drift audit trail meaningful.

---

## Provenance note

Every numeric claim above resolves against a committed CSV under `bench/data/`
via `scripts/check_claims.py` in CI. Artifacts produced on CPU-only hosts are
labeled `modeled`/`ncu-modeled` at row level and in their `.manifest.json`
files; re-generating them on a GPU host is documented in `REPRODUCE.md`.
