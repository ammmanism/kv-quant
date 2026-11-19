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

## ADR-008: Block-Diagonal vs. Full Walsh-Hadamard for Partial-RoPE Models

**Status**: Accepted — **Implements**: ADR-007 item 1 (v1.1.0-hadamard).

**Context.** v1.0 established that the INT4-K cliff is a post-RoPE channel
outlier problem ([#p6_v1_frontier_sweep.csv:acc_mean]) and that pre-RoPE
per-channel quantization is not viable inside a paged cache (ADR-003). The
remaining remedy is a *fixed orthogonal rotation*: apply the same map `R` to Q
and K so channel mass spreads and the INT4 grid sees a near-Gaussian shape,
without changing any attention logit.

Two architectures must be served. Llama-style models rotate the whole head
(full RoPE), so `R` may act on all `head_dim` channels. Qwen/Mistral-style
models rotate only the first `rotary_dim < head_dim` channels, leaving a
pass-through subspace whose content is *phase-free but position-mixed*; mixing
it into the rotary block would corrupt the positional subspace that RoPE just
encoded. Hence two scopes:

- `full`: `x' = H_D (D_seed x) / sqrt(D)`.
- `block_diag`:
  `x' = blkdiag(D1 H_r / sqrt(r), D2 H_p / sqrt(p)) x`, with `r = rotary_dim`,
  `p = D - r`, and independent seeded sign vectors (`seed`, `seed + 1`).
  When `p` is not a power of two the pass block is `I_p` (documented contract),
  which preserves orthogonality.

**Decision.** Implement both scopes in `kv_quant/hadamard.py`
(`fast_walsh_hadamard`, `hadamard_matrix`, `inverse_fast_walsh_hadamard`) with
`QuantConfig.resolve_hadamard_scope(head_dim, rotary_dim)` defaulting partial
RoPE to `block_diag` and full RoPE to `full`. Both Q and K are rotated *before*
`HybridPagedKVCache.append()`, so the memory-bound KV read loop in
`dequant_fused` gains **zero extra ops**: it computes `<Q', K'>` directly in
the Hadamard basis. `kv_quant/kernels/triton_hadamard.py` provides the fused
butterfly (`_triton_fwht_rotate_kernel`) with a PyTorch dispatch fallback.

**Evidence (measured, CPU).** Kurtosis is Pearson (3.0 = Gaussian) and is
measured on a scope-appropriate window, committed as the
`kurtosis_window` column: the rotary subspace for `block_diag` (the only part
that map touches) and the full head for `full`. Mixing those windows without
saying so is exactly the kind of comparison error that inflates a rescue.

On the synthetic Gaussian fixture `micro_kv_2k.pt` — the fixture the milestone
named — the rotation is a **null control**: per-channel kurtosis is already
≈ Gaussian, so `block_diag` moves 2.818 → 2.816 and changes INT4 SQNR by
−0.015 dB ([#p10_hadamard_rotation.csv:sqnr_delta_db], row 0), while `full`
moves 2.907 → 2.910 for +0.033 dB. The fixture contains no outlier structure to
rescue, and we report that rather than tune around it.

Rescue is therefore measured on `micro_kv_2k_outliers.pt`, an explicitly
synthetic derivative of that fixture (deterministic injection of 2 channels per
layer/head scaled `U(8, 32)`, seed 2026; injection parameters are committed in
`bench/activations/micro_kv_2k_outliers.injection.json`). There, `block_diag`
collapses rotary-subspace kurtosis from 14.842 to 2.011 with **+7.349 dB** INT4
SQNR ([#p10_hadamard_rotation.csv:sqnr_delta_db], row 2). `full` collapses
full-head kurtosis from 33.391 to 1.921 with +6.939 dB (row 3). The two
windows agree on the mechanism and disagree on the headline κ only because
`block_diag` deliberately leaves the pass-through channels alone — which is the
property that protects the positional subspace.

The projected needle sweep built on the `block_diag` gain keeps
`k4v4_hadamard_projected` within 0.01–0.03 absolute accuracy of FP16 across
4K–128K contexts, versus 0.845–0.950 for `k4v4_postrope`
([#p10_hadamard_rescue.csv:acc_mean]). That row's `config_id` carries the
`_projected` suffix because it is a modeled shift, not a measured forward pass;
the manifest states this and warns that the `ci95_*` columns are baseline
sampling intervals, not projection error bars.

**Honest deviation from the milestone target.** The milestone asked for
kurtosis `> 25 → ~3.0` on `micro_kv_2k.pt` and a `> 7.5 dB` SQNR gain. Two of
those three clauses are met only on the outlier derivative, and one is not met.
Kurtosis does exceed 25 (33.391) and does collapse to near-Gaussian (1.921),
but on `micro_kv_2k_outliers.pt`, not on the pristine fixture. The measured
gain is **+7.349 dB**, i.e. 0.151 dB short of the target
([#p10_hadamard_rotation.csv:sqnr_after_db]). This is a saturation effect, not
a tuning failure: post-rotation SQNR sits at ~23–24 dB, the INT4/group-32
Gaussian ceiling, so no rotation can gain much more. The **pre-declared**
sensitivity ladder (rows 4–9) confirms it: 2ch `U(20,50)` gives
17.574 → 1.829 / +7.157 dB; 3ch `U(20,60)` gives 16.896 → 1.876 / +4.490 dB;
1ch `U(10,40)` gives 9.750 → 2.351 / +7.142 dB. The ladder was fixed before
the numbers were seen and is committed so a reader can audit that no row was
selected to clear 7.5 dB. The residual gap is reported as unmet.

**Kernel status (ADR-006).** `_triton_fwht_rotate_kernel` is GPU-only code.
Triton ships no win32 wheel and this host has no CUDA device, so the butterfly
has not been compiled or executed here. The parity gate
(`tests/test_hadamard.py::TestRotateQkKernelDispatch::test_gpu_triton_rotation_matches_reference`)
is marked `gpu` and skips on CPU; the CPU dispatch path is asserted equal to
`fast_walsh_hadamard` on every scope. Until that gate runs on a GPU host, treat
the device path as unverified.

**Consequences.**
- The outlier fixture is **not committed** (18 MB binary). It is rebuilt
  deterministically from `micro_kv_2k.pt` plus the committed
  `micro_kv_2k_outliers.injection.json`, so `bench/data/p10_*` remains
  reproducible from a clean checkout by running `bench/hadamard_rescue.py`.
- **Open defect, not fixed here (pre-existing).** The legacy `use_hadamard=True`
  path does not use this API: `KVQuantWrapper.hadamard_hook` rotates Q with the
  sign-free `hadamard_transform`, while `quantizer.quantize_keys` rotates K with
  `hadamard_rademacher_transform` (seeded signs). Those are different maps, so
  `<Q', K'> != <Q, K>` on that path, and `KVQuantProfiler`'s `K4V4_Hadamard`
  schedule selects it. Fixing it is a protocol change spanning `wrapper.py`
  and `quantizer.py` (Q must receive the same signs K does), which the
  one-core-file-per-pass rule forbids in this milestone. `rotate_qk()` is the
  sanctioned entry point and is verified equal to the reference on every
  scope; do not enable `use_hadamard` until the integration slice lands.
- No bit-layout change: the golden layout hash is unchanged at
  `913cd248…`, so ADR-007's "re-pin at Hadamard landing" resolves to *no
  drift*, which the drift gate now confirms on every PR.
- Rotation is not self-inverse: `R = H D` and `D H` do not commute, so the
  inverse is `D H` (`inverse_fast_walsh_hadamard`). A property test caught the
  original docstring error; the forward/inverse pair round-trips to ≤1e-4.
- `full` scope requires `head_dim` to be a power of two and `block_diag`
  requires `rotary_dim` to be one; anything else raises rather than silently
  degrading, and the kernel wrapper falls back to the PyTorch path when a
  pass block is not a power of two.

---

## ADR-009: Retrieval-Head vs. Streaming-Head Mixed Precision

**Status**: Accepted — **Implements**: ADR-007 item 2 (v1.2.0-head-profiler).

**Context.** v1.1 (ADR-008) rescued the INT4-K cliff with an orthogonal
rotation, but not every head is equally fragile: retrieval/induction heads
route long-range softmax mass and are known to concentrate accuracy loss,
while local streaming heads tolerate compression. A profiler that scores
every `(layer_idx, kv_head_idx)` therefore lets a fixed bit budget go where
it buys accuracy. Three scores per head: `distant_attn_mass` (attention mass
outside the sink+local windows), `attn_entropy`, and `kl_perturbation`
(output KL under an INT4 perturbation of that head's logits).

**Decision.**
1. `profiler.compile_head_schedule(df, target_avg_bits=4.5)` ranks heads by
   `distant_attn_mass + kl_perturbation` and splits them so the schedule-wide
   mean of (K + V)/2 equals the budget: top-ranked heads become
   Retrieval/Induction heads at **K8V4** (long-range routing protected), the
   rest become Streaming heads at **K4V4 + Hadamard**. At the default 4.5-bit
   budget the derived fraction is exactly 25%. Ties break by (layer, head) so
   the schedule is deterministic.
2. `HybridPagedKVCache.set_head_schedule()` installs a schedule; pages are
   sealed per-head-width and record their `k_bits`/`v_bits`, so CoW cloning
   and dequantization stay consistent.
3. `dequant_fused.py` reads the per-head width **once per thread block** —
   the grid is `(batch * num_kv_heads, num_splits)`, so every program maps to
   one `kv_head` and `k_bits = tl.load(HEAD_K_BITS + kv_head)` is a uniform
   scalar; the 4-bit/8-bit unpack branch is therefore warp-uniform. The CPU
   reference path mirrors this with a per-head `head_k_bits` list and is
   parity-tested against a per-head dequant oracle
   (`test_head_adaptive_parity_cpu`).

**Evidence.** The schedule CSV commits two calibration sets
([#p11_head_sensitivity_schedule.csv:distant_attn_mass]). The *design*
calibration is a declared synthetic bimodal set at the mandated shape
(16 sequences × 8K context, 12 layers × 8 KV heads, ground truth ~20%
retrieval-biased): the compiler achieves **recall 20/20** on the true
retrieval heads with **precision 20/24** — the four extra picks are
borderline streaming heads whose random distant mass edged above the gap
(retrieval heads 0.596–1.000 vs streaming 0.557–0.594
([#p11_head_sensitivity_schedule.csv:kl_perturbation] for the companion
metric)). The *null control* profiles the committed Gaussian
`micro_kv_2k.pt` fixture and exposes a metric property a reviewer must know:
at 2K context the distant window covers 94% of the key range, so even uniform
attention scores `distant_attn_mass` ≈ 0.94 — the metric is only comparable
**within one calibration geometry**, and a short-context fixture cannot be
used to nominate retrieval heads.

**Honest deviation from the milestone target.** The milestone asked the
4.5-bit head-adaptive curve to "match FP16 within 0.3% at 128K". It does not:
the replay-adapter projection puts `head_adaptive_4p5_hadamard` at 0.9530
against FP16 0.9788 at 128K — a **2.6% gap** — and it does not even dominate
pure `k4v4_hadamard` (0.9576)
([#p11_head_rescue_curve.csv:acc_mean]). The bound is structural: the replay
model encodes K8V4 as a flat ~0.97 baseline, an irreducible ~2% FP16 gap, so
*any* blend containing a K8V4 term inherits it. What the schedule does buy
is visible in `delta_logp`: the adaptive curve has the lowest quantization
drift of all quantized configs at every context (0.1358–0.2614 vs
0.1464–0.2667 for pure Hadamard, and 1.1766 for the v1.0 cliff at 128K)
([#p11_head_rescue_curve.csv:delta_logp]), and the whole family holds the
v1.0 cliff (0.8455 at 32K) to ≥ 0.953. The 0.3% claim is reported as unmet
rather than reached by choosing a friendlier projection.

**Consequences.**
- Two schedule compilers now coexist: the phase-2 three-tier
  `compile_head_bit_schedule` (kept byte-stable for `p2_*` artifact
  reproduction) and this two-tier `compile_head_schedule`. New code must use
  the latter.
- The CPU cache path pads every channel of a mixed-width page to the widest
  packed length, so per-head bits do **not** shrink CPU-path page bytes; the
  "4.5-bit" figure is theoretical byte accounting over
  [#p11_head_sensitivity_schedule.csv:assigned_k_bits], realized only on the
  per-head-width GPU unpack path.
- V stays uniform: the kernel reads per-head *K* bits only (`HEAD_K_BITS`);
  per-head V bits are recorded on the page but not yet dispatched, matching
  the milestone's K-focused lookup.
- The Triton change (runtime `HEAD_K_BITS` scalar replacing the `K_BITS`
  constexpr) is GPU-only code and remains unverified on this CPU host per
  ADR-006; the `gpu`-marked parity test covers it on dispatch.
- No bit-layout change: the golden layout hash is untouched
  (`913cd248…`), confirmed by the drift gate.

---

## Provenance note

Every numeric claim above resolves against a committed CSV under `bench/data/`
via `scripts/check_claims.py` in CI. Artifacts produced on CPU-only hosts are
labeled `modeled`/`ncu-modeled` at row level and in their `.manifest.json`
files; re-generating them on a GPU host is documented in `REPRODUCE.md`.
