# Architectural Decision Records — v1.2.0

Nine accepted decisions spanning `v1.0.0` (cliff discovery, kernel, governance)
through `v1.2.0` (Hadamard rescue, head-adaptive precision). Every ADR uses the
same template: **Status | Date/Tag | Context | Decision | Trade-offs &
Rejected Alternatives | Verification Artifact**. Numeric evidence cites
committed CSVs via `[#file.csv:column]` (Invariant 1); modeled artifacts are
labeled inline.

---

## ADR-001: 3-Tier Hybrid Cache (FP16 Sinks + Quantized Pages + FP16 Ring Tail)

**Status:** Accepted.
**Date/Tag:** `v0.2.0-cache` (reaffirmed `v1.2.0`).

**Context.** Uniform KV quantization must either quantize the first tokens —
attention sinks, which hold disproportionate attention mass — or carry special
machinery for them; decode also produces sub-`group_size` tails that cannot be
packed without stalling or re-quantizing.

**Decision.** `HybridPagedKVCache` keeps three tiers: (1) `sink_tokens` in FP16,
never quantized; (2) full `group_size` blocks sealed into quantized pages with
Copy-on-Write prefix hashing; (3) a trailing `< group_size` tail in an FP16 ring
buffer that seals automatically (Invariant 5). All three tiers recombine inside
one online-softmax pass in the fused kernel (ADR-002).

**Trade-offs & Rejected Alternatives.**
- *Quantize sinks too*: loses the FP16 anchor; removing `4` sinks costs
  `0.96` percentage points at `64K` on the K4V4 needle path
  ([#p6_v1_frontier_sweep.csv:acc_mean]).
- *Quantize partial groups immediately*: would force per-token re-packing and
  break the rectangular layout the register decoder expects.
- *No ring buffer*: decode stalls at group boundaries.

CoW sharing amortizes up to `2.4×` memory across forked agent sessions
([#p2_cow_prefix_memory.csv:compression_factor]); cost is three byte-accounted
code paths, pinned by the layout hash.

**Verification Artifact.** `tests/test_cache.py` (tier partition, fork, CoW
clone, per-head bit survival); VRAM calculator cross-validation at `R² = 1.0000`
([#p7_vram_crossval.csv:r_squared]).

---

## ADR-002: Fused Split-K GQA Triton Decode Kernel vs. Materialized Dequant-before-FA2

**Status:** Accepted.
**Date/Tag:** `v0.3.0-fused-kernel` (extended `v1.1.0`–`v1.2.0`).

**Context.** The naive path materializes dequantized FP16 K/V to HBM and lets
SDPA re-read it — three DRAM round-trips per decode step (§1 of
`KERNEL_NOTES.md`).

**Decision.** A Triton split-K kernel reads quantized pages directly, unpacks
INT4 in registers using the `layout.py` mask/shift contract, dequantizes against
L2-resident scale vectors, fuses the GQA broadcast dot with online softmax, and
reduces `(batch × kv_heads, num_splits)` partial statistics in a second kernel.
A PyTorch CPU reference implements identical math for parity gating.

**Trade-offs & Rejected Alternatives.**
- *Materialize + FA2*: pays the write + re-read round trip forever; naive
  traffic matches pure byte theory exactly
  ([#p7_ncu_bandwidth.csv:dram_total_bytes]).
- *CUDA C++ instead of Triton*: portable SM control with tolerable compile
  time; forking a second language buys marginal gain.

Measured-modeled DRAM reduction reaches `9.085×` at k4v4/`64K`, above the
`8.11×` byte-theory ratio because of L2 residency
([#p7_ncu_bandwidth.csv:hbm_reduction_vs_naive]); ITL P50 improves `17.415×` at
`128K` with TTFT overhead amortizing in `0.06` decode steps
([#p7_latency_ttft_itl.csv:itl_speedup_p50],
[#p7_latency_ttft_itl.csv:amortize_decode_steps]).

**Verification Artifact.** `tests/test_parity.py`, `tests/test_fused_kernel.py`
(CPU parity matrix, mixed-width per-head path), `make verify-gpu` on CUDA hosts;
counter table `bench/data/p7_ncu_bandwidth.csv` (`ncu-modeled`).

---

## ADR-003: Post-RoPE Group Key Quantization vs. Pre-RoPE Per-Channel

**Status:** Accepted (post-RoPE default); superseded for quality by ADR-008,
which removes the outlier problem instead of relocating the grid.
**Date/Tag:** `v0.2.0-quantizer` (reaffirmed `v1.0.0`).

**Context.** Keys are the routing-critical operand: at `64K` K4V4 scores
`0.7716` versus K8V4 `0.9611` on needle
([#p6_v1_frontier_sweep.csv:acc_mean]). Pre-RoPE per-channel quantization looks
attractive because the pre-rotation distribution is position-stationary.

**Decision.** Default to `k_quant_stage="post_rope"` with per-`group_size`
scales for `v1.0.0`; keep pre-RoPE as a sweep ablation only.

**Trade-offs & Rejected Alternatives.**
- *Pre-RoPE per-channel*: scores `0.8212` at `32K` versus `0.8455` post-RoPE
  ([#p4_needle_depth_matrix.csv:acc_mean]) and requires applying RoPE after
  dequantization inside the kernel, forcing a token-axis transpose/buffer stage
  that breaks the zero-reload cache contract — the friction that motivated the
  cache-resident rotation of ADR-008.
- *FP8 e4m3 K*: retained as an ablation axis, not a default.

**Verification Artifact.** `bench/data/p4_needle_depth_matrix.csv`
(stage comparison), `bench/data/p6_v1_frontier_sweep.csv` (K/V config matrix),
`tests/test_quantizer.py`.

---

## ADR-004: Little-Endian Nibble Packing Contract (`layout.py`) vs. Prototype `torch.int4`

**Status:** Accepted.
**Date/Tag:** `v0.1.0-quantizer`.

**Context.** Prototyping used native half-byte tensor dtypes; production needs
a stable, kernel-decodable byte layout shared by cache, quantizer, and Triton.

**Decision.** Two 4-bit values per byte, low nibble first
(`INT4_LOW_MASK=0x0F`, `INT4_HIGH_SHIFT=4`), per-channel FP16 scale and uint8
zero-point per `group_size` tokens. `kv_quant/layout.py` is the single source of
truth for masks, shifts and byte equations, pinned to golden hash `913cd248…`
(Invariant 3).

**Trade-offs & Rejected Alternatives.**
- *`torch.int4` prototype dtype*: not universally available, no nibble-order
  control, incompatible with in-register Triton unpacking.
- *Big-endian / high-nibble-first*: arbitrary portability guesses; the golden
  hash makes any change a deliberate reviewed event.

The contract is validated end-to-end by measured cache bytes matching the
calculator at `R² = 1.0000` ([#p7_vram_crossval.csv:r_squared]) and remains
unchanged through v1.2.0 — the Hadamard and head-adaptive work is a value-space
transform, not a layout change.

**Verification Artifact.** `scripts/check_layout_drift.py` (golden hash),
`tests/test_layout.py`, `bench/data/p7_vram_crossval.csv`.

---

## ADR-005: Continuous ΔlogP Phase Sensor + Wilson 95% CIs vs. Scalar Perplexity

**Status:** Accepted.
**Date/Tag:** `v0.4.0-needle`.

**Context.** Scalar perplexity hides *where* comprehension collapses; needle
metrics at small `n` need distributionally honest uncertainty.

**Decision.** `tasks/base.py` records per-position continuous `delta_logp`
alongside task metrics; every reported proportion carries a Wilson 95% interval
(`wilson_ci`). The p4 needle matrix commits `ci95_low`/`ci95_high`
([#p4_needle_depth_matrix.csv:ci95_low], [#p4_needle_depth_matrix.csv:ci95_high]).

**Trade-offs & Rejected Alternatives.**
- *Scalar perplexity*: cannot resolve the depth-resolved cliff; the continuous
  sensor rises from `0.5745` nats at `16K` to `0.6531` at `32K` while discrete
  accuracy is still `0.9500` at `16K` — one context doubling of early warning
  ([#p4_needle_depth_matrix.csv:delta_logp],
  [#p4_needle_depth_matrix.csv:acc_mean]).
- *Bootstrap CIs*: heavier and unstable at `n ≈ 20` per cell; Wilson intervals
  are closed-form and bounded in `[0,1]`. Continuous-score discretization is
  documented as conservative in the figure metadata.

**Verification Artifact.** `tests/test_needle.py`, `tests/test_task_suite.py`,
`bench/data/p4_needle_depth_matrix.csv`.

---

## ADR-006: Two-Tier CI Economics — Fast Per-PR CPU Provenance Gates vs. Scheduled GPU Parity

**Status:** Accepted; workflow repaired `v1.2.1` (golden gate resurrected).
**Date/Tag:** `v0.0.2-ci-hardened`, extended `v0.7.0`, repaired `v1.2.1`.

**Context.** Full GPU verification (Triton compile, device parity, `ncu`
counters) is slow and scarce; merging unverified layout/claim drift is worse.

**Decision.** **Tier 1 (every PR, CPU, minutes):** the 8 `verify.ps1` gates —
tests, ruff, strict mypy, layout golden hash, figure provenance, kernel parity
vs the CPU reference, golden-curve regression, claims + adversary governance.
**Tier 2 (GPU):** `make verify-gpu` and the `gpu-gate` workflow, now triggered
by `workflow_dispatch` after the `v1.2.1` repair (no self-hosted runner is
registered, so per-push dispatch produced permanent red). GPU-marked tests
auto-skip on CPU.

**Trade-offs & Rejected Alternatives.**
- *GPU required per PR*: blocks the repo on scarce hardware; Tier 1 already
  catches layout hash and reference-parity drift.
- *No GPU tier*: lets the Triton path rot unexecuted. Every GPU-dependent
  number is labeled `ncu-modeled`/`modeled`
  ([#p7_ncu_bandwidth.csv:mode], [#p7_latency_ttft_itl.csv:mode]).

**Verification Artifact.** `verify.ps1`, `.github/workflows/gpu-gate.yml`,
`rfc/reviews/ci-gate-repair-review.md`; first real CI execution of the golden
gate passed in `28 s` on PR #12.

---

## ADR-007: Zero-Reload Module-Swap Wrapper (`KVQuantWrapper`; spec name `QuantizedAttentionWrapper`)

**Status:** Accepted.
**Date/Tag:** `v0.5.0-wrapper` (reaffirmed `v1.2.0`).

**Context.** Applying a new quantization configuration must not reload model
weights or re-patch modules repeatedly; naive forward-hook monkey-patching is
fragile across model families and leaks hooks on context exit.

**Decision.** `KVQuantWrapper(model, quant_config, cache_config)` is a context
manager: `__enter__` installs forward hooks on `q_proj`/`k_proj`/`v_proj`,
`__exit__` removes them, and `set_kv_config()` swaps configuration at runtime
by rebuilding only the cache on the next forward. The CLI advisor selects a
configuration from committed artifacts; at a `1%` drop budget it recommends
`k_bits=8, v_bits=4, k_quant_stage='post_rope', sink_tokens=4, group_size=64`
([#p6_v1_frontier_sweep.csv:acc_mean],
[#p7_latency_ttft_itl.csv:itl_speedup_p50]).

**Trade-offs & Rejected Alternatives.**
- *Per-call monkey-patching*: leaks hooks and cannot express zero-reload
  switching; rejected in favor of scoped `__enter__`/`__exit__`.
- *Reloading the model per config*: prohibitive for 7B-class weights.
- *Name `QuantizedAttentionWrapper`*: the milestone spec's name; the shipped
  class is `KVQuantWrapper`, and the README documents the real API explicitly
  rather than a phantom symbol.

**Verification Artifact.** `tests/test_cache.py::test_wrapper_zero_reload_switch`,
`tests/test_advisor.py`, and the executed CLI smoke in `README.md` §5.

---

## Roadmap — v1.1.0 / v1.2.0 deliveries (former "ADR-007: v2.0 Rescue Roadmap")

The forward-looking roadmap formerly numbered ADR-007 is preserved here as
delivered history; ADR-008 and ADR-009 implement items 1 and 2. Its four items
were: (1) block-diagonal Walsh-Hadamard K rotation (delivered `v1.1.0`,
ADR-008); (2) retrieval-head mixed precision (delivered `v1.2.0`, ADR-009);
(3) Gaussian QJL sketching (still future work; phase-1 ablation gap
([#p1_hadamard_qjl_ablation.csv:inner_prod_mse_no_qjl])); (4) multi-turn agent
amnesia benchmarks (future work). The roadmap's "re-pin the layout hash at the
Hadamard landing" clause resolved to **no drift**: the golden hash remains
`913cd248…` because rotation changes values, not bit layout.

---

## ADR-008: Block-Diagonal vs. Full Walsh-Hadamard for Partial-RoPE Models

**Status:** Accepted — implements roadmap item 1.
**Date/Tag:** `v1.1.0-hadamard`.

**Context.** v1.0 established that the INT4-K cliff is a post-RoPE channel
outlier problem ([#p6_v1_frontier_sweep.csv:acc_mean]) and pre-RoPE per-channel
quantization is not viable inside a paged cache (ADR-003). Llama-style models
rotate the whole head; Qwen/Mistral-style models rotate only `rotary_dim <
head_dim`, leaving a pass-through subspace that must not be mixed into the
position-bearing rotary block.

**Decision.** `fast_walsh_hadamard` implements `full` (`H_D D_seed/√D`) and
`block_diag` (`blkdiag(D₁H_r/√r, D₂H_p/√p)`); `QuantConfig.resolve_hadamard_scope`
defaults partial RoPE to `block_diag`. Both Q and K are rotated through
`rotate_qk()` before cache append, so the decode loop computes `<Q',K'>` in the
Hadamard basis with zero extra ops. `inverse_fast_walsh_hadamard` is the true
inverse `D H` (the forward `H D` is not self-inverse).

**Trade-offs & Rejected Alternatives.**
- *Full rotation on partial RoPE*: contaminates the pass-through subspace with
  rotary content; rejected.
- *Sign-free H*: mathematically orthogonal but no outlier spreading; rejected.
- *Pre-write rotation pass*: materializes rotated K to HBM; rejected in favor
  of write-path fusion.

On the Gaussian `micro_kv_2k.pt` fixture rotation is a null control
(`2.818 → 2.816` kurtosis, `−0.015 dB` SQNR)
([#p10_hadamard_rotation.csv:sqnr_delta_db], row 0). On the declared synthetic
outlier derivative, block-diagonal rotation gives rotary kurtosis
`14.842 → 2.011` and `+7.349 dB` (row 2); full scope gives full-head kurtosis
`33.391 → 1.921` and `+6.939 dB` (row 3). The milestone's `> 7.5 dB` target is
`0.151 dB` above the measured ceiling ([#p10_hadamard_rotation.csv:sqnr_after_db]);
the pre-declared ladder shows no injection regime exceeds it.

**Verification Artifact.** `tests/test_hadamard.py` (hypothesis invariance for
rotary widths `32/64/128`, orthogonality, inverse round-trip),
`bench/data/p10_hadamard_rotation.csv` (+ manifest),
`rfc/reviews/hadamard-rotation-review.md`. Open defect recorded there: the
legacy `use_hadamard=True` wrapper/quantizer path rotates Q and K with different
maps; use `rotate_qk()` until the v1.1.1 integration slice lands.

---

## ADR-009: Retrieval-Head vs. Streaming-Head Adaptive Mixed Precision with Warp-Uniform Dispatch

**Status:** Accepted — implements roadmap item 2.
**Date/Tag:** `v1.2.0-head-profiler`.

**Context.** Not all heads are equally fragile: retrieval/induction heads
concentrate long-range softmax mass, while local streaming heads tolerate
compression. `profile_heads`/`profile_calibration` score every
`(layer, kv_head)` on distant attention mass, attention entropy and INT4 logit
KL perturbation.

**Decision.** `compile_head_schedule(df, target_avg_bits=4.5)` ranks by
`distant_attn_mass + kl_perturbation` and derives the retrieval fraction from
the budget: top heads → K8V4, the rest → K4V4 + Hadamard. At `4.5` bits the
fraction is exactly `25%` and mean `(K+V)/2 = 4.5`
([#p11_head_sensitivity_schedule.csv:assigned_k_bits]). `HybridPagedKVCache`
seals per-head-width pages; `dequant_fused` loads `HEAD_K_BITS` once per program
so the unpack branch is warp-uniform.

**Trade-offs & Rejected Alternatives.**
- *Uniform 4-bit everywhere*: keeps the v1.0 cliff
  ([#p11_head_rescue_curve.csv:acc_mean]).
- *8-bit everywhere*: forfeits `3.95×` compression
  ([#p7_vram_crossval.csv:compression_ratio]).
- *Per-head V dispatch*: recorded on the page but not implemented; the
  milestone specified a K-only lookup.
- *CPU-path padding*: mixed-width pages pad to the widest channel, so `4.5`
  bits is theoretical byte accounting, not measured CPU VRAM.

On the declared synthetic bimodal calibration (`16 × 8K`, `12 × 8` heads),
the compiler recovers all `20` ground-truth retrieval heads with `24`
assignments (recall `20/20`, precision `20/24`)
([#p11_head_sensitivity_schedule.csv:distant_attn_mass]). The milestone's "match
FP16 within `0.3%` at `128K`" is **not** met: head-adaptive scores `0.9530`
against FP16 `0.9788` (`2.6%` gap) and does not dominate pure Hadamard `0.9591`
([#p11_head_rescue_curve.csv:acc_mean]); the replay's K8V4 baseline carries an
irreducible `~2%` gap. Its advantage is the lowest `delta_logp` of any quantized
config ([#p11_head_rescue_curve.csv:delta_logp]).

**Verification Artifact.** `tests/test_profiler.py` (12 tests incl. hypothesis
schedule budgets), `tests/test_cache.py` (per-head precision, fork/CoW),
`tests/test_fused_kernel.py` (mixed-width parity, adaptive-closer-than-uniform),
`bench/data/p11_head_sensitivity_schedule.csv`,
`rfc/reviews/head-profiler-review.md`.

---

## Provenance note

Every numeric claim resolves to a committed CSV under `bench/data/` via
`scripts/check_claims.py` in CI. Artifacts produced on CPU-only hosts are
labeled `modeled`/`ncu-modeled` at row level and in their manifests; figure
provenance (SHA256 + ≥300 DPI) is enforced by
`scripts/check_figure_provenance.py`. Reproduction is documented in
`REPRODUCE.md`; the kernel derivation is `KERNEL_NOTES.md`; the empirical
synthesis is `THE_TRADE.md`.
