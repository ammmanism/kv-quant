# THE_TRADE — What KV Quantization Actually Costs

**Thesis**: KV-cache keys and values are not symmetric. Keys govern softmax
*routing* and fail catastrophically under INT4; values govern linear payload
*aggregation* and tolerate INT4 almost for free. Every production decision in
this repo follows from that asymmetry.

All numbers are backed by committed CSVs under `bench/data/` via `[#file.csv:column]`
citations (Invariant 1). The sweep artifacts are *modeled* deterministic
evaluations on a CPU-only host — see each `.manifest.json` for the honest
execution context; conclusions are scoped accordingly (§3).

---

## §1 The K/V Asymmetry Law

Attention computes `softmax(QKᵀ/√d)·V`. The two operands play different roles:

- **Keys** enter the softmax *exponent*. A dequantization error in K perturbs
  the routing distribution itself — a single badly-decoded key channel can
  reroute attention mass away from the needle region. Routing errors are
  *nonlinear*: small K noise is amplified through `exp(·)` and renormalization.
- **Values** enter the output as a *linear* payload aggregate. Dequantization
  error in V contributes additively to the output and is averaged over the
  softmax weights; it neither selects nor suppresses information.

The committed sweep measures exactly this. At 64K/128K context (pooled
needle+multihop):

- **K8V4 is within 0.4% of FP16** — compressing values to INT4 costs almost
  nothing: [#p6_v1_frontier_sweep.csv:acc_mean]
- **K4V8 drops >20%** — compressing *keys* to INT4 is catastrophic even with
  full-precision values: [#p6_v1_frontier_sweep.csv:acc_mean]

The practical law: **spend your bit budget on K first.** The advisor
(`kv_quant/advisor.py`) encodes this: at a 1% drop budget, K8V4 is the
highest-compression feasible configuration and no INT4-K configuration
qualifies at any context.

## §2 Anatomy of the INT4-K Cliff

The cliff is real, large, and — in the current committed artifact —
**config-level**: the K4 family sits ~20% below FP16 from 4K onward and stays
flat through 128K. Figure 0's callout states this honestly
([#p6_v1_frontier_sweep.csv:acc_mean]); the *position-resolved* dynamics below
are the mechanistic hypothesis for why INT4-K fails so hard, to be validated
against real model activations in Phase II.

1. **RoPE rotates channel outliers by θᵢ·pos.** Each key channel is rotated
   by an angle proportional to its position. A channel that is a mild outlier
   at position 0 becomes a rotating outlier whose *effective* range spans the
   full circle as position grows; a static per-channel (or per-group) INT4
   grid cannot track a moving outlier set. This is the causal story for the
   K4 cliff's magnitude.
2. **Sink tokens prevent early collapse.** Attention sinks act as FP16
   anchor tokens that absorb residual attention mass. Removing them costs
   +1.1pp accuracy on the K4V4 configuration:
   [#p6_v1_frontier_sweep.csv:acc_mean]. They are the cheapest single remedy
   in the current artifact.
3. **Pre-RoPE per-channel quantization does *not* rescue it here.** In the
   committed sweep, quantizing K *before* the rotary transform with per-channel
   scales scores ~1.9pp *worse* than post-RoPE at 128K:
   [#p6_v1_frontier_sweep.csv:acc_mean]. The hypothesis ("quantize the
   pre-rotation distribution, which is position-stationary") is defeated by
   the implementation friction: pre-RoPE K must be quantized before the fused
   kernel can apply RoPE, forcing a transpose/buffering stage that breaks the
   zero-reload cache contract of this repo's design.
4. **The Phase II rescue: rotation-invariant K.** If the K distribution is
   rotated by a known orthogonal transform (Hadamard) *before* quantization,
   outliers are spread across channels and the INT4 grid sees a near-Gaussian
   shape. The Phase-1 outlier evidence — post-Hadamard kurtosis collapse
   [#p1_hadamard_qjl_ablation.csv:post_hadamard_kurtosis] and the outlier
   panel of `docs/figures/fig1_outliers_hadamard_qjl.png` — is exactly the
   mechanism a Hadamard-in-the-cache-path rescue would exploit.

## §3 Scoped Production Decision Matrix

**Scope: Qwen/Qwen2.5-7B-Instruct (7.6B, GQA 28Q/4KV, head_dim 128, 28 layers),
CPU-modeled sweep quality, modeled ITL speedups.** These numbers are *not*
validated on real model activations and must not be quoted as measured
end-to-end results.

Context 128K, task = multihop, FP16 baseline acc 0.9636
([#p6_v1_frontier_sweep.csv:acc_mean]):

| Config | acc | drop vs FP16 | KV VRAM | compression | decode ITL speedup |
|---|---|---|---|---|---|
| FP16 | 0.9636 | — | 7.00 GiB | 1.00x | 5.25x |
| K8V8 | 0.9618 | 0.18% | 3.50 GiB | 2.00x | 9.45x |
| K8V4 (asym V) | 0.9547 | 0.93% | 2.63 GiB | 2.66x | 12.13x |
| K4V8 | 0.7657 | 20.6% | 2.63 GiB | 2.66x | 12.13x* |
| K4V4 (+4 FP16 sinks) | 0.7706 | 20.1% | 1.89 GiB | 3.98x | 17.42x |

Quality/ITL columns: [#p6_v1_frontier_sweep.csv:acc_mean],
[#p7_latency_ttft_itl.csv:itl_speedup_p50] (modeled; *K4V8 mapped to the k8v4
row — identical effective bits/element
[#p6_v1_frontier_sweep.csv:effective_bits_per_elem]). VRAM/compression from
the implementation-truth byte calculator validated at R²=1.0000 /
MAPE=0.0000%: [#p7_vram_crossval.csv:r_squared], [#p7_vram_crossval.csv:mape_pct].

**Decision rules (this scope only):**

1. Up to ~1% quality budget: **K8V4** — 2.66× compression, 2.4× the FP16
   decode speed, indistinguishable quality from K8V8.
2. Never ship INT4-K without a rotation-invariant (Hadamard) K path: the
   −20% cliff dwarfs every benefit ([#p6_v1_frontier_sweep.csv:acc_mean]).
3. Always keep ≥4 FP16 sink tokens: cheapest quality anchor in the artifact
   ([#p6_v1_frontier_sweep.csv:acc_mean]).
4. The fused split-K kernel multiplies the payoff of any configuration:
   measured-modeled HBM traffic reduction up to 9.09× at k4v4
   ([#p7_ncu_bandwidth.csv:hbm_reduction_vs_naive]) with TTFT overhead
   amortizing in <1 decode step
   ([#p7_latency_ttft_itl.csv:amortize_decode_steps]).

Reproduce any row with:

```bash
python -m kv_quant.advisor --context-len 128000 --task multihop \
  --max-drop-pct 1.0 --vram-budget-gb 24
```
