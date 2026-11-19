# Hadamard Rotation and Gaussian QJL — The Two Proofs Behind the v1.x Rescues

Scope: the exact mathematics behind ADR-008 (block-diagonal Walsh–Hadamard
rotation) and ADR-010 (Gaussian 1-bit Quantized Johnson–Lindenstrauss residual
sketch). Every empirical figure cites a committed CSV under `bench/data/`
(Invariant 1). Both results share one theme: *orthogonality buys exactness*.

---

## Part I — Block-Diagonal Hadamard Invariance (ADR-008)

**Claim.** Let `R` be an orthogonal map (`RᵀR = I`) applied to both the query
`q` and the key `k`. Then the attention logit is *exactly* invariant:

```text
<q R, k R> = q R Rᵀ kᵀ = <q, k>
```

No approximation, no condition on `q` or `k` — only orthogonality.

**Why block-diagonal for partial RoPE.** A partial-RoPE model rotates only the
first `r = rotary_dim` channels of each head; the remaining `p = D − r`
channels pass through unrotated but are still position-mixed content. Define

```text
R = blkdiag(D₁ H_r / √r , D₂ H_p / √p)
```

where `H_m` is the Sylvester–Hadamard butterfly (an `m×m` power of two) and
`D₁, D₂` are seeded Rademacher diagonals (`seed`, `seed + 1`). Orthogonality
of each block makes `R` orthogonal; the block structure guarantees **zero
subspace contamination**: a vector supported on the pass-through channels has no
component in the rotary block after rotation, and vice versa. Mixing the two
subspaces with a full-width `H_D` would blur the positional phase structure
RoPE just encoded.

**What the rotation buys quantitatively.** The butterfly spreads channel mass
uniformly: a single massive channel becomes a small perturbation across all
`r` channels, collapsing the kurtosis the INT4 grid cannot track. Measured on
the declared synthetic outlier fixture, rotary-subspace Pearson kurtosis falls
`14.842 → 2.011` with an INT4 SQNR gain of `+7.349 dB`
([#p10_hadamard_rotation.csv:pre_kurtosis], [#p10_hadamard_rotation.csv:sqnr_delta_db]);
the pristine Gaussian fixture is a null control (`2.818 → 2.816`,
`−0.015 dB`) because it has no outliers to spread
([#p10_hadamard_rotation.csv:sqnr_delta_db]).

**One subtlety worth writing down.** `R = H D` (signs, then butterfly) is
orthogonal but *not* self-inverse: `H` and `D` do not commute. The inverse is
the transpose `D H` — butterfly, then signs — implemented as
`inverse_fast_walsh_hadamard`. A property test caught the wrong version of this
sentence before it shipped.

---

## Part II — The Gaussian QJL Arcsine/Half-Normal Proof (ADR-010)

**Setting.** A key `k` is quantized to symmetric INT2, giving the reconstruction
`k̂` and residual `r = k − k̂ ∈ R^D`. We store, per key vector:

- the packed INT2 payload (4 crumbs per byte, layout-pinned by
  `kv_quant/layout.py` + `quantizer`),
- `M` sign bits `z_m = sign(<s_m, r>)` of a Gaussian projection
  `S ∈ R^{M×D}`, `S_ij ~ N(0,1)` i.i.d. — packed two `uint32` words at the
  prototype width `M = 64`,
- the FP16 residual norm `‖r‖₂`.

**Claim (the exact identity).**

```text
E_S [ √(π/2) / M · ‖r‖₂ · <S q, sign(S r)> ] = <q, r>
```

**Proof.** Fix `r ≠ 0` and `q`. Because a Gaussian row `s ~ N(0, I_D)` is
*rotationally invariant*, rotate the basis so `r = ‖r‖ e₁`. Then:

1. `sign(<s, r>) = sign(‖r‖ s₁) = sign(s₁)`.
2. `<s, q> = s₁ q₁ + Σ_{j>1} s_j q_j`, and
   `E[sign(s₁) s_j] = 0` for `j > 1` (odd symmetry of the Gaussian).
3. `E[sign(s₁) s₁] = E[|s₁|] = √(2/π)` — the half-normal mean.

So each row contributes `E[sign(<s,r>) <s,q>] = √(2/π) · q₁ = √(2/π) · <q, r>/‖r‖`.
Averaging `M` i.i.d. rows and multiplying by `‖r‖ · √(π/2)/M` telescopes the
constant exactly to `<q, r>`. ∎

**Variance.** Per row, `Var[sign(<s,r>)<s,q>] ≈ ‖q‖² (1 − (2/π)cos²θ)`; summing
and rescaling gives the estimator standard deviation
`σ ≈ √(π/2) · ‖r‖ ‖q‖ / √M`. Empirically, at `D=128`, `M=64` the per-sample
sketch noise in attention units (`1/√D` scaling, unit queries) measures
`0.118` against the theory value `0.123`
([#p12_qjl_mse_ablation.csv:mse] at `m_bits=64` versus the
`(π/2)·D/M` scaling of the ladder rows).

**Why the correction matters.** The symmetric INT2 grid `{−s, 0, +s}` with
`s = max|k|` collapses most channels to zero, so `k̂` *shrinks* `k`. For an
attention-like aligned query `q ∥ k`, `E[<q, r>]` is systematically negative:
measured naive logit bias `−0.1735` in attention units
([#p12_qjl_mse_ablation.csv:bias_mean], aligned regime). The QJL term restores
it: empirical bias `+0.00277 < 0.01` at `M = 64`
([#p12_qjl_mse_ablation.csv:bias_mean]) with MSE ratio `0.2665 < 0.35`
versus naive ([#p12_qjl_mse_ablation.csv:mse_ratio_vs_naive]). MSE falls as
`1/M` across the pre-declared ladder
([#p12_qjl_mse_ablation.csv:mse]).

**Why Gaussian and NOT Rademacher.** The proof's first step — "rotate the
basis so `r = ‖r‖e₁`" — is legal *only* because `N(0, I_D)` is invariant under
general rotations. A Rademacher row `s ∈ {±1}^D` is invariant only under
coordinate permutations and coordinate sign flips: for a generic direction the
distribution of `<s, r>` depends on *where* `r`'s mass sits (its sparsity
pattern), not merely on `‖r‖`. Concretely, if `r = e₁` then
`E[sign(s₁)<s,q>] = q₁` (no `√(2/π)`), while for a dense `r` the CLT makes
`<s, r>` approximately Gaussian and the constant drifts back toward
`√(2/π)` — the factor is *direction-dependent*, so no single scale makes the
estimator exactly unbiased for every residual. The committed ablation is
honest about the practical consequence: on this fixture's *dense* Gaussian
residuals the Rademacher variant tracks the Gaussian closely
(bias `0.001482` vs `0.002765`, MSE ratio `0.2675` vs `0.2665` at `M=64`
([#p12_qjl_mse_ablation.csv:bias_mean], [#p12_qjl_mse_ablation.csv:mse_ratio_vs_naive]))
— an approximation, not a proof. The Gaussian projection is mandated because
the identity is *exact for every residual, sparse or dense*, which is what an
unbiasedness contract requires; Rademacher offers no such guarantee and its
advantage is only a cheaper multiply.

**Honest boundary condition.** QJL's variance is real. When the query is
*independent* of the key (naive residual already mean-zero in inner product),
the sketch noise dominates and the MSE ratio at `M=64` is `2.914 ≈ (π/2)·D/M`,
crossing below `1.0` only at `M ≥ 128` and below the `0.35` gate at
`M ≈ 512` ([#p12_qjl_mse_ablation.csv:mse_ratio_vs_naive], independent regime).
INT2+QJL is therefore a *bias-removal* tool for aligned query regimes, not a
free lunch.

---

## Packing contract (prototype)

Per key vector, `Int2QJLPayload` stores: packed INT2 crumbs
(`4/byte`, `quantizer.pack_int2`, layout-pinned), the base-grid `QuantParams`
(scale), `M = 64` sign bits in two `uint32` words
(`qjl_sketch.pack_qjl_signs`), and the FP16 `‖r‖₂`. The residual norm's FP16
rounding is the only systematic deviation from exact unbiasedness in the
reference estimator; it is below the `0.01` gate at FP16 resolution
([#p12_qjl_mse_ablation.csv:bias_mean]).

Why this is a *reference prototype* and not the production path: see ADR-010 —
the fused Split-K kernel stays INT4/INT8-only; 2-bit PTX unpack paths would
add fragile divergence to a memory-bound loop that currently has none.
