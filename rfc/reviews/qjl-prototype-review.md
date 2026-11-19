# Adversarial Review: Gaussian QJL Residual Sketch Prototype (v1.3.0)

**Scope**: `qjl-prototype` — `kv_quant/qjl_sketch.py` (Gaussian reference
API), `tests/test_qjl_sketch.py` (3-way gate), `bench/qjl_mse_ablation.py`,
`bench/data/p12_qjl_mse_ablation.csv` (+ manifest),
`docs/HADAMARD_AND_QJL_MATH.md`, ADR-010.
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-19
**Branch**: `feat/qjl-gaussian-sketch-prototype`
**Target Tag**: `v1.3.0-qjl-prototype`

**Method**: re-derived the half-normal identity by hand before reading the
implementation, numerically probed the estimator's variance against theory,
re-executed the ablation for determinism, recomputed every gate statistic
from the committed CSV, and specifically hunted for the failure mode this
milestone invites: a beautiful theorem whose empirical section quietly
contradicts it.

---

## Findings

### Finding 1: My own draft overstated the Rademacher empirical case (severity: HIGH)
- [x] **Finding**: The first draft of `HADAMARD_AND_QJL_MATH.md` and ADR-010
  claimed the ablation "demonstrates the Rademacher bias empirically". The
  committed measurement says otherwise: on this fixture's *dense Gaussian*
  residuals, the Rademacher variant with the same `√(π/2)` constant lands at
  bias `0.001482` (aligned) and MSE ratio `0.2675` — statistically
  indistinguishable from the Gaussian rows (`0.002765`, `0.2665`)
  ([#p12_qjl_mse_ablation.csv:bias_mean], [#p12_qjl_mse_ablation.csv:mse_ratio_vs_naive]).
  The mechanism is the CLT: for a dense residual, `<±1 row, r>` is
  approximately Gaussian, so the arcsine constant approximately holds. The
  *theorem's* claim — the constant is direction-dependent and exact only under
  rotational invariance — remains true (a counterexample: `r = e₁` gives factor
  `1`, not `√(2/π)`), but this fixture cannot exhibit it.
- **Resolution**: RESOLVED — both documents were rewritten before commit to
  state the exact result: the identity is exact for Gaussian and *unproven*
  for Rademacher; the ablation honestly records that on dense residuals the
  approximation is close, which is the boundary of the claim, not evidence
  for it. The Rademacher row stays committed precisely so a reader can see
  this. This is the review's most important finding: the draft would have
  shipped an empirical claim its own data refutes.

### Finding 2: The mandated D=128 does not exist in the fixture (severity: MEDIUM)
- [x] **Finding**: `micro_kv_2k.pt` has head_dim `64`. The mandated "1,000
  high-dimensional attention heads (D=128)" cannot be read off directly;
  silently using D=64 would have changed every threshold's meaning.
- **Resolution**: RESOLVED — keys are formed by seeded pairing of two KV-head
  channel blocks into D=128 vectors (seed `2026`, construction committed in
  the manifest and mirrored in the test helper). Disclosed in the manifest's
  `construction` field; the test asserts against the same construction.

### Finding 3: A naive reading would declare Gate 3 impossible (severity: MEDIUM)
- [x] **Finding**: The independent-query regime makes the QJL sketch look
  useless — MSE ratio `2.914` at `M=64`, matching the theory
  `(π/2)·D/M = 3.14` ([#p12_qjl_mse_ablation.csv:mse_ratio_vs_naive]). A
  reviewer who only ran random queries could conclude the mandate's
  `0.35×` target is unreachable, and a dishonest one would then quietly
  switch regimes.
- **Resolution**: RESOLVED — the regime dependence is the physics, not a bug:
  QJL exists to remove the *systematic shrinkage bias* of the INT2 grid, which
  exists only when the query is correlated with the key (the attention-like
  aligned regime, naive bias `−0.173524`
  [#p12_qjl_mse_ablation.csv:bias_mean]). Both regimes are committed, the
  gate asserts the aligned regime (which is the one the mandate's own bias
  language describes), and the boundary is stated in ADR-010 and the math note.

### Finding 4: Runtime circular-import risk in the new API (severity: MEDIUM)
- [x] **Finding**: `quantizer.py` already imports *from* `qjl_sketch.py`
  (phase-1 legacy), so a top-level `from kv_quant.quantizer import ...` in the
  new code would create an import cycle.
- **Resolution**: RESOLVED — runtime imports are function-local (allowed by
  the configured `PLC0415` ignore, with a comment), and the type-only imports
  live under `TYPE_CHECKING` with `from __future__ import annotations` so
  `mypy --strict` still resolves them. Verified: module imports cleanly and
  strict typecheck passes.

### Finding 5: The estimator's per-sample noise exceeds the naive residual it corrects (severity: LOW)
- [x] **Finding**: At `M=64`, `D=128`, unit queries, attention scaling: the
  sketch noise std is `≈ √(π/2)·‖r‖/√M/√D ≈ 0.118–0.123`, i.e. the QJL
  correction of a single logit is *noisier than the bias it removes*
  (`0.1735`). The gate passes only because MSE (not per-sample error) is the
  metric and the correction removes the systematic part.
- **Resolution**: RESOLVED (documented, no change) — this is inherent to 1-bit
  sketching and is stated with the variance formula in the math note; the
  `1/M` ladder (measured: MSE `0.013299 → 0.000467` from M=64 to M=2048,
  [#p12_qjl_mse_ablation.csv:mse]) shows the escape hatch.

### Finding 6: The `cfg` parameter is accepted but nearly ignored (severity: LOW)
- [x] **Finding**: `quantize_k_int2_qjl(k, cfg, ...)` takes a `QuantConfig`
  but uses it only to reject non-symmetric schemes — the mandated signature
  implies configuration the prototype does not have.
- **Resolution**: RESOLVED — the docstring says exactly that ("accepted for
  interface compatibility"), and the asym rejection is tested
  (`test_asym_scheme_rejected`). No hidden config semantics.

### Finding 7: The legacy phase-1 QJL helpers coexist with the new contract (severity: LOW)
- [x] **Finding**: `qjl_sketch.py` now contains the legacy Rademacher-default
  helpers (kept byte-stable for `p1_*` reproduction) *and* the Gaussian
  reference API. A new caller importing `compute_qjl_sketch` gets the
  unproven constant by default.
- **Resolution**: RESOLVED (documented boundary, no code change) — the
  reference-contract section header states new code must use the Gaussian
  API; ADR-010 names the Rademacher rejection; changing legacy signatures
  would break p1 byte-stable reproduction, which the repo treats as
  stronger than convenience.

---

## Empirical checks performed

1. **Theory vs measurement**: estimator std from a 500-sample independent
   probe: `0.11797` measured vs `0.1226` from
   `√(π/2)·‖r‖/√M/√D` — the arcsine/half-normal law holds to ~4%.
2. **Gate recomputation from the committed CSV** (not from prose):
   - aligned QJL bias `+0.002765 < 0.01` ✓; naive bias `-0.173524`, |bias| > 0.1 ✓;
   - aligned MSE ratio `0.2665 < 0.35` ✓;
   - MSE ladder decays as `1/M` (`0.013299, 0.007028, 0.003518, 0.001705,
     0.000911, 0.000467`) ✓;
   - independent-regime ratio at M=64 `2.914 ≈ (π/2)·D/M = 3.14` ✓, gate
     passed at M=512 (`0.3469`) ✓.
3. **Test suite**: `tests/test_qjl_sketch.py` — 10 tests, all passing,
   including 1000-vector statistical gates and the `1/M` shrink assertion
   (M=256 at least `2.5×` below M=64).
4. **Determinism**: ablation re-run produced a byte-identical CSV
   (line-ending-normalized SHA256 prefix `7045425f3e8336f1` before and
   after); the manifest's recorded hash matches the file on disk.
5. **Claims gate**: `264` claims total including the new p12 citations
   (`bias_mean`, `mse`, `mse_ratio_vs_naive`), zero `[FAIL]`.
6. **Layout contract untouched**: golden hash still `913cd248…`; INT2 packing
   uses the existing `quantizer` path (no new bit layout).

## Verdict

The prototype delivers what the mandate actually requires: an exactly
unbiased Gaussian QJL estimator, layout-pinned INT2 packing, deterministic
seeded sketches, and statistical gates that pass with committed evidence —
plus two honest boundaries (dense-residual Rademacher closeness;
independent-regime variance dominance) that the first draft got wrong and
this review forced into the open before shipping. Approved for landing at
`v1.3.0-qjl-prototype` with the reference-vs-production scoping stated in
ADR-010.
