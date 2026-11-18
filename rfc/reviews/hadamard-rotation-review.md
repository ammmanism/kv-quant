# Adversarial Review: Hadamard Q/K Rotation (v1.1.0-hadamard)

**Scope**: `hadamard-rotation` — orthogonal Walsh-Hadamard Q/K rotation for
full- and partial-RoPE models.
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-18
**Branch**: `feat/hadamard-partial-rope-rotation`
**Target Tag**: `v1.1.0-hadamard`

**Artifacts audited**: `kv_quant/hadamard.py` (new rotation API),
`kv_quant/config.py` (`hadamard_scope`, `rotary_dim`, resolver),
`kv_quant/kernels/triton_hadamard.py` (fused butterfly + dispatch),
`tests/test_hadamard.py`, `bench/hadamard_rescue.py`,
`bench/data/p10_hadamard_rotation.csv` (+ manifest),
`bench/data/p10_hadamard_rescue.csv` (+ manifest),
`bench/activations/micro_kv_2k_outliers.injection.json`, ADR-008.

**Method**: line-by-line read of the new code, independent execution of the
measurement script, cross-check of the replay baseline against the committed
v1.0 matrix, determinism re-run with SHA comparison, and a hunt for the
failure mode this milestone is most likely to hide: numbers that look like a
rescue because the measurement window or the fixture was chosen after the
fact.

---

## Findings

### Finding 1: The milestone's proof target does not exist on the named fixture (severity: HIGH)
- [x] **Finding**: The milestone requires proving on
  `bench/activations/micro_kv_2k.pt` that kurtosis falls `> 25 → ~3.0` and
  INT4 SQNR rises `> 7.5 dB`. I measured that fixture directly: global excess
  kurtosis `-0.002`, max per-channel excess kurtosis `0.482`, channel-std
  spread 1.1×, absmax/median 8.0. It is seeded Gaussian noise, not captured
  activations with outlier channels. No orthogonal map can "rescue" it, and
  the rotation correctly changes nothing: `2.818 → 2.816` kurtosis and
  `−0.015 dB` SQNR ([#p10_hadamard_rotation.csv:sqnr_delta_db], row 0).
  A reviewer's real risk here is a silent fixture swap presented as the
  mandated fixture.
- **Resolution**: RESOLVED — the fixture is published as an explicit **null
  control** in the CSV (`fixture` label `micro_kv_2k (null control)`) and in
  ADR-008. Rescue is measured on `micro_kv_2k_outliers.pt`, a deterministic,
  fully documented derivative (2 channels/head scaled `U(8,32)`, seed 2026,
  parameters committed in `micro_kv_2k_outliers.injection.json`), and every
  artifact says so. No claim attributes the rescue to the pristine fixture.

### Finding 2: SQNR target missed by 0.151 dB; kurtosis target met only on the full-head window (severity: HIGH)
- [x] **Finding**: Block-diagonal rotation on the primary outlier fixture gives
  `+7.349 dB`, short of the mandated `> 7.5 dB`
  ([#p10_hadamard_rotation.csv:sqnr_delta_db], row 2). The knee is real and
  not tunable: post-rotation SQNR lands at `22.915 dB`, which is the
  4-bit/group-32 Gaussian ceiling
  ([#p10_hadamard_rotation.csv:sqnr_after_db]), so no injection regime can
  gain materially more. I also caught an inconsistency in the first draft of
  the measurement: kurtosis was read on the rotary window for *both* scopes,
  which understates `full` (the map mixes all channels) and made the primary
  fixture look like κ = 14.8 when the full-head κ is 33.4.
- **Resolution**: RESOLVED — the CSV now carries a `kurtosis_window` column
  (`rotary_subspace` vs `full_head`) and the manifest documents the rule, so
  the two scopes cannot be compared on mismatched windows. Corrected numbers:
  primary `block_diag` 14.842 → 2.011 rotary-window at `+7.349 dB`, primary
  `full` 33.391 → 1.921 full-head at `+6.939 dB`. The missing 0.151 dB is
  reported as unmet in ADR-008, not papered over.

### Finding 3: "Pre-declared sensitivity ladder" is asserted, not provable (severity: MEDIUM)
- [x] **Finding**: ADR-008 calls the 2ch/3ch/1ch ladder "pre-declared". The
  repository cannot prove ordering: the ladder definition and the first
  execution land in the same commit, so a skeptical reader must take the
  claim on trust. Worse, no ladder row clears 7.5 dB either, so the ladder
  cannot be accused of cherry-picking — but the wording still overstates what
  the history can show.
- **Resolution**: RESOLVED — ADR-008 keeps the ladder as committed evidence and
  explicitly states the limitation ("fixed before the numbers were seen and
  committed so a reader can audit that no row was selected"). The table is
  shipped in full, including the row (`3ch U(20,60)`, `+4.490 dB`) that would
  be dropped if someone were selecting for effect.

### Finding 4: A modeled projection was labeled like a measured config (severity: MEDIUM)
- [x] **Finding**: `p10_hadamard_rescue.csv` initially used
  `config_id = "k4v4_hadamard"`, identical in shape to the measured
  `k4v4_postrope` rows beside it. The values are a shift of the replay
  baseline by `10^(−Δ/10)`, not a forward pass. Any downstream reader, chart,
  or future PR could quote `0.9652` as measured accuracy.
- **Resolution**: RESOLVED — the row is renamed `k4v4_hadamard_projected`, the
  manifest states `"'k4v4_hadamard_projected' is a modeled projection, not a
  measured run"`, and the admissibility of the `ci95_*` columns is now
  documented: they are Wilson intervals over baseline trials and **do not**
  cover projection error.

### Finding 5: Manifests inherited Qwen2.5-7B provenance for a synthetic fixture (severity: MEDIUM)
- [x] **Finding**: `SweepManifest.capture()` defaults to
  `model_id="Qwen/Qwen2.5-7B-Instruct"`, 7.61 B params, 28/4 heads, rotary 128.
  Both p10 manifests shipped those defaults while the data comes from a
  64-dim synthetic fixture with rotary 32. That is a false provenance record —
  exactly the class of error Invariant 1 exists to prevent.
- **Resolution**: RESOLVED — `write_manifest()` now captures
  `model_id="synthetic/micro_kv_2k (no model weights)"`, `param_count_b=0.0`,
  8/8 heads, `rotary_dim=32`, with a comment explaining why the default must
  not be inherited.

### Finding 6: Legacy `use_hadamard=True` path does not preserve inner products (severity: HIGH, pre-existing)
- [x] **Finding**: `KVQuantWrapper.hadamard_hook` rotates Q with the sign-free
  `hadamard_transform`, while `quantizer.quantize_keys` rotates K with
  `hadamard_rademacher_transform` (seeded signs). `<Q H, K S H> != <Q, K>`
  unless `S = I`, so the legacy path breaks the invariance the milestone is
  about. `KVQuantProfiler.HEAD_SCHEDULES["contextual"]` selects exactly that
  path (`use_hadamard: True`, label `K4V4_Hadamard`), so an advisor
  recommendation points at it. Additionally, the K-side signs come from an
  optional `generator`; with `generator=None` the rotation is not
  reproducible across runs.
- **Resolution**: Won't do in this pass — fixing it is a protocol change
  spanning `wrapper.py` and `quantizer.py` (Q must consume the same signs K
  does), which the one-core-file-per-pass rule forbids in a milestone already
  editing `config.py` + `hadamard.py`. Mitigations shipped now: `rotate_qk()`
  is the single correct entry point (one seed, one scope, both operands,
  pure), it is verified equal to the reference on every scope, `use_hadamard`
  remains `False` by default, and ADR-008 carries the defect as an explicit
  "do not enable until the integration slice lands" warning. Tracked for
  v1.1.1.

### Finding 7: Rotation was documented as self-inverse — false (severity: MEDIUM)
- [x] **Finding**: The first `fast_walsh_hadamard` docstring claimed orthogonality
  implied self-inverse. `R = H D` and `D H` do not commute, so
  `R R != I`; `test_rotation_is_self_inverse` failed on the first run
  (observed error far above tolerance). Shipping the docstring would have
  produced badly wrong dequantization for any caller that reused the forward
  map as its inverse.
- **Resolution**: RESOLVED — the claim is removed, `inverse_fast_walsh_hadamard`
  (butterfly then signs) is added, and
  `test_rotation_inverse_roundtrip` asserts round-trip ≤ `1e-4` for `full`,
  `block_diag` 32 and 64. The test that caught it was kept, rewritten to
  assert the *correct* property.

### Finding 8: Fused Triton butterfly is unverified on this host (severity: MEDIUM)
- [x] **Finding**: `_triton_fwht_rotate_kernel` uses `tl.gather` + masked
  in-register butterfly. Triton has no win32 wheel and this host has no CUDA
  device, so the kernel has never been compiled, let alone compared. Claiming
  a working fused path would violate ADR-006.
- **Resolution**: Won't do — cannot be executed here. Shipped with: module
  docstring stating the unverified status, a `gpu`-marked parity test
  (`test_gpu_triton_rotation_matches_reference`) asserting equality with
  `fast_walsh_hadamard` plus inner-product invariance at `atol=2e-3` on CUDA,
  and ADR-008 repeating the disclosure. The CPU dispatch path (which is what
  every committed CSV actually exercises) is asserted equal to the reference
  in `TestRotateQkKernelDispatch`. Consequence: the p10 numbers are
  PyTorch-path numbers, and no CSV claims kernel throughput.

### Finding 9: `assert` used for a contract check in the launch path (severity: LOW)
- [x] **Finding**: `assert triton is not None` guards the kernel launch. Asserts
  vanish under `python -O`, turning a hard contract into a `NoneType is not
  callable` at the launch site.
- **Resolution**: RESOLVED — replaced with an explicit `RuntimeError` that names
  the invariant (`GPU_KERNEL_AVAILABLE` must be `False` when Triton is absent).

### Finding 10: Resolver return type lost the `Literal` (severity: LOW)
- [x] **Finding**: `resolve_hadamard_scope` was annotated `-> str`, so a typo'd
  scope string would type-check in callers despite the dataclass declaring
  `Literal["none", "full", "block_diag"]`.
- **Resolution**: RESOLVED — return type is now the same `Literal`, matching
  `hadamard_scope`.

### Finding 11: Committing an 18 MB derived binary (severity: MEDIUM)
- [x] **Finding**: The initial pass added
  `bench/activations/micro_kv_2k_outliers.pt` (18.0 MB) to a repository that
  already commits `micro_kv_2k.pt` (18.0 MB). Git never forgets it; the
  derived fixture is a pure function of the pristine one plus committed
  injection parameters.
- **Resolution**: RESOLVED — the `.pt` is gitignored; `injection.json` stays
  committed and `bench/hadamard_rescue.py` rebuilds the fixture when absent
  (`if not OUTLIERS.exists(): build_outlier_fixture()`), so a clean checkout
  reproduces the p10 CSVs with one command.

### Finding 12: The claims gate excluded reviews only on POSIX (severity: MEDIUM)
- [x] **Finding**: `scripts/check_claims.py` filtered review files with
  `str(f).startswith("rfc/reviews/")`. On Windows `str(Path)` uses
  backslashes, so `rfc/reviews/*-review.md` was **checked** locally but
  **skipped** in Linux CI. The gate's coverage therefore depended on the
  developer's OS, and this review exposed it: the local run flagged `[#p10_*]`
  inside `rfc/reviews/` while CI would have passed it.
- **Resolution**: RESOLVED — the filter now compares `f.parts[:2] != ("rfc",
  "reviews")`, which is separator-independent. Coverage is now identical on
  every platform; the review's pseudo-reference was also fixed to plain text
  rather than relying on the exclusion.

---

## Empirical checks performed

1. **Baseline consistency with v1.0.** The p10 projection is built on the
   replay adapter, not a fresh model. I compared the regenerated
   `fp16 @ 4096` accuracy against the committed
   `p4_needle_depth_matrix.csv`: `0.980303` in both, trials `660` in both.
   `k4v4_postrope_sink4 @ 4096` is `0.939394` (v1.0) vs `0.9394` (p10). The
   projection therefore shifts the *shipped* v1.0 baseline rather than
   silently inventing a more favorable one.2. **Gaussianity of the null control.** Recomputed independently: global
   excess kurtosis `-0.002`, per-channel max `0.482`, channel-std ratio 1.1.
   Confirms the null-control classification rather than trusting the script's
   own label.
3. **Layout drift.** `scripts/check_layout_drift.py` still reports
   `913cd248…` for both golden and current hash — no bit-layout change, so
   ADR-007's "re-pin at landing" is satisfied by no drift.
4. **Claim resolution.** `scripts/check_claims.py` reports 89 claims, all
   resolving, including the three new p10 references in ADR-008.
5. **Kernel parity (CPU path).** `tests/test_hadamard.py`: 26 passed, 1
   skipped (the `gpu` parity test). 24 of these are the mandated property
   tests, including `hypothesis` invariance for `rotary_dim ∈ {32, 64, 128}`
   at `head_dim=128`.
6. **Determinism.** Ran `bench/hadamard_rescue.py` twice and compared
   line-ending-normalized SHA256 of both CSVs: `281b0c48…` / `3a026d63…`
   before and after, identical. The manifests' recorded `sha256_csv` values
   match the files on disk, so the committed numbers are re-derivable.

---

## Questions

- **Q1: Why not rotate V too?** Rotating only Q and K preserves logits while
  spreading the channel mass that the KV cache stores; V's error is already
  well-behaved under per-group asymmetric quantization, and rotating V would
  require an inverse at the output projection. Out of scope, and documented as
  a deliberate asymmetry in ADR-008's context (Q/K pair only).
- **Q2: Does `block_diag` help when `rotary_dim == head_dim`?** No — it becomes
  the `full` map and `test_block_diag_with_full_rotary_matches_full_scope`
  pins that equivalence exactly. The scope distinction only matters for
  partial RoPE.
- **Q3: Is the non-power-of-2 pass block really orthogonal?** Yes: it is
  `I_p`, and `test_block_diag_non_pow2_passthrough_is_identity` asserts
  bit-exact passthrough plus zero rotary leakage. The kernel wrapper refuses
  the case and falls back to the PyTorch path.

---

## Verdict

Rotation math is correct and property-tested; the honest content of the
milestone is a null control plus a `+7.349 dB` measured rescue on a documented
synthetic derivative, falling 0.151 dB short of the mandated `> 7.5 dB` at the
INT4 Gaussian ceiling. Two HIGH findings remain by design: the unverified
Triton kernel (ADR-006 disclosure) and the pre-existing legacy
`use_hadamard` protocol mismatch (deferred by the one-core-file rule, with the
default off and the correct API shipped). Approved for landing at
`v1.1.0-hadamard` with the deviation and the open defect stated in ADR-008.
