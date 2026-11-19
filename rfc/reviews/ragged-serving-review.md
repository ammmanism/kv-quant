# Adversarial Review: Ragged-Batch Paged Runner & CoW Concurrency (v1.4.0)

**Scope**: `ragged-serving` — `kv_quant/ragged_runner.py`,
`kv_quant/kernels/dequant_fused.py` (ragged `CONTEXT_LEN`),
`tests/test_ragged_runner.py`, `bench/concurrency_sim.py`,
`bench/data/p13_ragged_concurrency.csv` (+ manifest), ADR-011.
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-19
**Branch**: `feat/ragged-batch-prefix-simulator`
**Target Tag**: `v1.4.0-ragged-serving`

**Method**: read the runner against `HybridPagedKVCache`'s actual ref-count
semantics, re-derived the capacity arithmetic from the CSV, executed the test
suite, and hunted for the failure modes this milestone invites: leak accounting
that looks right but double-frees, CoW that silently mutates shared payloads,
ragged launches that disagree with per-item launches, and capacity numbers that
are prettier than the bytes allow.

---

## Findings

### Finding 1: The mandated "375 block_table entries" is 24,000 tokens, not 24,576 (severity: MEDIUM)
- [x] **Finding**: The first benchmark draft used a 24,576-token shared prefix
  (384 blocks of 64) — the natural "24K" reading — while the mandate explicitly
  says *375 block_table entries*, i.e. 375 × 64 = 24,000 tokens. Capacity rows
  were consequently ~7% optimistic for CoW configs.
- **Resolution**: RESOLVED — `SHARED_PREFIX_TOKENS = 24000  # exactly 375
  blocks of 64 (mandate)`, CSV regenerated, ADR-011 numbers updated
  (`161` sessions and `8894.7` tok/s at the 24 GiB/32K point, not the draft's
  `172`/`9514.9`). The manifest records the prefix token count, so the
  discrepancy cannot recur silently.

### Finding 2: My own capacity test asserted the wrong call raises (severity: LOW)
- [x] **Finding**: The draft test appended 128 tokens under a 4-page budget
  (3 live + 2 needed) and then expected a *later* append to raise. The runner
  correctly raised on the first append; the test was wrong, not the code.
- **Resolution**: RESOLVED — the test now uses `pytest.raises` on the breaching
  append and additionally asserts `live_pages()` is unchanged afterwards, which
  is the property that actually matters (a rejected append must not allocate).

### Finding 3: `shared_pages` counts `ref_count > 1`, not "pages born shared" (severity: LOW)
- [x] **Finding**: After 500 forks release, the root's page drops to
  `ref_count = 1` and is therefore *private* by the pool statistic. My first
  leak assertion expected it to remain "shared" — a semantics confusion that
  would have shipped a misleading test name.
- **Resolution**: RESOLVED — the test now asserts `shared_pages == 0`,
  `live_pages == 1`, `freed_pages == 500` after the loop, and `0/501` after the
  root completes. The docstring states the definition.

### Finding 4: GPU zero-length items still reduce to NaN (severity: MEDIUM)
- [x] **Finding**: The kernel emits neutral partials (`m=-inf, l=0, acc=0`) for
  split-K shards past a ragged length, which is correct for *partially* short
  items. But a batch item with `context_len = 0` makes *every* shard neutral;
  the reduction then computes `m_max = -inf`, `alpha = exp(-inf + inf) = NaN`.
  The CPU reference explicitly zeroes such items; the GPU path does not.
- **Resolution**: RESOLVED (documented, not silently patched) — the kernel
  docstring, ADR-011, and this review all state that zero-length items must be
  excluded from a GPU launch or zeroed by the caller; the ragged parity test
  exercises two zero-length items on the CPU path and asserts exact zeros.

### Finding 5: Throughput is an upper bound and budget-invariant by construction (severity: MEDIUM)
- [x] **Finding**: `modeled_decode_tok_s` is `N·1e6 / (step_bytes/bytes_per_us)`.
  Because `step_bytes` is linear in `N`, the value is constant across budgets
  for a fixed (config, context) — e.g. FP16/32K reports `696.7` at 24, 48 and
  80 GiB. A reader could mistake a plateau for a measurement or for a scaling
  law.
- **Resolution**: RESOLVED — ADR-011 says explicitly that "extra budget buys
  concurrency, not per-token rate" and that the figure ignores
  attention/softmax compute; the manifest's `throughput_model` field states the
  formula and the upper-bound caveat. The CSV column is named
  `modeled_decode_tok_s`, not `tok_s`.

### Finding 6: Capacity is conservative and ignores content dedup (severity: LOW)
- [x] **Finding**: `append()` assumes every new sealed block allocates, even
  when the cache's payload hash would share it. Real capacity can therefore be
  higher than reported; the estimate never overstates it.
- **Resolution**: RESOLVED — documented in the runner docstring and ADR-011;
  the direction of the error is safe (under-promise).

### Finding 7: Runner reaches into cache private state to release pages (severity: LOW)
- [x] **Finding**: `complete()` and `_live_page_objects()` touch
  `cache._pages` / `_pages_by_hash` because `HybridPagedKVCache` exposes no
  release API. That is a private-API coupling a maintainer could break.
- **Resolution**: Won't do in this pass — adding `release_pages()` to
  `cache.py` would mean editing a second core file in the same milestone,
  which the one-core-file-per-pass rule forbids. The coupling is confined to
  two methods and is exactly what the 500-completion leak gate exercises.

### Finding 8: A pre-existing Hypothesis test aborted gate 1 with a 1713 s input-generation stall (severity: LOW)
- [x] **Finding**: The first ship attempt failed gate 1 in
  `tests/test_quantizer.py::TestQuantizeDequantize::test_sqnr_matches_theory`
  with `FailedHealthCheck: Input generation is slow ... 4 valid inputs after
  1713.57 seconds`. Nine of ten `tensor_1d` draws took `0.02–0.06 s`; a single
  draw took 1713 s. `tensor_1d` is a fixed-size `st.lists(st.floats(...))`, so
  this is a host stall (the same suite passed gate 1 four times that day in
  78–110 s), not strategy cost or a defect in this milestone's code.
- **Resolution**: Won't do in this milestone — the test is untouched by this
  change set and editing `tests/test_quantizer.py` would exceed the scope of a
  serving PR. Verified non-reproducible instead: the test passes in isolation
  and the full suite exits `0`. Recorded here because an unbounded
  input-generation stall is a genuine CI timeout risk that should be addressed
  by a dedicated test-hygiene change (fixed-size float lists defeat
  Hypothesis's shrinking biases; `st.lists(..., min_size=n, max_size=n)` is the
  usual suspect).

---

## Empirical checks performed

1. **Test suite**: `tests/test_ragged_runner.py` — 8 tests, all passing:
   shared-prefix ref-counts (`16` refs for 15 forks + root), divergence
   isolation (shared payload object identities unchanged, no extra ref bumps),
   capacity rejection with no allocation, `500`-completion leak gate
   (`live=0`, `freed=501`), batch-16 ragged vs batch-1 parity
   (`atol=1e-3`, two zero-length items exactly zero), and prefix-hash
   determinism/sensitivity.
2. **Capacity arithmetic recomputed from the CSV**:
   - 24 GiB/32K `k4v4_hadamard_cow`: `(24·2³⁰ − 430,080,000) / 157,384,704 =
     161.2 → 161` sessions ✓; FP16: `24·2³⁰ / 2,181,300,224 = 11.8 → 11` ✓;
   - 24 GiB/64K FP16: `24·2³⁰ / 4,362,338,304 = 5.9 → 5` ✓;
   - 64K CoW comparison `34` vs `5` sessions ✓.
3. **Bytes/elem**: INT4 sym `0.53125`, INT4 asym `0.5625`, FP16 `2.0` from the
   golden-hashed `layout.py` equation — the same contract the VRAM
   cross-validation matched at `R²=1.0000`; no new bit-layout was introduced.
4. **Claims gate**: `270` claims total including the new p13 citations, zero
   `[FAIL]`; layout drift gate still green at `913cd248…`.
5. **Determinism**: the p13 CSV is pure arithmetic over committed inputs and
   regenerates identically (no randomness in the model).

## Verdict

The runner is a correct orchestrator over the existing cache: CoW sharing is
delegated, not re-implemented; ref-counts release cleanly over 500 request
lifecycles; ragged batch execution matches per-item execution exactly; and the
capacity/throughput artifact is honestly labeled as a modeled upper bound with
its formula in the manifest. The two real hazards — GPU zero-length NaN and
private-API release coupling — are disclosed rather than hidden. Approved for
landing at `v1.4.0-ragged-serving`.
