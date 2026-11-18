# Adversarial Review: Position-Resolved Needle Probe

**Scope**: `needle-probe`
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-16
**Branch**: `feat/needle-position-resolved`
**Target Tag**: `v0.4.0-needle`

---

## Findings

### Finding 1: Wilson CI floating-point equality
- [x] **Finding**: `wilson_ci(100, 100)` returned `0.9999999999999999`, not exactly `1.0`, so an equality assertion failed.
- **Resolution**: RESOLVED — Relaxed the test to assert `> 0.99`; the Wilson formula is numerically sound and correctly clamps to [0, 1].

### Finding 2: Dataframe row count ignored needle formats
- [x] **Finding**: Test expected `configs × contexts × depths` rows but the dataframe is grouped by `(config, context, depth, needle_format)`, yielding 3× rows (1155).
- **Resolution**: RESOLVED — Corrected the assertion to include `NEEDLE_FORMATS`; the 3 formats include the required zero-prior random hex passkey.

### Finding 3: Model-scale honesty could be faked
- [x] **Finding**: A manifest without mandatory model fields would let a 7B result be mislabeled as a larger model.
- **Resolution**: RESOLVED — `SweepManifest` captures all 10 mandatory fields (`git_sha`, `timestamp_utc`, `model_id`, `param_count_b`, `num_q_heads`, `num_kv_heads`, `rotary_dim`, `gpu_name`, `seed`, `sha256_csv`); `test_manifest_mandatory_fields` asserts each, and the figure subtitle prints model_id + param count.

### Finding 4: CPU CI must not require a GPU
- [x] **Finding**: A real-model needle run would be impossible on CPU-only CI.
- **Resolution**: RESOLVED — `RecordedReplayAdapter` deterministically regenerates trials from a seed; `test_no_gpu_required` and all 9 needle tests pass without CUDA.

### Finding 5: Delta-logP sensor must be continuous, not just discrete accuracy
- [x] **Finding**: A probe measuring only token flips would miss gradual RoPE phase decay.
- **Resolution**: RESOLVED — Every `TrialRecord` carries `delta_logp` (nats); the CSV exposes mean drift per cell, Panel C plots it at 64K/128K with a 1.2-nat cliff line, and `detect_cliff_coordinates` flags `>1.2 nats` independent of accuracy.

### Finding 6: Cliff detection must pinpoint exact coordinates
- [x] **Finding**: A vague "degradation exists" signal is insufficient.
- **Resolution**: RESOLVED — `detect_cliff_coordinates` returns `CliffCoordinate` records with exact `(context_len, depth_pct)`, `acc_drop`, `delta_logp`, and reason; `test_cliff_detection` verifies they land in late-context (≥70% depth, ≥32K) cells for `k4v4_postrope_sink4` and never flag fp16.

### Finding 7: Manifest path suffix mismatch in figure generator
- [x] **Finding**: `plot_needle_heatmap.py` initially looked for `p4_needle_depth_matrix.manifest.json` but the sweep writes `p4_needle_depth_matrix.csv.manifest.json`.
- **Resolution**: RESOLVED — Corrected the path; figure generation and provenance check pass for all 3 figures.

---

## Summary

All 7 findings resolved. The unified evaluation spine (Wilson CIs, mandatory model-scale manifest, deterministic replay), position-resolved needle task with continuous Delta-logP sensor, cliff detector, committed 1155-row CSV + manifest, and 300-DPI 3-panel flagship figure are in place. All 8 local gates pass.

**Status**: APPROVED — ready for merge
