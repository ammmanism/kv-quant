# Adversarial Review: Flagship README Overhaul (v1.0 + v1.2 synthesis)

**Scope**: `readme-overhaul` — `README.md` and all five flagship figure artifacts.
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-19
**Branch**: `docs/readme-flagship-overhaul`

**Ownership map respected**: `agents.planner` authored `README.md` exclusively;
`agents.harness` verified (did not regenerate) the PNGs and `.png.meta.json`
files; this review is confined to `rfc/reviews/**`. No core file was touched.

**Method**: I re-derived every table number from the committed CSVs with pandas,
ran the claims gate, ran an independent uncited-number scanner over the prose
(the claims gate only validates tags that are *present*, not tags that are
*missing*), checked all five image links and their provenance hashes, executed
the documented CLI command, and compared the mandated prompt numbers against
the committed artifacts instead of trusting either side.

---

## Findings

### Finding 1: A mandated source CSV does not exist (severity: MEDIUM)
- [x] **Finding**: The prompt lists `bench/data/p1_quantizer_sqnr.csv` as Step-1
  ground truth. No such file exists. The committed phase-1 SQNR artifact is
  `bench/data/p1_sqnr_error_bounds.csv` (90 rows; `sqnr_db`,
  `effective_bytes_per_elem`, `theoretical_sqnr_db` columns), and that is what
  the README cites.
- **Resolution**: RESOLVED — the README cites the actual committed file. Recorded
  here so a future prompt revision does not send an agent hunting a phantom
  artifact.

### Finding 2: The mandated `4.5625 B/elem` naive figure is not in any committed artifact (severity: HIGH)
- [x] **Finding**: The prompt requires the byte equation
  `b/8 + 2(1+I_asym)/G` and a `0.5625 B/elem` vs `4.5625 B/elem` comparison
  yielding `8.11×`. Checks: `0.5625` exists as
  `effective_bytes_per_elem` for row `4b_asym_normal_g64` in
  `p1_sqnr_error_bounds.csv`; **`4.5625` appears in no CSV, manifest, or doc**.
  `kv_quant/layout.py` (Invariant 3, single source of truth) defines the
  equation as `(b + I_qjl)/8 + 2(1+I_asym+I_qjl)/G`; for FP16 that yields
  `2.0625 B/elem`, not `4.5625`. The repo's `8.11×` comes from a *traffic*
  derivation (write-and-re-read of the FP16 materialization: `73 MB → 9 MB`
  in `p3_hbm_traffic_model.csv`), not from dividing `4.5625` by `0.5625`.
- **Resolution**: RESOLVED — README §3.3 prints the authoritative equation
  (including the QJL indicator), cites `0.5625 B/elem` where it actually lives,
  and derives `8.11×` from the p3 traffic columns. An explicit blockquote states
  that no `4.5625 B/elem` artifact exists and points to this review. Inventing
  the number to satisfy the prompt would have violated Invariant 1.

### Finding 3: The mandated `quantized_kv_context` API does not exist (severity: MEDIUM)
- [x] **Finding**: The prompt asks for a snippet `with quantized_kv_context(model, cfg):`.
  No such symbol exists in `kv_quant/`. The real context manager is
  `KVQuantWrapper(model, quant_config, cache_config)` — it implements
  `__enter__`/`__exit__` and patches attention projections; a one-argument
  convenience function would be a different API.
- **Resolution**: RESOLVED — the README quickstart uses the verified three-argument
  constructor and `with KVQuantWrapper(...) as wrapped:`, and says explicitly
  that no `quantized_kv_context` exists, rather than shipping a snippet that
  fails on copy-paste.

### Finding 4: Mixed model provenance could be read as one experiment (severity: HIGH)
- [x] **Finding**: Quality artifacts are Qwen2.5-7B-Instruct (`7.61 B`, `28/4`
  heads = GQA `7:1`, rotary `128`; p4/p6 manifests); kernel/latency artifacts
  are meta-llama/Llama-2-7b-hf with **no param count recorded in the manifest**;
  rescue projections are synthetic fixtures (`synthetic/micro_kv_2k`,
  `8` heads/rotary `32`; `synthetic/bimodal-calibration`, `8` heads/rotary `64`).
  A table that silently mixes them overstates the evidence.
- **Resolution**: RESOLVED — Table 1's preamble states the model and GQA ratio
  for every recommendation class and marks synthetic/projection rows; Table 3's
  caption names Llama-2-7b-hf and the CPU-only modeled mode; Table 2's caption
  names Qwen2.5-7B-Instruct/GQA 7:1; the hero caption states model, parameter
  count, head counts and rotary width; §3.4 labels the rescue curves as
  projections. I verified the manifest fields directly rather than trusting the
  prose.

### Finding 5: Model metadata in the hero caption has no CSV *column* to cite (severity: LOW)
- [x] **Finding**: `7.61 B`, `28/4` heads and `rotary_dim=128` live in
  `bench/data/p4_needle_depth_matrix.csv.manifest.json`, not in a CSV column.
  `check_claims.py` validates column existence, so a citation cannot mechanically
  prove those numbers.
- **Resolution**: RESOLVED — the caption cites the CSV for the data rows and
  explicitly names the manifest as the metadata source; I opened the manifest
  and confirmed every stated field. Honest pointer beats a misleading tag.

### Finding 6: `CITATION.cff` still pins v1.0.0 while the repo is at v1.2.0 (severity: LOW)
- [x] **Finding**: `CITATION.cff` declares `version: 1.0.0` and a v1.0.0 release
  URL. The README's BibTeX block reproduces that file faithfully.
- **Resolution**: Won't do in this pass — `CITATION.cff` is outside the
  planner's single owned file. The README says the citation "pins the v1.0.0
  release; later milestones are listed in `CHANGELOG.md`", so no reader is
  misled. Track a CFF refresh for the next release-managed milestone.

### Finding 7: Two committed sweeps give different INT4-K numbers and could be confused (severity: MEDIUM)
- [x] **Finding**: §3.1 quotes `0.8455` at 32K (p4 needle spine, sink-4
  post-RoPE) while §3.2 quotes `0.7715` at 32K (p6 v1-frontier sweep,
  `k4v4_post_rope_s4`). Both are real but they are different configurations and
  evaluation paths; juxtaposed without a caveat they look contradictory.
- **Resolution**: RESOLVED — §3.2 now carries an explicit scope-trap sentence
  with both citations, telling the reader not to mix the two evaluations.

### Finding 8: LongSum/LongCode have no 128K cells (severity: LOW)
- [x] **Finding**: Table 2 requests a `128K` column, but `p5_ecological_baseline.csv`
  measures those tasks only at 8K/32K/64K. Printing `0` would be a fabricated
  cliff; leaving a bare dash would be unexplained.
- **Resolution**: RESOLVED — dashes are used with an explicit "not measured at
  that context" note and a citation to the `context_len` column; 8K values are
  labeled inline instead of being smuggled into the 4K column.

### Finding 9: Uncited numbers in headers, badges and one honesty note (severity: MEDIUM)
- [x] **Finding**: An independent scan (numeric-token regex on non-code lines
  lacking a citation tag) flagged 17 lines. Sixteen are UI labels or section
  headers (badges "Python 3.11 & 3.12", `100% Verified`, figure alt text
  "4×4", "95% CI", table headers "4K/32K", section "3.4", etc.) and are not
  research claims; one was substantive — the blockquote about the absent
  `4.5625 B/elem` figure also stated `8.11×` uncited.
- **Resolution**: RESOLVED — the honesty note now cites the p3 reduction column;
  table headers 2 and 3 cite their model/context source; every prose line with a
  measured quantity carries at least one valid tag. Labels remain labels and are
  enumerated here so the decision is auditable rather than invisible.

### Finding 10: All five flagship figures verified; no regeneration needed (severity: INFO)
- [x] **Finding**: The prompt asked the harness to verify/regenerate five
  300-DPI PNGs. All five exist with valid metadata:
  `fig0_hero_cliff` 4469×1796, `fig1_needle_depth_heatmap` 5670×1631,
  `fig2_kv_asymmetry_contour` 4170×1631, `fig3_roofline_and_vram` 4168×1631,
  `fig4_hadamard_and_head_rescue` 4469×1618 — all exactly 300 DPI, all source
  CSV SHA256 hashes matching the current files, all image links resolving from
  the README. The checker also validates the two non-flagship figures.
- **Resolution**: RESOLVED — no regeneration was necessary; rewriting a figure
  would have changed its source hashes and invalidated the manifests for no
  reason. `scripts/check_figure_provenance.py` reports 7/7 OK.

---

## Empirical checks performed

1. **Claims gate**: `scripts/check_claims.py` — `156` claims total, `103` in
   `README.md`, zero `[FAIL]`; all references resolve to committed CSV columns.
2. **Uncited-number scan**: independent regex pass over all non-code README
   lines; 17 hits triaged, substantive ones fixed (Finding 9).
3. **Numeric recomputation from CSVs** (not copied from prose):
   - cliff gaps: `(0.9924−0.8455)/0.9924 = 14.80%` at 32K;
     `(0.9788−0.8682)/0.9788 = 11.30%` at 128K;
   - rescue gaps: `(0.9788−0.9591)/0.9788 = 2.01%` (Hadamard),
     `(0.9788−0.9530)/0.9788 = 2.63%` (head-adaptive);
   - K8V4 at 32K: `(0.9656−0.9596)/0.9656 = 0.62%`;
   - VRAM compression: `2.6494→2.65×`, `3.9537→3.95×`;
   - ITL: `6.907→6.91×`, `11.282→11.28×`, `12.131→12.13×`, `17.415→17.42×`;
   - ncu K4V4 @32K: naive `1,224,802,304 B`, fused `134,820,191 B`,
     ratio `9.085×`, L2 `1.22% → 10.75%`.
   Every value matches the published table to the last printed digit.
4. **Head-schedule counts**: from `p11_head_sensitivity_schedule.csv`, `24`
   K8V4 assignments of `96` heads (25%), and the declared synthetic ground
   truth of `20` retrieval-biased heads gives recall `20/20`, precision `20/24`
   — matching §3.4.
5. **CLI smoke**: executed the documented advisor command verbatim; it printed
   the `128000→131072` context normalization, recommended `K8V4`, `0.93%`
   drop and `12.13×` ITL — matching §5 and its citations.
6. **Figure provenance**: 7/7 figures OK; 5/5 flagship links present, 300 DPI,
   hashes current.

## Verdict

The README is a faithful synthesis of both acts: every measured or modeled
number is traceable to a committed CSV, every recommendation names its model
and GQA ratio, and the three prompt artifacts that do not exist in this
repository (`p1_quantizer_sqnr.csv`, `4.5625 B/elem`, `quantized_kv_context`)
are disclosed rather than silently substituted with fiction. The mandated
"within 0.3% of FP16" claim is stated as unmet, consistent with ADR-009.
Approved for commit at `docs(readme): overhaul README with full visual gallery,
benchmark tables, and two-act narrative`.
