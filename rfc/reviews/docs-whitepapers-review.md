# Adversarial Review: Whitepapers & ADR Synchronization (v1.2.2-docs)

**Scope**: `docs-whitepapers` — `docs/THE_TRADE.md`, `docs/KERNEL_NOTES.md`,
`docs/DECISIONS.md` (ADR-001 … ADR-009).
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-19
**Branch**: `docs/whitepapers-and-adrs-v1.2`

**Ownership**: `agents.planner` authored all three documents; this review is
confined to `rfc/reviews/**`. No core file, CSV, or figure was modified.

**Method**: read every committed CSV and manifest named in the prompt, ran the
claims gate, wrote an independent *paragraph-level* uncited-number scanner (the
claims gate only validates tags that exist, not missing ones), recomputed every
published table value from the CSVs with pandas, verified ADR template
completeness, and checked all figure links against `check_figure_provenance.py`.

---

## Findings

### Finding 1: The mandated `4.5625 B/elem` figure still has no committed source (severity: HIGH)
- [x] **Finding**: Section 1 of the prompt requires the byte account
  `naive = 4.5625 B/elem`, `fused = 0.5625 B/elem`, `8.11×`. I re-checked every
  CSV, manifest and the authoritative `kv_quant/layout.py`: `0.5625` is the
  `effective_bytes_per_elem` of `4b_asym_normal_g64`
  (`p1_sqnr_error_bounds.csv`); `4.5625` is absent everywhere, and layout.py
  evaluates FP16 + metadata to `2.0625`, not `4.5625`. The repo's `8.11×` is a
  *traffic* ratio (`naive_hbm_mb / fused_hbm_mb = 73 / 9`, `p3_hbm_traffic_model.csv`),
  equivalent to `37,376 / 4,608` bytes per token (write-and-re-read of the FP16
  materialization), not to `4.5625 / 0.5625`.
- **Resolution**: RESOLVED — `KERNEL_NOTES.md` §1 prints the authoritative
  equation (including the QJL indicator), derives the ratio from committed
  traffic columns, and carries a blockquote naming the absent figure with a
  citation to the derivation. No number was invented to satisfy the prompt.

### Finding 2: The "`< 0.5%` loss at 128K" mandate is contradicted by the committed projection (severity: HIGH)
- [x] **Finding**: The prompt asks Section 3 to claim the `4.5`-bit schedule
  loses `< 0.5%` at `128K`. `p11_head_rescue_curve.csv` gives
  `head_adaptive_4p5_hadamard = 0.9530` vs FP16 `0.9788` — a `2.6%` gap — and
  the adaptive curve is below pure Hadamard (`0.9591`). The structural cause is
  the replay adapter's flat ~`0.97` K8V4 baseline, which any K8-containing blend
  inherits.
- **Resolution**: RESOLVED — `THE_TRADE.md` §3.2 and ADR-009 both state the
  measured gap and the reason, with the lowest-`delta_logp` advantage presented
  as the real benefit. No "0.5%" claim appears in any of the three documents.

### Finding 3: ADR-007 was retitled; the old roadmap content had to be preserved (severity: MEDIUM)
- [x] **Finding**: The prompt assigns ADR-007 to the zero-reload wrapper, but the
  committed ADR-007 was the forward-looking "v2.0 Rescue Roadmap", cited by
  ADR-008/009 as "item 1/2". Blindly overwriting it would orphan those
  references and erase delivered history.
- **Resolution**: RESOLVED — ADR-007 now documents the wrapper
  (`KVQuantWrapper`), and the roadmap survives as an unnumbered "Roadmap —
  v1.1.0/v1.2.0 deliveries" section; ADR-008/009 say "implements roadmap item
  1/2". All internal references were checked; the only remaining old wording is
  in an immutable historical review (`hadamard-rotation-review.md`), which is
  correctly left untouched.

### Finding 4: Two more phantom prompt artifacts (severity: MEDIUM)
- [x] **Finding**: The prompt again names `p1_quantizer_sqnr.csv` (nonexistent —
  the committed artifact is `p1_sqnr_error_bounds.csv`) and
  `QuantizedAttentionWrapper` (nonexistent — the shipped class is
  `KVQuantWrapper`). Either could silently produce uncitable claims.
- **Resolution**: RESOLVED — all three documents cite the real CSV; ADR-007's
  title records both the spec name and the shipped name, and explicitly says
  the README documents the real API rather than a phantom symbol.

### Finding 5: Four invented citation targets caught by the claims gate (severity: MEDIUM)
- [x] **Finding**: My first draft cited `arch_compat_matrix.csv:int4_nibble_unpack`
  etc. — but those strings are *values* in the `feature` column, not columns —
  and `p4_needle_depth_matrix.csv:num_kv_heads`, which lives only in the
  manifest. `check_claims.py` rejected all four.
- **Resolution**: RESOLVED — replaced with `[<ch_compat_matrix.csv:feature]` and
  a `[<p4_needle_depth_matrix.csv:acc_mean]` pointer plus an explicit "manifest
  field" note. Recorded because this gate only checks column existence: a
  plausible-looking but wrong tag otherwise looks verified.

### Finding 6: Numeric prose spanning line breaks needed paragraph-level auditing (severity: LOW)
- [x] **Finding**: A line-level uncited-number scan produced 44/24/48 hits across
  the three docs, nearly all continuation lines of a sentence whose citation is
  on a later line, plus version labels and section headers. A naive "one tag per
  line" fix would have littered the documents with redundant tags.
- **Resolution**: RESOLVED — re-ran the scan at paragraph granularity: 4/5/15
  remaining blocks were titles, ADR status/tag lines, table intros and image alt
  text; the one substantive block (the `4.5625` honesty note) now carries a p3
  citation. Every paragraph with a measured quantity carries at least one valid
  tag, which is what the mandate actually requires.

### Finding 7: ADR-001's `2.4×` CoW claim is a maximum over five synthetic points (severity: LOW)
- [x] **Finding**: "up to `2.4×` memory amortization" reads like a general
  guarantee but is the maximum of `p2_cow_prefix_memory.csv:compression_factor`
  over five agent-count rows.
- **Resolution**: RESOLVED — ADR-001 says "up to" and cites that column; the
  THE_TRADE decision matrix uses separately computed byte-calculator ratios
  (`2.65×`, `3.95×`) with their own citations. No conflict, but the scope words
  matter and are preserved.

### Finding 8: LongCode value precision is non-monotonic — disclosed, not smoothed (severity: INFO)
- [x] **Finding**: At `64K`, K8V4 (`0.850`) beats K8V8 (`0.650`) on AST LongCode
  (`p5_ecological_baseline.csv:ast_exact_match`). A symmetric "more bits is
  always better" narrative would have been wrong.
- **Resolution**: RESOLVED — THE_TRADE §1 calls out the exception explicitly with
  the citation; the K/V asymmetry conclusion is retained because every INT4-K
  configuration still collapses.

---

## Empirical checks performed

1. **Claims gate**: `scripts/check_claims.py` — `245` claims total, including
   `65` in `THE_TRADE.md`, `36` in `KERNEL_NOTES.md`, `28` in `DECISIONS.md`;
   zero `[FAIL]` after Finding 5 was fixed.
2. **Table recomputation from CSVs**:
   - per-task K/V table @64K: needle `0.9624/0.9611/0.7738/0.7716`,
     multihop `0.9639/0.9594/0.7734/0.7699` (p6), LongSum
     `97.333/95.000/90.333/64.333%`, LongCode `0.650/0.850/0.700/0.550` (p5) —
     every cell matches the published table;
   - ncu @64K: k8v8 naive `2,717,974,528 B` vs fused `538,072,561 B`,
     reduction `5.051×`, L2 `5.68%`; k4v4 `2,449,539,072 B` vs `269,637,105 B`,
     reduction `9.085×`, L2 `10.73%` — matches §4;
   - rescue @128K: `0.9788 / 0.8682 / 0.9591 / 0.9530` (p10 + p11) — matches §3.2;
   - sink effect @64K: `0.7620` (s0) vs `0.7716` (s4) = `0.96` points — matches §2.
3. **Model provenance**: verified manifest fields directly — p4/p6
   `Qwen/Qwen2.5-7B-Instruct`, `7.61 B`, `28/4` heads (GQA `7:1`), rotary `128`;
   p7 `meta-llama/Llama-2-7b-hf` with no param count recorded; p10/p11 synthetic
   fixtures. Every document's scope statement matches.
4. **ADR template**: 9 ADRs found; all nine contain Status, Date/Tag, Context,
   Decision, Trade-offs & Rejected Alternatives, and Verification Artifact.
5. **Figures**: fig1, fig2 and fig4 links in THE_TRADE and fig3 in KERNEL_NOTES
   resolve; `check_figure_provenance.py` remains 7/7 OK with current SHA256s.
6. **Uncited-number paragraph scan**: one substantive gap found and fixed
   (Finding 6); all remaining hits are labels/headers.

## Verdict

The three documents now describe the full arc with every number traceable to a
committed CSV: the K/V asymmetry law, the depth-resolved cliff and early-warning
sensor, the Hadamard kurtosis rescue, the head-adaptive schedule, the byte and
roofline derivations, and nine uniformly templated ADRs. The two headline
numbers the repo cannot support — `4.5625 B/elem` and `< 0.5%` loss at 128K —
are reported as unmet/absent rather than smuggled in. Approved for commit at
`docs(whitepapers): synchronize THE_TRADE, KERNEL_NOTES, and ADRs 001-009 with v1.2.0 rescues`.
