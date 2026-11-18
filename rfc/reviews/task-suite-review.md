# Adversarial Review: Ecological Task Suite

**Scope**: `task-suite`
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-18
**Branch**: `feat/ecological-task-suite`
**Target Tag**: `v0.5.0-task-suite`

---

## Findings

### Finding 1: TrialRecord had no place for task-specific metrics
- [x] **Finding**: The frozen `TrialRecord` carried only needle fields, so multihop/longsum/longcode metrics (em, f1, rouge_l, fact_retention_pct, ast_exact_match, pass_at_1) had nowhere to live.
- **Resolution**: RESOLVED — Added an optional `metrics: dict[str, float]` field to `TrialRecord` and a generic merge in `EvalTask.to_dataframe` that averages each metric key across trials. The 20-turn agent harness (Phase II) can extend this cleanly.

### Finding 2: Hop1/hop2 isolation must be real, not cosmetic
- [x] **Finding**: A "multi-hop" task that reports a single accuracy cannot isolate late-context bridge failure.
- **Resolution**: RESOLVED — Facts A→B sit at 15–30% depth and bridge B→C at 75–95%; `hop1_recall` and `hop2_recall` are tracked separately. `test_hop1_hop2_isolation` asserts early recall exceeds late recall by >0.2 for `k4v4_postrope_sink4` at ≥32K while fp16 keeps both >0.9.

### Finding 3: LLM-judge variance would confound summarization
- [x] **Finding**: An LLM-judge-based summarization score is non-deterministic and would invalidate CSV provenance.
- **Resolution**: RESOLVED — `fact_retention_pct` is exact retention of 15 planted IDs/numbers/dates (`planted_fact_ids` is deterministic); `judge_score` is an optional *seeded* column. `test_fact_retention_determinism` verifies stability and depth sensitivity.

### Finding 4: `exec`-based unit assertions are a security smell
- [x] **Finding**: Running candidate code with `exec` for pass@1 would introduce code execution in CI and trip Bandit.
- **Resolution**: RESOLVED — `run_committed_assertions` checks structure on the canonical AST string only (no `exec`/`eval`); 4 committed assertions inspect function name, cross-file import, LIMIT guard, and return. Bandit reports no medium+ issues.

### Finding 5: AST exact-match must ignore formatting/comments
- [x] **Finding**: String comparison would flag semantically identical completions with different whitespace.
- **Resolution**: RESOLVED — `canonicalize_ast` uses `ast.parse` + `ast.dump(include_attributes=False)`; `test_ast_canonicalization_whitespace_invariant` verifies whitespace/comment invariance and syntax-error handling.

### Finding 6: Assertion substring bug
- [x] **Finding**: `ast.unparse("from constants import LIMIT")` yields `"from constants import LIMIT"`, which does not contain the substring `"import constants"`, so the committed assertion would always fail.
- **Resolution**: RESOLVED — Changed the assertion to check `"constants" in canon and "import" in canon`; all 4 assertions pass against the expected source.

### Finding 7: Ambiguous depth expression
- [x] **Finding**: `depth_pct=late and 85 or 85` was a confusing no-op in the multihop record.
- **Resolution**: RESOLVED — Replaced with the literal `85` and a clarifying comment that the bridge sits at ~85% depth.

---

## Summary

All 7 findings resolved. Three real-world single-turn tasks (multi-hop bridge QA, fact-retention longsum, AST longcode) are implemented on the shared evaluation spine, with 10 passing tests, a committed 96-row baseline CSV + manifest, and 8 new CSV-backed README claims. All 8 local gates pass.

**Status**: APPROVED — ready for merge
