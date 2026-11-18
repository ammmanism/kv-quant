# Adversarial Review: gpu-gate Workflow Repair (golden-compare resurrection)

**Scope**: `ci-gate-repair` — `.github/workflows/gpu-gate.yml` and its golden
pipeline (`scripts/summarize_quality.py`).
**Reviewer**: Adversarial Agent (different model family)
**Date**: 2026-11-18
**Branch**: `fix/gpu-gate-workflow`

**Method**: I treated the symptom reported by the user ("gpu-gate red on every
push since at least v0.8.0") as unproven, reproduced the failure from the
GitHub Actions API (runs exist, conclusion `failure`, `total_count: 0` jobs,
run *name* = raw file path `.github/workflows/gpu-gate.yml`), then parsed the
workflow file with a real YAML parser and executed every repaired step
locally before proposing the merge.

---

## Findings

### Finding 1: gpu-gate.yml was invalid YAML — the workflow never ran once (severity: HIGH)
- [x] **Finding**: The step
  `run: uv run python -c "` (Golden baseline drift check) opened a
  double-quoted scalar that never closed on that line. PyYAML reproduces
  GitHub's rejection exactly: *"while scanning a quoted scalar … found
  unexpected end of stream"* at line 27, consuming the rest of the file. Every
  `gpu-gate` run since the file landed (v0.8.0 era, PR #9 onward, including
  v1.0.0 and v1.1.0-hadamard) failed at workflow validation: **zero jobs
  created, zero check-runs, no test ever executed**.
- **Impact**: This is not cosmetic red noise. The `golden-regression` job is
  the CI implementation of **Invariant 7 (Golden Curve Protection)** and CI
  job 7 of the 8-gate contract in `AGENTS.md`. It has been silently
  unenforced for the entire v1.x line: `README.md`'s claim that the smoke
  baseline "is pinned at `bench/baselines/quality-golden.csv` and gated by"
  CI was false in the strict sense — the gate existed in the repo but has
  never executed anywhere.
- **Resolution**: RESOLVED — the step is gone entirely (superseded by
  `scripts/summarize_quality.py`, Finding 3), the file parses, and every
  remaining `run:` uses safe block scalars or `>`-folded scalars. Verified:
  `yaml.safe_load` parses both jobs and three triggers.

### Finding 2: Output-path mismatch — compare would fail even with valid YAML (severity: HIGH)
- [x] **Finding**: The sweep step wrote
  `--output bench/data/quality-current-proxy.csv`, but
  `scripts/check_golden_curve.py` hard-reads
  `bench/data/quality-current.csv`. With the file absent the gate fails
  ("Current results not found"); with a *stale* file present it would silently
  compare against an old run — the worst failure mode a drift gate can have.
- **Resolution**: RESOLVED — the repaired pipeline now writes exactly what
  the checker consumes: sweep → `bench/data/quality-current-sweep.csv`, then
  `scripts/summarize_quality.py` → `bench/data/quality-current-summary.csv`,
  then `check_golden_curve.py --current …summary.csv`. The checker gained an
  optional `--current` argument (default unchanged, so the Makefile target
  and any existing caller keep their behavior); no threshold or comparison
  logic was touched.

### Finding 3: Schema mismatch — the compare could never pass at any path (severity: HIGH)
- [x] **Finding**: `check_golden_curve.py` expects a two-row `metric,value`
  table (like the committed golden file), but `quality_sweep.py` emits one
  row per (config, context) cell. Running the original job verbatim with the
  path fixed would still crash: `KeyError: 'metric'` — reproduced locally
  against the real sweep output. The aggregation existed only inside
  `quality_sweep.py --refresh-golden`, which *writes the golden file* and is
  therefore unusable as a per-run CI step.
- **Resolution**: RESOLVED — new `scripts/summarize_quality.py` reduces a
  sweep CSV to the two gated metrics using byte-identical aggregation to
  `--refresh-golden` (mean `acc_mean`, mean `theoretical_vram_mib` over
  `status == "OK"` rows), and refuses to write an empty summary if zero OK
  rows exist. Verified end-to-end locally (Finding 5).

### Finding 4: Per-push GPU job on a runner that does not exist (severity: MEDIUM)
- [x] **Finding**: `gpu-kernel-parity` ran `if: github.event_name == 'push'`
  on `runs-on: [self-hosted, gpu]`. No self-hosted GPU runner is registered,
  so after Finding 1 was fixed this job would queue for its full
  `timeout-minutes: 10` and then fail — turning the workflow red on every
  merge to `main` forever, which is exactly the noise the user asked to
  remove.
- **Resolution**: RESOLVED — the job is now `if: github.event_name ==
  'workflow_dispatch'` with a `workflow_dispatch` trigger added and an inline
  comment naming the reason (no runner registered; ADR-006's two-tier
  economics: GPU parity is *scheduled/manual*, not per-push). When a GPU
  host is available, dispatching the workflow runs the parity gate unchanged.

### Finding 5: Unverified CI budget and thresholds (severity: MEDIUM)
- [x] **Finding**: The job's `timeout-minutes: 6` and its pass/fail thresholds
  were never validated — the workflow had never executed, so "green" was
  fiction and the first real CI run could have timed out or tripped drift.
- **Resolution**: RESOLVED — the full pipeline was executed locally on this
  host before the PR: sweep 96/96 cells in ~4 min (timeout raised 6 → 15 with
  a comment recording the measurement), then summarize, then compare:
  `Accuracy: golden=0.8979, current=0.8979, drift=0.00% [OK]` and
  `VRAM: golden=20085.6, current=20085.6, drift=0.00% [OK]`. The smoke sweep
  reproduces the golden baseline *exactly* (deterministic replay adapter,
  seed 42), so the gate passes on a clean tree and would only fire on real
  drift.

### Finding 6: Transient gate artifacts could be committed accidentally (severity: LOW)
- [x] **Finding**: `bench/data/quality-current*.csv` are per-run artifacts of
  the golden pipeline sitting inside the directory that Invariant 1 treats
  as committed evidence. Nothing ignored them; a local gate run followed by
  `git add -A` (which `scripts/ship_pr.ps1` runs) would commit a mutable
  "current" file next to the frozen golden one.
- **Resolution**: RESOLVED — `.gitignore` now carries
  `bench/data/quality-current-*.csv` and
  `bench/data/quality-current*.csv.manifest.json`, with a comment noting that
  the tracked stub is deliberately not matched. The frozen
  `bench/baselines/quality-golden.csv` remains committed (REPRODUCE.md:
  "frozen golden — do not regenerate").

### Finding 7: The golden pipeline would clobber a tracked file (severity: MEDIUM)
- [x] **Finding**: My first repair draft had the summarize step write
  `bench/data/quality-current.csv` — which is a **tracked** file: a phase-0
  stub (hardcoded `accuracy 0.95`, `vram 1024.0`, `Llama-2-7b-hf`,
  concurrency/isl/osl columns) documented in `REPRODUCE.md` as the output of
  the bench smoke command. Worse, `sweeps/smoke-ci.yaml`'s default `output:`
  was that same path, so any local smoke-sweep run without an explicit
  `--output` silently overwrote a committed artifact — a trap that had
  already fired once: an orphaned
  `bench/data/quality-current.csv.manifest.json` (SHA of a 96-row sweep file
  that was never committed) was sitting untracked in `bench/data/`,
  referencing content that exists nowhere.
- **Resolution**: RESOLVED — the pipeline writes only untracked paths
  (`quality-current-sweep.csv`, `quality-current-summary.csv`);
  `sweeps/smoke-ci.yaml`'s default output now points at
  `quality-current-sweep.csv` with a comment explaining the collision, so the
  stub can no longer be clobbered by accident; the orphaned manifest was
  deleted (generated by `SweepOrchestrator._write_manifest`, referenced by
  nothing, its SHA matches no committed file); and the stub itself is left
  byte-identical to HEAD. Honest note: the stub's placeholder values pass the
  default `make golden-compare` trivially (drift only fails on *regression*,
  and the stub "improves" on golden) — that path is cosmetic until someone
  regenerates it via the documented smoke command; the CI gate is the real
  enforcement and now compares live sweep output.

---

## Honest accounting

- **What was broken**: three stacked defects (invalid YAML; wrong output
  path; incompatible schema) plus a fourth waiting behind them (per-push GPU
  job with no runner), and a fifth trap around the corner (a sweep default
  that would clobber a tracked stub). Any one of the first three alone would
  have kept the golden gate dead; the YAML bug masked the rest by preventing
  any execution.
- **What was claimed vs. real**: every prior PR body and `README.md` listed
  `golden-compare` as a gate. It never ran on GitHub Actions. It was,
  however, genuinely enforced *nowhere else either* — `verify.ps1` (8 local
  gates) does not include it, and the Makefile's `golden-compare` target
  compares against a placeholder stub that cannot fail (drift only trips on
  regression). The gate has now been executed end-to-end for the first time
  and passes with 0.00% drift on both metrics against live sweep output.
- **Not fixed here (out of scope, recorded)**: adding `golden-compare` to the
  8 local pre-flight gates would make local and CI parity complete but adds
  ~4 minutes to every pre-flight; deferred as a decision for the repo owner.
  `check_adversarial_review.py` remains dead code (recorded at v1.0.0 as
  v2.0 debt; unchanged here).

## Verdict

Root cause established (invalid YAML — not a missing runner), four
additional latent failures found by executing the job locally before
believing any fix, and the golden-compare gate now demonstrably runs and
passes with exact reproduction of the frozen baseline. Approved for merge;
PR will itself be the first PR in repo history to run the
`Golden Baseline Regression (CPU proxy)` check.
