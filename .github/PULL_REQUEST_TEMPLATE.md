<!--
Thank you for contributing to kv-quant!

Before submitting, ensure:
1. All 8 CI gates pass (test, lint, typecheck, drift-check, figure-check, kernel-parity, golden-compare, no-claims-without-artifact)
2. Adversarial review committed to rfc/reviews/<scope>-review.md with all findings resolved
3. No numeric claims without CSV artifacts in bench/data/
4. No figures without .meta.json provenance files
-->

## Summary
<!-- One-paragraph summary of the change -->

## Scope
<!-- What area does this touch? e.g., quantizer, kernel, cache, bench, docs -->

## Motivation
<!-- Why is this change needed? Link to issue/RFC if applicable -->

## Changes
<!-- Bullet list of concrete changes -->
- 
- 

## Verification
<!-- How was this tested? -->
- [ ] Unit tests pass (`make test`)
- [ ] Lint passes (`make lint`)
- [ ] Typecheck passes (`make typecheck`)
- [ ] Layout drift check passes (`make drift-check`)
- [ ] Figure provenance check passes (`make figure-check`)
- [ ] Kernel parity passes (`make kernel-parity`)
- [ ] Golden curve compare passes (`make golden-compare`)
- [ ] Claims check passes (`make no-claims-without-artifact`)
- [ ] Adversarial review completed and committed

## Adversarial Review
<!-- Required per Invariant 9 -->
- Review file: `rfc/reviews/<scope>-review.md`
- All findings resolved:
  - [ ] Fixed in `<sha>`
  - [ ] Won't do — `<reason>`
  - [ ] Question

## Performance Claims
<!-- Required per Invariant 1 -->
- [ ] No new numeric claims in README.md or docs/*.md
- [ ] OR all new claims cite committed CSVs in bench/data/ via `[#file.csv:column]`

## Visual Changes
<!-- Required per Invariant 2 -->
- [ ] No new PNG figures added
- [ ] OR all new figures have `.meta.json` provenance files

## Layout Changes
<!-- Required per Invariant 3 -->
- [ ] No changes to kv_quant/layout.py
- [ ] OR LAYOUT_GOLDEN_HASH and LAYOUT_VERSION updated in layout.py

## Checklist
- [ ] Code follows project style (ruff, mypy strict)
- [ ] No more than 200 lines changed in a single core file
- [ ] No silent test fixes (all failures reported)
- [ ] Correctness precedes performance