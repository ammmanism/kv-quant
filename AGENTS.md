# kv-quant Agent Instructions

## Absolute Rules

1. **No performance claim without a committed CSV under `bench/data/`.**  
   A number without a CSV is a hallucination and must be deleted.

2. **Do not touch more than ONE core file per task.**  
   `scheduler.py` and `blocks.py` are coupled; `engine.py` and `overlap.py` are coupled; never edit two core files at once.

3. **Never add more than 200 lines of new code in one pass.**  
   Deliver by function/class.

4. **Never mark a task complete without running its verification command and pasting the actual output.**  
   Do not paraphrase test results.

5. **If a test fails, stop and report it.**  
   Do not "fix around" it silently.

6. **Correctness precedes performance.**  
   Greedy parity vs HuggingFace on 200 prompts is the gate for everything downstream.

## Verification Commands

- **Unit tests:** `pytest tests/ -x -q`
- **Parity:** `pytest tests/test_parity.py -x`
- **Bench smoke:** `python bench/run_suite.py --concurrency 1 --isl 128 --osl 32`

## Task Workflow

(a) State which files you will modify and which you will not  
(b) Show the interface/function signatures  
(c) Implement in ≤ 200 lines  
(d) Run the verification command, paste real output  
(e) Stop and wait for review

---

## Nine Non-Negotiable Architecture Invariants

### Invariant 1 — Evidence Traceability
Every numeric claim in `README.md` or `docs/*.md` **MUST** cite a committed CSV via ``.
- Enforced by `scripts/check_claims.py` in CI.
- The script parses markdown for `[#...]` references, resolves them to CSV files in `bench/data/`, and verifies the cited statistic exists in that CSV.

### Invariant 2 — Visual Provenance
Every PNG in `docs/figures/*.png` **MUST** have a companion `docs/figures/<name>.png.meta.json` recording:
- Generating script path
- Source CSV paths
- Source CSV SHA256 hashes
- Image dimensions (≥300 DPI)
- Enforced by `scripts/check_figure_provenance.py` in CI.

### Invariant 3 — Single Layout Contract
`kv_quant/layout.py` is the **single source of truth** for:
- Bit masks
- Shifts
- QJL sketch words
- Byte-accounting equations
- Enforced by `scripts/check_layout_drift.py` in CI (compares against a committed golden layout hash).

### Invariant 4 — Pure Functional Quantizer & Orthogonal Transforms
Quantizer, Hadamard, and QJL functions are **pure and stateless**.
- No hidden global state, no in-place mutation of inputs.
- All randomness is explicit via seeded `torch.Generator` arguments.

### Invariant 5 — 3-Tier Hybrid Cache + CoW
`HybridPagedKVCache` keeps:
- `sink_tokens` in FP16 (never quantized)
- Full `group_size` blocks sealed into quantized pages with **Copy-on-Write prefix hashing**
- Trailing `< group_size` tokens in an FP16 ring buffer

### Invariant 6 — Kernel Parity Gate
`dequant_fused.py` must match `dequant_naive.py` within `1e-3` on:
- Synthetic fixtures
- Real captured activation fixtures
- Enforced by `tests/test_parity.py` and CI job `kernel-parity`.

### Invariant 7 — Golden Curve Protection
Drift **>1.5% accuracy** or **>2.0% VRAM error** vs `bench/baselines/quality-golden.csv` fails CI.
- Enforced by CI job `golden-compare`.

### Invariant 8 — Three-Strike Revert Rule
Three failed fixes → **revert to last green commit** automatically.
- Tracked per-PR in CI; after 3 failed workflow runs on the same PR, a bot forces revert.

### Invariant 9 — Adversarial Gate
Every PR **MUST** commit `rfc/reviews/<scope>-review.md` with all findings resolved:
- `Fixed in <sha>`
- `Won't do — <reason>`
- `Question`
- Enforced by CI job `no-claims-without-artifact` (checks for review file existence and resolution markers).

---

## Repository Structure (Skeleton)

```
kv-quant/
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       └── ci.yml
├── kv_quant/
│   ├── __init__.py
│   ├── config.py
│   ├── layout.py
│   ├── quantizer.py
│   ├── hadamard.py
│   ├── qjl_sketch.py
│   ├── profiler.py
│   ├── cache.py
│   ├── wrapper.py
│   ├── vram.py
│   ├── advisor.py
│   └── kernels/
│       ├── __init__.py
│       ├── dequant_naive.py
│       └── dequant_fused.py
├── scripts/
│   ├── check_claims.py
│   ├── check_layout_drift.py
│   └── check_figure_provenance.py
├── tests/
│   ├── __init__.py
│   ├── test_parity.py
│   └── test_layout.py
├── bench/
│   ├── baselines/
│   │   └── quality-golden.csv
│   ├── data/
│   ├── run_suite.py
│   └── __init__.py
├── tasks/
├── sweeps/
├── rfc/
│   └── reviews/
├── docs/
│   └── figures/
├── Makefile
├── pyproject.toml
├── opencode.json
├── AGENTS.md
└── README.md
```

---

## Agent Ownership Matrix

| Agent | Model Class | Owned Paths |
|-------|-------------|-------------|
| `quant-core` | opus-class | `kv_quant/{config,layout,quantizer,hadamard,qjl_sketch,profiler,cache,wrapper,vram,advisor}.py` |
| `kv-kernel` | opus-class | `kv_quant/kernels/` ONLY |
| `harness` | mid-tier | `tasks/`, `bench/`, `sweeps/`, `scripts/` |
| `planner` | opus-class | `rfc/*.md`, `docs/*.md` |
| `adversary` | **different model family** | `rfc/reviews/**` only (edit deny on `*`) |

---

## CI Gate Jobs (8 Required)

1. `test` — `pytest tests/ -x -q`
2. `lint` — `ruff check .`
3. `typecheck` — `mypy .`
4. `drift-check` — `python scripts/check_layout_drift.py`
5. `figure-check` — `python scripts/check_figure_provenance.py`
5. `kernel-parity` — `pytest tests/test_parity.py -x`
7. `golden-compare` — compares current bench run vs `bench/baselines/quality-golden.csv`
8. `no-claims-without-artifact` — `python scripts/check_claims.py` + review file check

---

## Makefile Targets

```make
verify: test lint typecheck drift-check figure-check kernel-parity golden-compare no-claims-without-artifact
test: pytest tests/ -x -q
lint: ruff check .
typecheck: mypy .
drift-check: python scripts/check_layout_drift.py
figure-check: python scripts/check_figure_provenance.py
kernel-parity: pytest tests/test_parity.py -x
golden-compare: python scripts/check_golden_curve.py
no-claims-without-artifact: python scripts/check_claims.py
charts: python scripts/render_charts.py
clean: rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache bench/data/tmp
```