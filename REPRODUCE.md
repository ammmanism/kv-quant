# REPRODUCE — Reproducing Every Number in kv-quant v2.0.0

This document is the reproducibility lock for `v2.0.0`. Every CSV artifact and
flagship figure committed at this tag is listed with its SHA256 checksum, and
the exact commands that regenerate each one are given. The reproducibility
contract is **column-level equality to float64 precision** (§2.2), not byte
identity: if a regenerated file's hash (after CRLF→LF normalization, §4) does
not match the table, first diff the columns — last-ULP float noise is
expected; a *value* difference is a bug or an environment difference, and
must never be silently committed as drift.

**Honesty note (read first).** v1.0.0 was produced on a CPU-only Windows host
(no CUDA, no Triton runtime). Consequently:
- Quality sweeps are **modeled** deterministic evaluations, not real-model
  forward passes. Each artifact's `.manifest.json` records the execution
  context (`host: cpu`, `modeled: true`) at row granularity where applicable.
- Latency/DRAM numbers are **ncu-modeled** traffic models, not hardware
  counters. Regenerating them on a GPU host with `ncu` attached replaces the
  models with real counters (§2.4) — that is the intended upgrade path.
- All *code-level* guarantees (kernel parity ≤1e-3, layout golden hash, VRAM
  cross-validation R²=1.0000) are real, executed tests, not models.

---

## 1. Environment

```bash
# Python 3.11 or 3.12; dependency lock via uv
uv sync                      # creates .venv from uv.lock
uv run python -c "import torch, transformers; print(torch.__version__)"
```

- OS used for v1.0.0 artifacts: Windows 11 (win32), CPU-only.
- CI matrix: Ubuntu, Python 3.11 and 3.12, CPU-only (the "CPU Matrix" check).
- GPU verification is a separate tier (`make verify-gpu`); it auto-skips on
  CPU hosts and must be run explicitly on a CUDA host (ADR-006 in
  `docs/DECISIONS.md`).

## 2. Exact commands, per artifact

Run all commands from the repository root. Every sweep is resumable and
writes a `.manifest.json` beside the CSV (10 mandatory fields incl. seed,
host context, git SHA).

### 2.1 The local and CI verification gates

There are three gate sets, not one — they overlap but are not identical:

| Runner | Gates |
|---|---|
| `.\verify.ps1` (9 steps) | pytest `not gpu`, ruff check **+ format**, mypy strict, layout drift, claims, figure provenance, explorer drift, pr-governance, bandit |
| `make verify` (9 targets) | pytest, ruff check, mypy, layout drift, figure provenance, kernel parity, golden compare, no-claims-without-artifact (**claims + strict adversarial review**), explorer drift |
| CI `verify-cpu` (7 steps, ×2 Python versions) | pytest with coverage, ruff check + format, mypy, layout drift, claims, figure provenance, explorer drift |

The strict adversarial checker (`check_adversarial_review.py`) runs in
`make verify` but not in CI; CI's `check_pr_governance.py` checks a different
governance surface. Do not describe them as "the same gates".

Expected at v2.0.0: `166 passed, 2 skipped (GPU-marked), 4 deselected`;
`All figures have valid provenance metadata`; `[PASS: explorer-drift]`;
claims checker `100% resolved`; adversarial review gate `[OK]`.

### 2.2 Quality artifacts (phases 1–6)

| Artifact | Command |
|---|---|
| `bench/data/p1_hadamard_qjl_ablation.csv`, `bench/data/p1_sqnr_error_bounds.csv` | `uv run python scripts/generate_phase1_artifacts.py` |
| `bench/data/arch_compat_matrix.csv`, `bench/data/p3_hbm_traffic_model.csv` | `uv run python scripts/generate_phase3_artifacts.py` |
| `bench/data/p2_cow_prefix_memory.csv`, `bench/data/p2_head_sensitivity_map.csv` | `uv run python scripts/generate_phase2_artifacts.py` |
| `bench/data/p4_needle_depth_matrix.csv` | see snippet below (`tasks/needle.py` sweep API) |
| `bench/data/p5_ecological_baseline.csv` | `uv run python scripts/generate_ecological_baseline.py` |
| `bench/data/p6_v1_frontier_sweep.csv` | `uv run python bench/quality_sweep.py --config sweeps/v1-frontier.yaml` |
| `bench/data/quality-current.csv` | `uv run python bench/run_suite.py --concurrency 1 --isl 128 --osl 32` |
| `bench/baselines/quality-golden.csv` | **frozen golden** — do not regenerate; it is the Invariant 7 reference |

**Reproducibility contract.** All sweeps are deterministic under their
recorded seeds. The contract is **column-level equality to float64
precision**, not byte identity: re-runs may differ in the last 1–2 ULP of
float columns (summation-order sensitivity — verified empirically for
`p4_needle_depth_matrix.csv`, max |Δ| = 8.9e-16 over 1,155 rows), so byte
checksums may drift in those digits. A checksum mismatch is a signal to diff
columns, not automatically a bug.

The phase-4 needle matrix has no standalone CLI runner; it is regenerated
through the committed task API (7 configs × 5 contexts × 11 depths × 3
formats × 20 trials = 1,155 rows):

```python
from pathlib import Path
from tasks.base import RecordedReplayAdapter, SweepManifest
from tasks.needle import PositionResolvedNeedleTask

task = PositionResolvedNeedleTask(RecordedReplayAdapter(trials_per_cell=20))
task.sweep(SweepManifest.capture(seed=42), Path("bench/data/p4_needle_depth_matrix.csv"))
```

**Verified at v1.0.0**: this snippet reproduces the committed artifact with
every column identical except last-ULP float noise in `delta_logp`
(max |Δ| = 8.9e-16 over 1,155 rows; summation-order sensitivity), i.e.
value-identical to float64 precision. See §4 for the reproducibility
contract this implies.

### 2.3 Kernel & VRAM artifacts (phase 7)

| Artifact | Command | Host |
|---|---|---|
| `bench/data/p7_latency_ttft_itl.csv` | `uv run python bench/latency.py` | CPU (modeled ITL) |
| `bench/data/p7_ncu_bandwidth.csv` | `uv run python bench/bandwidth.py --sweep` (GPU host: real `ncu` counters) | CPU → ncu-modeled |
| `bench/data/p7_vram_crossval.csv` | `uv run python bench/vram_crossval.py` | CPU (real allocator) |

### 2.4 Upgrading modeled artifacts to real-GPU artifacts

On a CUDA host with Triton and `ncu` available:

```bash
uv run pytest tests/test_parity.py tests/test_fused_kernel.py -m gpu -q
make verify-gpu
```

**Honest status of the real-counter path at v1.0.0.** `bench/bandwidth.py`
exposes `--mode cuda` and `--use-ncu` flags, but at v1.0.0 no flag replaces
the modeled rows with hardware counters — `write_ncu_bandwidth_csv` always
emits `ncu-modeled` rows (`cuda_counter_bytes` is itself a model until an
`ncu`-drivered capture is wired in). Wiring real counters is deliberately
deferred to v2.0 GPU verification work (ADR-006, `docs/DECISIONS.md`); until
then the committed `p7_ncu_bandwidth.csv` must keep its `ncu-modeled`
labeling.

When real counters are wired in, regenerate the CSV with the new rows labeled
`modeled: false` in the manifest and re-pin the checksum in §3 (a checksum
update is a normal, reviewable event — commit it with a note in
`CHANGELOG.md`).

### 2.5 Agent & fleet artifacts (phases 13–14)

| Artifact | Command | Host |
|---|---|---|
| `bench/data/p13_ragged_concurrency.csv` | `uv run python bench/concurrency_sim.py` | CPU (modeled byte model) |
| `bench/data/p14_agent_trajectory_drift.csv` | `uv run python -m tasks.agent_trajectory --jobs 4` | CPU (simulated harness) |

`--jobs` is throughput-only: parallel and serial sweeps were compared
row-for-row and are bit-identical (ADR-012). `tasks/` has no `__init__.py`,
so the `-m` module invocation is required — `python tasks/agent_trajectory.py`
fails with `ModuleNotFoundError`.

### 2.6 Interactive explorer

```bash
uv run python scripts/build_interactive_explorer.py   # regenerates docs/explorer.html
uv run python scripts/check_explorer_drift.py         # rebuild + byte-compare gate
node scripts/explorer_smoke.js                        # headless drive of all 111 control combos
```

The explorer is deterministic by construction (no timestamps, no absolute
paths, sorted keys) and re-verifies all 19 tracked CSV manifests on every
build. `quality-current.csv` is a CI-regenerated stub whose manifest is
deliberately gitignored — the explorer excludes it and the drift gate pins
the artifact count at 19.

## 3. SHA256 checksums (v2.0.0)

Computed over byte content with `\r\n` normalized to `\n` (see §4). Full
64-hex digests.

| File | SHA256 |
|---|---|
| bench/data/arch_compat_matrix.csv | 02147490c75ce16d4e9d006125c30b4452c28316e1fda06b37567f4322211e76 |
| bench/data/p10_hadamard_rescue.csv | 3a026d63cfe2e3cae3d9dff691ca5bfbdaed9cf8773cc524663e8a05523a5fcf |
| bench/data/p10_hadamard_rotation.csv | 281b0c48cc74fac08bb128593d21f499aaa5202ed50a6d5c21068b0f14876944 |
| bench/data/p11_head_rescue_curve.csv | 55691ae8a8e73bf4272185612b371caf821631424fa0548ab979168c8a2f24ef |
| bench/data/p11_head_sensitivity_schedule.csv | 2a712857a218d7cf8c2a71372eedf004f5a86f1668e35864bc23ec58c84422ce |
| bench/data/p12_qjl_mse_ablation.csv | 7045425f3e8336f1af477af590bd47e3cfa578bf081b7bfa2e1f57462d7fd74d |
| bench/data/p13_ragged_concurrency.csv | 5ee7c8bdda4ae8116141041e8ecdf8da95e6ba59ebd9b841bdab07676a654cb3 |
| bench/data/p14_agent_trajectory_drift.csv | ebae3e9f92950440d05f87d26b5a9ad03d4dbd27dfbd2e8d0809b805ab2c8465 |
| bench/data/p1_hadamard_qjl_ablation.csv | b4bd68b16236209e40bffeed736d84cef2f3f813844ede06e10c6a99018fb8f3 |
| bench/data/p1_sqnr_error_bounds.csv | 59270534873944181ee0bf655a63518df3e15277638983a1e8c69a3610000742 |
| bench/data/p2_cow_prefix_memory.csv | 5ccc508421e8a2c35e132a8a7a02bf002b5c3cf327aad6ba5f16f30336c5c346 |
| bench/data/p2_head_sensitivity_map.csv | 960d56927732cbc7d02299035c605187d255274354d5416957c4a3eda4052e4c |
| bench/data/p3_hbm_traffic_model.csv | 14e956e30399cba66ce8a765be1c74cb867c477ee6b62deb88115d063204c478 |
| bench/data/p4_needle_depth_matrix.csv | a097db2aba5bbde6613d6957393e8bc421a458de39a099061e039310465d14a7 |
| bench/data/p5_ecological_baseline.csv | 43ddf6413aaea83b6ba28c1aec7f01089e46480d2c5c7f9db1cf641c65a60166 |
| bench/data/p6_v1_frontier_sweep.csv | ca9859cacd52797c75c6473fe3dc190d6ec3e755abd27a8143704dd091d5f1f7 |
| bench/data/p7_latency_ttft_itl.csv | 9c6ef7d9613e7c1f19c35e50c76c568528137ee5f60a89c62609dc57c3ae27a2 |
| bench/data/p7_ncu_bandwidth.csv | 840c5c75e32fbe6cd1fffca054a99947a06280ce2ac9097ed4250e09382480fe |
| bench/data/p7_vram_crossval.csv | f0c88381ea7b83fefbb89f93ad6759c66227ca21ea0d40aea2b63fca298d4953 |
| bench/baselines/quality-golden.csv | 8ce2b65066965065ec68d4918677fdc7c6ba12be80b37437f783e3b36605e7b7 |
| docs/figures/fig0_hero_cliff.png | 73a211825d9d04add3313d3dd31f2744b62c93ee9b070b3433ba36b3f1eb4d30 |
| docs/figures/fig1_needle_depth_heatmap.png | 187c25f3390a6d76123eaad11b78b701117ff76bdc10530245495dc1f2e044d7 |
| docs/figures/fig1_outliers_hadamard_qjl.png | e39bb10aa98aa366ca6c3dbda5f6d5fdc18ae34f2c4c8a10f3f33cbff154bad7 |
| docs/figures/fig2_head_bimodality_and_cow.png | dba72a1f23ef9a271b01c7919c443f47150a8c4db24d709b0411b8ebb7d0c98e |
| docs/figures/fig2_kv_asymmetry_contour.png | 59769505f464351830da62293a3b9e239d5dd7c791412ab77a6e5eb6b12bddf0 |
| docs/figures/fig3_roofline_and_vram.png | 2e02511fe527c701b764d28074e8332af81726cbb14fe147523ee2665c46380e |
| docs/figures/fig4_hadamard_and_head_rescue.png | 32b153258d815bc2270fe88e4b6aa9297e492dbae0b2eb155f2a79024c361738 |
| docs/figures/fig5_agent_amnesia_vs_syntax.png | f4b30086c09e08b7bed25f21a782c2055ce73ad098de14045071aa9770b42e37 |
| docs/explorer.html | 71e1030d16c586f36e65336b6d4e1c98b9b163165138d11c44d4a992aeddc56a |

`bench/data/quality-current.csv` is a CI-regenerated stub (its manifests are
gitignored by design, see `.gitignore`); it is excluded from this table and
from the explorer's provenance block.

## 4. Checksum canonicalization

CSVs and figure PNGs are hashed after normalizing line endings so that
Windows (CRLF) and Linux (LF) checkouts agree:

```python
import hashlib
from pathlib import Path


def sha256_normalized(p: str) -> str:
    data = Path(p).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
```

PNGs are binary; the normalization is a no-op for them, but the same function
is used so one code path covers the whole table.

## 5. Figures

Each flagship figure is regenerated by its committed script; the
`.png.meta.json` beside each records the generating script, source CSVs,
source CSV SHA256s, dimensions, and DPI (Invariant 2, enforced by
`scripts/check_figure_provenance.py`):

```bash
uv run python scripts/plot_hero_cliff.py              # fig0_hero_cliff.png
uv run python scripts/plot_needle_heatmap.py          # fig1_needle_depth_heatmap.png
uv run python scripts/generate_phase1_artifacts.py    # fig1_outliers_hadamard_qjl.png
uv run python scripts/generate_phase2_artifacts.py    # fig2_head_bimodality_and_cow.png
uv run python scripts/plot_kv_asymmetry.py            # fig2_kv_asymmetry_contour.png
uv run python scripts/plot_roofline_and_vram.py       # fig3_roofline_and_vram.png
uv run python scripts/generate_phase11_artifacts.py   # fig4_hadamard_and_head_rescue.png
uv run python scripts/generate_phase14_artifacts.py   # fig5_agent_amnesia_vs_syntax.png
```

## 6. Verifying the claims themselves

```bash
uv run python scripts/check_claims.py
```

This resolves every `[#file.csv:column]` citation in the markdown corpus
against the committed CSVs — the mechanism that makes every number in
`README.md`, `docs/*.md` traceable (Invariant 1).
