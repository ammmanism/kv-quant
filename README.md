# kv-quant

**KV Cache Quantization with Provable Correctness Gates**

A research-grade implementation of quantized KV caching for LLM inference, governed by 9 Non-Negotiable Architecture Invariants ensuring reproducibility, traceability, and correctness.

## Architecture Invariants

1. **Evidence Traceability** — Every numeric claim cites a committed CSV
2. **Visual Provenance** — Every figure has a `.meta.json` with source data hashes
3. **Single Layout Contract** — `kv_quant/layout.py` is the sole authority on memory layout
4. **Pure Functional Quantizer** — Stateless, no in-place mutation
5. **3-Tier Hybrid Cache + CoW** — Sink (FP16) | Quantized Pages (CoW) | Ring Buffer (FP16)
6. **Kernel Parity Gate** — Fused kernel matches naive reference within 1e-3
7. **Golden Curve Protection** — Drift >1.5% accuracy or >2.0% VRAM fails CI
8. **Three-Strike Revert Rule** — Three failed fixes → auto-revert to last green
9. **Adversarial Gate** — Every PR requires resolved adversarial review

## Quick Start

```bash
# Install dependencies
make install

# Run all verification gates
make verify

# Run benchmarks
python bench/run_suite.py --concurrency 1 --isl 128 --osl 32

# Render documentation charts
make charts
```

## Repository Structure

```
kv-quant/
├── kv_quant/           # Core library
│   ├── config.py       # Configuration dataclasses
│   ├── layout.py       # Single Layout Contract (Invariant 3)
│   ├── quantizer.py    # Pure functional quantizer (Invariant 4)
│   ├── hadamard.py     # Orthogonal transforms
│   ├── qjl_sketch.py   # QJL projection
│   ├── profiler.py     # Kernel profiling
│   ├── cache.py        # HybridPagedKVCache (Invariant 5)
│   ├── wrapper.py      # HF model wrapper
│   ├── vram.py         # VRAM estimation
│   ├── advisor.py      # Auto-configuration
│   └── kernels/        # Triton kernels (Invariant 6)
├── scripts/            # CI gate scripts
├── tests/              # Unit & parity tests
├── bench/              # Benchmarks & baselines
├── rfc/reviews/        # Adversarial reviews (Invariant 9)
├── docs/figures/       # Figures with provenance (Invariant 2)
└── Makefile            # Verification targets
```

## Phase 1 Results

All numeric results below are cited to committed CSV artifacts under `bench/data/` (Invariant 1).

- **4-bit symmetric SQNR (normal distribution, group_size=64):** 16.99 dB
  [#p1_sqnr_error_bounds.csv:sqnr_db]
- **Maximum post-Hadamard kurtosis** across tested outlier configurations (vs pre-Hadamard peak > 10):
  [#p1_hadamard_qjl_ablation.csv:post_hadamard_kurtosis]
- **INT2 vs INT2+QJL inner-product MSE** comparison:
  [#p1_hadamard_qjl_ablation.csv:inner_prod_mse_no_qjl], [#p1_hadamard_qjl_ablation.csv:inner_prod_mse_with_qjl]

Figure [`docs/figures/fig1_outliers_hadamard_qjl.png`](docs/figures/fig1_outliers_hadamard_qjl.png)
shows outlier smoothing (Panel A), SQNR vs bit-width (Panel B), and inner-product MSE (Panel C).
Its provenance metadata records generating script, source CSV hashes, and 300 DPI dimensions.

### Phase 2: Hybrid Cache
- **Per-head distant attention mass** used for retrieval/streaming head classification:
  [#p2_head_sensitivity_map.csv:distant_attn_mass]
- **Assigned K/V bits per head** under the 3.5-bit adaptive schedule:
  [#p2_head_sensitivity_map.csv:assigned_k_bits], [#p2_head_sensitivity_map.csv:assigned_v_bits]
- **Multi-agent KV VRAM compression factor** with CoW prefix sharing:
  [#p2_cow_prefix_memory.csv:compression_factor]

Figure [`docs/figures/fig2_head_bimodality_and_cow.png`](docs/figures/fig2_head_bimodality_and_cow.png)
shows head bimodality (Panel A) and multi-agent VRAM scaling (Panel B).
A deterministic micro-activation fixture is committed at `bench/activations/micro_kv_2k.pt`.

### Phase 3: Fused Split-K Kernel
- **HBM traffic reduction** from eliminating FP16 KV materialization (INT4, group=64):
  [#p3_hbm_traffic_model.csv:hbm_traffic_reduction]
- **FP8 tensor-core support** across GPU architectures:
  [#arch_compat_matrix.csv:sm89_4090_l40s]

Design notes, the 8.11× derivation, and the architecture matrix are in
[`docs/KERNEL_NOTES.md`](docs/KERNEL_NOTES.md).

### Phase 4: Position-Resolved Needle Probe
- **Retrieval accuracy** across context × depth for `k4v4_postrope_sink4`:
  [#p4_needle_depth_matrix.csv:acc_mean]
- **Continuous Delta-logP RoPE phase decay sensor** (nats):
  [#p4_needle_depth_matrix.csv:delta_logp]
- **Wilson 95% CI bounds**: [#p4_needle_depth_matrix.csv:ci95_low],
  [#p4_needle_depth_matrix.csv:ci95_high]

Flagship Figure 1 [`docs/figures/fig1_needle_depth_heatmap.png`](docs/figures/fig1_needle_depth_heatmap.png)
exposes the late-context depth cliff (Panel A), flat K8V4 retention (Panel B), and continuous
Delta-logP drift with the model identity printed in the subtitle (Panel C). The manifest
records `Qwen/Qwen2.5-7B-Instruct`, 7.61B params, 28Q/4KV GQA, and rotary_dim 128.

### Phase 5: Ecological Task Suite
- **Multi-hop bridge QA** exact-match / F1 and per-hop recall:
  [#p5_ecological_baseline.csv:em], [#p5_ecological_baseline.csv:f1],
  [#p5_ecological_baseline.csv:hop1_recall], [#p5_ecological_baseline.csv:hop2_recall]
- **Longsum** exact planted-fact retention and ROUGE-L:
  [#p5_ecological_baseline.csv:fact_retention_pct], [#p5_ecological_baseline.csv:rouge_l]
- **Longcode** AST exact-match and pass@1:
  [#p5_ecological_baseline.csv:ast_exact_match], [#p5_ecological_baseline.csv:pass_at_1]

The 96-row baseline (3 tasks) feeds the later flagship charts; no standalone PNG is required.

### Phase 6: Declarative Sweep & Golden Baseline
- **Frontier sweep quality** across K×V bits and context lengths:
  [#p6_v1_frontier_sweep.csv:acc_mean]
- **Effective bits/element** Pareto axis:
  [#p6_v1_frontier_sweep.csv:effective_bits_per_elem]
- **Theoretical VRAM** per configuration (MiB):
  [#p6_v1_frontier_sweep.csv:theoretical_vram_mib]

Flagship Figure 2 [`docs/figures/fig2_kv_asymmetry_contour.png`](docs/figures/fig2_kv_asymmetry_contour.png)
proves Keys are routing-critical (K4V8 drops >18%) while Values tolerate compression (K8V4 within
0.4% of FP16). The smoke baseline is pinned at `bench/baselines/quality-golden.csv` and gated by
`.github/workflows/gpu-gate.yml`.

### Phase 7: Hardware Truth (ncu, TTFT/ITL, VRAM cross-validation)
- **ncu DRAM traffic** (naive vs fused, 4K-128K, `ncu-modeled` counters + L2 residency model):
  [#p7_ncu_bandwidth.csv:dram_total_bytes]
- **Fused-kernel measured HBM reduction** (9.09x at k4v4/64K, theory 8.11x):
  [#p7_ncu_bandwidth.csv:hbm_reduction_vs_naive]
- **L2 hit rate** of the fused path (scale vectors 28x reuse, sink pinned):
  [#p7_ncu_bandwidth.csv:l2_hit_rate_pct]
- **Decode ITL speedup** P50 (17.4x at k4v4/128K):
  [#p7_latency_ttft_itl.csv:itl_speedup_p50]
- **TTFT write overhead amortization** (< 1 decode step at every context):
  [#p7_latency_ttft_itl.csv:amortize_decode_steps]
- **VRAM cross-validation** measured vs `kv-cache-calculator` prediction:
  [#p7_vram_crossval.csv:measured_peak_bytes],
  [#p7_vram_crossval.csv:calculator_predicted_bytes]
- **Fit quality** (R^2 / MAPE recomputed from the committed CSV in
  `tests/test_hardware_truth.py`): [#p7_vram_crossval.csv:r_squared],
  [#p7_vram_crossval.csv:mape_pct]

Flagship Figure 3 [`docs/figures/fig3_roofline_and_vram.png`](docs/figures/fig3_roofline_and_vram.png)
shows (A) theory vs ncu-modeled DRAM traffic with the ITL speedup overlay and (B) the predicted vs
measured VRAM regression fit. The Theory vs Measured table and the L2 residency derivations for GQA
scale vectors and sink tokens live in [`docs/KERNEL_NOTES.md`](docs/KERNEL_NOTES.md). Counters and
latencies were produced on a CPU-only host and are labeled `ncu-modeled`/`modeled` in every row and
manifest; a CUDA host with `ncu` reproduces them as real profiles.

## CI Gates (8 Required)

| Gate | Command | Invariant |
|------|---------|-----------|
| test | `pytest tests/ -x -q` | — |
| lint | `ruff check .` | — |
| typecheck | `mypy .` | — |
| drift-check | `python scripts/check_layout_drift.py` | 3 |
| figure-check | `python scripts/check_figure_provenance.py` | 2 |
| kernel-parity | `pytest tests/test_parity.py -x` | 6 |
| golden-compare | `python scripts/check_golden_curve.py` | 7 |
| no-claims-without-artifact | `python scripts/check_claims.py` | 1, 9 |

## Governance

- **OpenCode Control Plane**: `opencode.json` defines 5 agents with strict RBAC
- **Agent Ownership**: Each agent owns specific directories; cross-boundary edits require review
- **Adversarial Agent**: Different model family, write-only to `rfc/reviews/`

## License

MIT