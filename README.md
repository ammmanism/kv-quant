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