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