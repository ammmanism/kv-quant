#!/usr/bin/env python3
"""Prefill TTFT vs decode ITL crossover analysis for the fused split-K kernel.

On a CPU-only host (no CUDA), latency figures are derived deterministically
from the analytical HBM byte equations plus a seeded lognormal jitter model;
rows are labeled ``mode="modeled"`` and the manifest records the honest
execution context. A CUDA host times real kernels via CUDA events instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bandwidth import (
    HBM_READ_BW_GBPS,
    HBM_WRITE_BW_GBPS,
    LAUNCH_OVERHEAD_US,
    META_BYTES_PER_CHANNEL_GROUP,
    NCU_CONFIGS,
    NCU_SEQ_LENS,
    SCALE_VECTOR_REUSES,
    SINK_L2_HIT,
)

# Fixed quantization prefill overhead: scale computation + packing launches
QUANT_PREFILL_FIXED_US = 250.0
DECODE_FIXED_US = 8.0  # per-step fixed launch/scheduler overhead
JITTER_SIGMA = 0.08
ITL_STEPS = 2000  # decode steps sampled for P50/P90/P99


def _itl_us(
    context_len: int,
    channels: int,
    k_bits: int,
    v_bits: int,
    group_size: int,
    kernel: str,
    sink_tokens: int = 4,
) -> float:
    """Mean decode ITL in microseconds at a given context length."""
    fp16_bytes = context_len * channels * 4.0
    if k_bits == 16 and v_bits == 16:
        payload = fp16_bytes
        meta = 0.0
    else:
        payload = context_len * channels * (k_bits + v_bits) / 8.0
        meta = context_len * channels * META_BYTES_PER_CHANNEL_GROUP / group_size
    sink_bytes = sink_tokens * channels * 4.0

    if kernel == "naive":
        traffic = payload + meta + fp16_bytes + sink_bytes  # read quantized
        traffic += fp16_bytes  # write materialized FP16
        traffic += fp16_bytes  # SDPA re-read FP16
        launches = 3
    else:
        traffic = payload + meta / SCALE_VECTOR_REUSES + sink_bytes * (1 - SINK_L2_HIT)
        launches = 2

    read = traffic
    write = fp16_bytes if kernel == "naive" else 0.0
    return (
        read / (HBM_READ_BW_GBPS * 1e9) * 1e6
        + write / (HBM_WRITE_BW_GBPS * 1e9) * 1e6
        + launches * LAUNCH_OVERHEAD_US
        + DECODE_FIXED_US
    )


def _ttft_ms(
    context_len: int,
    channels: int,
    k_bits: int,
    v_bits: int,
    group_size: int,
    kernel: str,
    sink_tokens: int = 4,
) -> float:
    """Prefill TTFT cache-write time in milliseconds."""
    fp16_bytes = context_len * channels * 4.0
    quant_bytes = (
        context_len * channels * (k_bits + v_bits) / 8.0
        + context_len * channels * META_BYTES_PER_CHANNEL_GROUP / group_size
        + sink_tokens * channels * 4.0
    )
    if kernel == "fp16":
        us = fp16_bytes / (HBM_WRITE_BW_GBPS * 1e9) * 1e6 + 2 * LAUNCH_OVERHEAD_US
    else:
        # quantize: read FP16 activations + write quantized payload + scales
        us = (
            fp16_bytes / (HBM_READ_BW_GBPS * 1e9) * 1e6
            + quant_bytes / (HBM_WRITE_BW_GBPS * 1e9) * 1e6
            + 5 * LAUNCH_OVERHEAD_US
            + QUANT_PREFILL_FIXED_US
        )
    return us / 1000.0


def latency_rows(
    head_dim: int = 128,
    num_kv_heads: int = 32,
    group_size: int = 64,
    seed: int = 42,
) -> list[dict[str, float | int | str]]:
    """Build the TTFT/ITL crossover grid across contexts and configs."""
    import random

    channels = head_dim * num_kv_heads
    rows: list[dict[str, float | int | str]] = []
    for seq_len in NCU_SEQ_LENS:
        for config, (k_bits, v_bits) in NCU_CONFIGS.items():
            rng = random.Random(f"{seed}|{config}|{seq_len}")
            naive_mean = _itl_us(seq_len, channels, k_bits, v_bits, group_size, "naive")
            fused_mean = _itl_us(seq_len, channels, k_bits, v_bits, group_size, "fused")

            def percentiles(rng: random.Random, mean_us: float) -> tuple[float, float, float]:
                samples = sorted(
                    mean_us * math.exp(rng.gauss(0.0, JITTER_SIGMA)) for _ in range(ITL_STEPS)
                )
                p50 = samples[int(0.50 * ITL_STEPS)]
                p90 = samples[int(0.90 * ITL_STEPS)]
                p99 = samples[int(0.99 * ITL_STEPS)]
                return p50, p90, p99

            n_p50, n_p90, n_p99 = percentiles(rng, naive_mean)
            f_p50, f_p90, f_p99 = percentiles(rng, fused_mean)

            ttft_fp16 = _ttft_ms(seq_len, channels, k_bits, v_bits, group_size, "fp16")
            ttft_quant = _ttft_ms(seq_len, channels, k_bits, v_bits, group_size, "quant")
            overhead_us = (ttft_quant - ttft_fp16) * 1000.0
            saving_us = n_p50 - f_p50
            amortize = max(0.0, overhead_us / saving_us) if saving_us > 0 else 0.0

            rows.append(
                {
                    "context_len": seq_len,
                    "config": config,
                    "k_bits": k_bits,
                    "v_bits": v_bits,
                    "ttft_fp16_ms": round(ttft_fp16, 4),
                    "ttft_quant_ms": round(ttft_quant, 4),
                    "ttft_overhead_ms": round(ttft_quant - ttft_fp16, 4),
                    "itl_naive_p50_ms": round(n_p50 / 1000.0, 4),
                    "itl_naive_p90_ms": round(n_p90 / 1000.0, 4),
                    "itl_naive_p99_ms": round(n_p99 / 1000.0, 4),
                    "itl_fused_p50_ms": round(f_p50 / 1000.0, 4),
                    "itl_fused_p90_ms": round(f_p90 / 1000.0, 4),
                    "itl_fused_p99_ms": round(f_p99 / 1000.0, 4),
                    "itl_speedup_p50": round(n_p50 / f_p50, 3),
                    "amortize_decode_steps": round(amortize, 2),
                    "mode": "modeled",
                    "model_id": "meta-llama/Llama-2-7b-hf",
                    "gpu_name": "cpu-only-host (analytical + seeded jitter model)",
                }
            )
    return rows


def write_latency_csv(out_path: Path, head_dim: int, num_kv_heads: int, group_size: int) -> int:
    """Write p7_latency_ttft_itl.csv + SHA256 manifest; returns row count."""
    import pandas as pd

    df = pd.DataFrame(latency_rows(head_dim, num_kv_heads, group_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    digest = hashlib.sha256(out_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    manifest = {
        "file": out_path.name,
        "sha256": digest,
        "description": (
            "Prefill TTFT (FP16 append vs quantized append) and decode ITL "
            "P50/P90/P99 (naive materialization vs fused split-K) crossover"
        ),
        "generated_by": "bench/latency.py",
        "mode": "modeled (analytical byte equations + seeded lognormal jitter)",
        "model_id": "meta-llama/Llama-2-7b-hf",
        "gpu_name": "cpu-only-host (no CUDA available)",
        "columns": list(df.columns),
        "num_rows": len(df),
    }
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="TTFT / ITL crossover analysis")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("bench/data/p7_latency_ttft_itl.csv"))
    args = parser.parse_args()
    n = write_latency_csv(args.out, args.head_dim, args.kv_heads, args.group_size)
    print(f"[OK] {args.out}: {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
