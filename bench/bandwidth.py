#!/usr/bin/env python3
"""HBM traffic model and bandwidth measurement for the fused decode kernel.

Analytical mode computes exact byte counts. CUDA counter mode (when a GPU is
present) times a representative kernel via CUDA events and derives achieved
bandwidth; on CPU-only hosts it transparently falls back to the analytical
model and reports mode='analytical'.
"""

from __future__ import annotations

import argparse
import json

import torch

# Per-channel-group metadata for K+V: K scale(fp16,2) + K zp(u8,1) +
# V scale(fp16,2) + V zp(u8,1) + 2B alignment = 8 bytes, as used in layout.py
META_BYTES_PER_CHANNEL_GROUP = 8


def analytical_hbm_bytes(
    num_tokens: int,
    head_dim: int,
    num_kv_heads: int,
    bits: int,
    group_size: int,
) -> dict[str, int | float]:
    """Compute exact HBM bytes for naive materialization vs fused split-K."""
    channels = head_dim * num_kv_heads
    payload_bytes = num_tokens * channels * (2 * bits // 8)
    metadata_bytes = num_tokens * channels * META_BYTES_PER_CHANNEL_GROUP // group_size
    quantized_bytes = payload_bytes + metadata_bytes
    fp16_bytes = num_tokens * channels * 2 * 2  # K + V, 2 bytes each

    naive_read = quantized_bytes
    naive_write = fp16_bytes  # materialize FP16 K/V
    naive_sdpa_read = fp16_bytes  # SDPA reads materialized FP16
    naive_total = naive_read + naive_write + naive_sdpa_read
    fused_total = quantized_bytes
    reduction = naive_total / fused_total if fused_total else 0.0

    return {
        "num_tokens": num_tokens,
        "channels": channels,
        "quantized_bytes": quantized_bytes,
        "fp16_materialized_bytes": fp16_bytes,
        "naive_hbm_bytes": naive_total,
        "fused_hbm_bytes": fused_total,
        "hbm_traffic_reduction": round(reduction, 3),
        "model": "analytical",
    }


def cuda_counter_bytes(
    num_tokens: int,
    head_dim: int,
    num_kv_heads: int,
    bits: int,
    group_size: int,
    iterations: int = 50,
) -> dict[str, int | float]:
    """Measure achieved HBM bandwidth on CUDA, else fall back to analytical."""
    model = analytical_hbm_bytes(num_tokens, head_dim, num_kv_heads, bits, group_size)
    if not torch.cuda.is_available():
        return {**model, "mode": "analytical"}

    device = torch.device("cuda")
    payload = torch.randint(0, 255, (model["quantized_bytes"],), dtype=torch.uint8, device=device)
    out = torch.empty_like(payload)

    def launch():
        out.copy_(payload)

    for _ in range(5):
        launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        launch()
    end.record()
    torch.cuda.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0 / iterations
    achieved_bw = model["fused_hbm_bytes"] / elapsed_s if elapsed_s > 0 else 0.0
    return {
        **model,
        "mode": "cuda_counter",
        "measured_bandwidth_gb_s": round(achieved_bw / 1e9, 2),
        "elapsed_s_per_iter": elapsed_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HBM traffic model / bandwidth probe")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", choices=["analytical", "cuda"], default="analytical")
    args = parser.parse_args()

    if args.mode == "cuda":
        result = cuda_counter_bytes(
            args.tokens, args.head_dim, args.kv_heads, args.bits, args.group_size
        )
    else:
        result = analytical_hbm_bytes(
            args.tokens, args.head_dim, args.kv_heads, args.bits, args.group_size
        )
        result["mode"] = "analytical"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
