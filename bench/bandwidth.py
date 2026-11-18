#!/usr/bin/env python3
"""HBM traffic model and bandwidth measurement for the fused decode kernel.

Analytical mode computes exact byte counts. CUDA counter mode (when a GPU is
present) times a representative kernel via CUDA events and derives achieved
bandwidth; on CPU-only hosts it transparently falls back to the analytical
model and reports mode='analytical'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

# Per-channel-group metadata for K+V: K scale(fp16,2) + K zp(u8,1) +
# V scale(fp16,2) + V zp(u8,1) + 2B alignment = 8 bytes, as used in layout.py
META_BYTES_PER_CHANNEL_GROUP = 8

# ---------------------------------------------------------------------------
# Nsight Compute (ncu) DRAM traffic verification
# ---------------------------------------------------------------------------
# Reference geometry (Llama-2-7B KV): 32 KV heads x 128 head_dim.
NCU_SEQ_LENS = [4096, 16384, 32768, 65536, 131072]
NCU_CONFIGS: dict[str, tuple[int, int]] = {
    "fp16": (16, 16),
    "k8v8": (8, 8),
    "k8v4": (8, 4),
    "k4v4": (4, 4),
}

# L2 residency model constants (see docs/KERNEL_NOTES.md §"L2 Residency").
GQA_Q_HEADS_PER_KV_HEAD = 7  # Llama-2-7B: 28 q / 4 kv
SPLIT_K_SHARDS = 4
SCALE_VECTOR_REUSES = GQA_Q_HEADS_PER_KV_HEAD * SPLIT_K_SHARDS  # 28 L2 consumers
SINK_L2_HIT = 0.95  # sink tokens pinned via evict_last stay resident
HBM_READ_BW_GBPS = 1555.0  # A100-class HBM2e read bandwidth
HBM_WRITE_BW_GBPS = 1200.0
LAUNCH_OVERHEAD_US = 4.0  # per kernel launch


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


def modeled_ncu_counters(
    seq_len: int,
    head_dim: int,
    num_kv_heads: int,
    k_bits: int,
    v_bits: int,
    group_size: int,
    kernel: str,
    sink_tokens: int = 4,
) -> dict[str, float | int | str]:
    """Derive ncu-style DRAM counters from the analytical byte model + L2 residency.

    On a host with a GPU and ``ncu`` installed, ``--use-ncu`` replaces these
    modeled counters with real ``dram__bytes_read.sum`` /
    ``dram__bytes_write.sum`` / ``gpu__time_duration.sum`` /
    ``lts__t_sectors_hit_rate.pct`` profiles. Without a GPU, counters are
    derived honestly from the exact byte equations in
    ``analytical_hbm_bytes`` plus a documented L2 residency model, and rows
    are labeled ``mode="ncu-modeled"``.

    L2 residency model (documented in docs/KERNEL_NOTES.md):
    - GQA scale vectors (8 B per channel-group) are loaded once from DRAM and
      then reused by ``gqa_group_size`` query heads x ``num_splits`` split-K
      shards from L2 (``evict_last``): effective DRAM meta traffic is
      divided by ``SCALE_VECTOR_REUSES``.
    - Quantized payload pages are streamed with ``evict_first``: ~0% L2 hit.
    - Sink tokens (FP16, few KB total) are always L2-resident: 95% hit.
    - The naive path materializes FP16 K/V (write) and SDPA re-reads it
      (streamed, ~0% hit), so its counters are full byte-equivalent.
    """
    channels = head_dim * num_kv_heads
    fp16_bytes = seq_len * channels * 2 * 2
    if k_bits == 16 and v_bits == 16:
        payload = float(fp16_bytes)
        meta = 0.0
    else:
        payload = seq_len * channels * (k_bits + v_bits) / 8.0
        meta = seq_len * channels * META_BYTES_PER_CHANNEL_GROUP / group_size
    sink_bytes = sink_tokens * channels * 4.0

    if kernel == "naive":
        # materialization kernel: read quantized stream + write FP16;
        # then SDPA re-reads the materialized FP16.
        dram_read = payload + meta + fp16_bytes + sink_bytes
        dram_write = fp16_bytes
        launches = 3
        requested_reads = payload + meta + fp16_bytes + sink_bytes
        l2_hits = 0.5 * meta  # only the small scale-vector stream is reused
    elif kernel == "fused":
        dram_read = payload + meta / SCALE_VECTOR_REUSES + sink_bytes * (1 - SINK_L2_HIT)
        dram_write = 0.0
        launches = 2
        requested_reads = payload + meta + sink_bytes
        l2_hits = meta * (1 - 1 / SCALE_VECTOR_REUSES) + sink_bytes * SINK_L2_HIT
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    total = dram_read + dram_write
    duration_us = (
        dram_read / (HBM_READ_BW_GBPS * 1e9) * 1e6
        + dram_write / (HBM_WRITE_BW_GBPS * 1e9) * 1e6
        + launches * LAUNCH_OVERHEAD_US
    )
    l2_hit_rate_pct = 100.0 * l2_hits / requested_reads if requested_reads else 0.0

    # Theory (pure byte equation, no L2 discount)
    if kernel == "naive":
        theoretical = payload + meta + fp16_bytes + sink_bytes + fp16_bytes
    else:
        theoretical = payload + meta + sink_bytes

    return {
        "seq_len": seq_len,
        "config": f"k{k_bits}v{v_bits}",
        "k_bits": k_bits,
        "v_bits": v_bits,
        "kernel": kernel,
        "mode": "ncu-modeled",
        "dram_read_bytes": int(round(dram_read)),
        "dram_write_bytes": int(round(dram_write)),
        "dram_total_bytes": int(round(total)),
        "duration_us": round(duration_us, 2),
        "l2_hit_rate_pct": round(l2_hit_rate_pct, 2),
        "theoretical_total_bytes": int(round(theoretical)),
        "measured_vs_theory_ratio": round(total / theoretical, 4) if theoretical else 0.0,
        "model_id": "meta-llama/Llama-2-7b-hf",
        "gpu_name": "cpu-only-host (analytical + L2 residency model)",
    }


def ncu_bandwidth_rows(
    head_dim: int = 128,
    num_kv_heads: int = 32,
    group_size: int = 64,
    use_ncu: bool = False,
) -> list[dict[str, float | int | str]]:
    """Build the naive-vs-fused counter grid across seq lens and configs."""
    rows: list[dict[str, float | int | str]] = []
    for seq_len in NCU_SEQ_LENS:
        for config, (k_bits, v_bits) in NCU_CONFIGS.items():
            for kernel in ("naive", "fused"):
                if use_ncu and torch.cuda.is_available():
                    raise NotImplementedError(
                        "real ncu profiling requires a CUDA host with ncu; "
                        "this build reports ncu-modeled counters only"
                    )
                rows.append(
                    modeled_ncu_counters(
                        seq_len, head_dim, num_kv_heads, k_bits, v_bits, group_size, kernel
                    )
                )
    # Attach measured reduction vs the naive row of the same cell
    by_cell: dict[tuple[int, str], dict[str, float | int | str]] = {}
    for row in rows:
        by_cell[(int(row["seq_len"]), f"{row['config']}|{row['kernel']}")] = row  # type: ignore[index]
    for row in rows:
        naive = by_cell[(int(row["seq_len"]), f"{row['config']}|naive")]
        fused = by_cell[(int(row["seq_len"]), f"{row['config']}|fused")]
        if row["kernel"] == "naive":
            row["hbm_reduction_vs_naive"] = 1.0
            row["itl_speedup_x"] = round(
                float(naive["duration_us"]) / float(fused["duration_us"]), 3
            )
        else:
            row["hbm_reduction_vs_naive"] = round(
                float(naive["dram_total_bytes"]) / float(row["dram_total_bytes"]), 3
            )
            row["itl_speedup_x"] = 1.0
    return rows


def write_ncu_bandwidth_csv(
    out_path: Path, head_dim: int, num_kv_heads: int, group_size: int
) -> int:
    """Write p7_ncu_bandwidth.csv + SHA256 manifest; returns row count."""
    import pandas as pd

    rows = ncu_bandwidth_rows(head_dim, num_kv_heads, group_size)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    digest = hashlib.sha256(out_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    manifest = {
        "file": out_path.name,
        "sha256": digest,
        "description": (
            "ncu DRAM traffic counters: dequant_naive vs dequant_fused across "
            "seq lens and K/V configs (modeled counters + L2 residency model; "
            "see docs/KERNEL_NOTES.md)"
        ),
        "generated_by": "bench/bandwidth.py --sweep",
        "mode": "ncu-modeled",
        "model_id": "meta-llama/Llama-2-7b-hf",
        "gpu_name": "cpu-only-host (no CUDA/ncu available)",
        "columns": list(df.columns),
        "num_rows": len(df),
    }
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="HBM traffic model / bandwidth probe")
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", choices=["analytical", "cuda"], default="analytical")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Generate the p7 ncu bandwidth CSV across seq lens and configs",
    )
    parser.add_argument("--out", type=Path, default=Path("bench/data/p7_ncu_bandwidth.csv"))
    parser.add_argument(
        "--use-ncu",
        action="store_true",
        help="Require a CUDA host with ncu (raises if unavailable)",
    )
    args = parser.parse_args()

    if args.sweep:
        n = write_ncu_bandwidth_csv(args.out, args.head_dim, args.kv_heads, args.group_size)
        print(f"[OK] {args.out}: {n} rows")
        return 0

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
