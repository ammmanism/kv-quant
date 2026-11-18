#!/usr/bin/env python3
"""CLI: cross-validate measured KV-cache bytes against the analytical calculator.

Writes bench/data/p7_vram_crossval.csv + SHA256 manifest. Each row builds a
real HybridPagedKVCache through the public append path and compares its live
byte accounting against ``kv_quant.vram.kv_cache_calculator_bytes``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# 10 K/V combos x 2 context lengths = 20 configs
BIT_COMBOS: list[tuple[int, int]] = [
    (16, 16),
    (8, 8),
    (8, 4),
    (4, 8),
    (4, 4),
    (6, 6),
    (6, 4),
    (4, 6),
    (8, 6),
    (6, 8),
]
SEQ_LENS = [4096, 16384]


def main() -> int:

    from kv_quant.vram import cross_validate_vram

    parser = argparse.ArgumentParser(description="VRAM cross-validation sweep")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=4)  # Qwen2.5-7B geometry
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("bench/data/p7_vram_crossval.csv"))
    args = parser.parse_args()

    configs = [(kb, vb, seq) for seq in SEQ_LENS for kb, vb in BIT_COMBOS]
    df = cross_validate_vram(
        configs,
        num_layers=args.num_layers,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    df["model_id"] = "Qwen/Qwen2.5-7B-Instruct"
    df["num_layers"] = args.num_layers
    df.to_csv(args.out, index=False)

    digest = hashlib.sha256(args.out.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    manifest = {
        "file": args.out.name,
        "sha256": digest,
        "description": (
            "Measured HybridPagedKVCache byte accounting (real append path, "
            "shared pages counted once) vs kv-cache-calculator analytical "
            "prediction across 20 K/V configs"
        ),
        "generated_by": "bench/vram_crossval.py",
        "measure_method": "cache_object_bytes (CPU; CUDA peak on GPU hosts)",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpu_name": "cpu-only-host",
        "num_layers": args.num_layers,
        "num_kv_heads": args.kv_heads,
        "head_dim": args.head_dim,
        "r_squared": float(df["r_squared"].iloc[0]),
        "mape_pct": float(df["mape_pct"].iloc[0]),
        "columns": list(df.columns),
        "num_rows": len(df),
    }
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[OK] {args.out}: {len(df)} rows, R^2={manifest['r_squared']:.6f}, "
        f"MAPE={manifest['mape_pct']:.4f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
