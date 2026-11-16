#!/usr/bin/env python3
"""Generate Phase 3 committed CSV artifacts with manifests."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.bandwidth import analytical_hbm_bytes

DATA_DIR = Path("bench/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_manifest(csv_path: Path, description: str) -> None:
    df = pd.read_csv(csv_path)
    manifest = {
        "file": csv_path.name,
        "sha256": sha256(csv_path),
        "description": description,
        "generated_by": "scripts/generate_phase3_artifacts.py",
        "columns": list(df.columns),
        "num_rows": len(df),
    }
    csv_path.with_suffix(csv_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def generate_arch_matrix() -> Path:
    """Hardware compatibility matrix for fused-kernel features."""
    rows = [
        {
            "feature": "int8_unpack",
            "sm80_a100": "yes",
            "sm86_rtx3090": "yes",
            "sm89_4090_l40s": "yes",
            "sm90_h100": "yes",
            "sm100_b200": "yes",
            "notes": "uint8 ld.global + int8 tl.dot on all Ampere+ SMs",
        },
        {
            "feature": "int4_nibble_unpack",
            "sm80_a100": "yes",
            "sm86_rtx3090": "yes",
            "sm89_4090_l40s": "yes",
            "sm90_h100": "yes",
            "sm100_b200": "yes",
            "notes": "in-register AND/shift using INT4_LOW_MASK/HIGH_SHIFT from layout.py",
        },
        {
            "feature": "fp8_e4m3_tensorcore",
            "sm80_a100": "no",
            "sm86_rtx3090": "no",
            "sm89_4090_l40s": "yes",
            "sm90_h100": "yes",
            "sm100_b200": "yes",
            "notes": "native FP8 MMA requires sm89+ (Ada/Hopper/Blackwell)",
        },
        {
            "feature": "l2_eviction_hints",
            "sm80_a100": "yes",
            "sm86_rtx3090": "yes",
            "sm89_4090_l40s": "yes",
            "sm90_h100": "yes",
            "sm100_b200": "yes",
            "notes": "evict_last on sink/scales, evict_first on streaming KV pages",
        },
        {
            "feature": "splitk_gqa_broadcast",
            "sm80_a100": "yes",
            "sm86_rtx3090": "yes",
            "sm89_4090_l40s": "yes",
            "sm90_h100": "yes",
            "sm100_b200": "yes",
            "notes": "grid=(batch*kv_heads, num_splits); K loaded once, dotted vs GQA group",
        },
    ]
    df = pd.DataFrame(rows)
    path = DATA_DIR / "arch_compat_matrix.csv"
    df.to_csv(path, index=False)
    write_manifest(
        path, "Fused kernel feature support across NVIDIA GPU architectures (sm80-sm100)"
    )
    return path


def generate_hbm_model() -> Path:
    """Analytical HBM traffic model across quantization configs."""
    rows = []
    for group_size in (32, 64, 128):
        for bits in (16, 8, 4, 2):
            m = analytical_hbm_bytes(
                num_tokens=2048,
                head_dim=128,
                num_kv_heads=32,
                bits=bits,
                group_size=group_size,
            )
            rows.append(
                {
                    "head_dim": m["channels"] // 32,
                    "num_kv_heads": 32,
                    "group_size": group_size,
                    "bits": bits,
                    "quantized_mb": round(m["quantized_bytes"] / (1024**2), 4),
                    "fp16_materialized_mb": round(m["fp16_materialized_bytes"] / (1024**2), 4),
                    "naive_hbm_mb": round(m["naive_hbm_bytes"] / (1024**2), 4),
                    "fused_hbm_mb": round(m["fused_hbm_bytes"] / (1024**2), 4),
                    "hbm_traffic_reduction": m["hbm_traffic_reduction"],
                }
            )
    df = pd.DataFrame(rows)
    path = DATA_DIR / "p3_hbm_traffic_model.csv"
    df.to_csv(path, index=False)
    write_manifest(
        path, "Analytical HBM traffic model: naive FP16 materialization vs fused split-K"
    )
    return path


def main() -> int:
    print("Generating Phase 3 artifacts...")
    a = generate_arch_matrix()
    print(f"  [OK] {a}")
    b = generate_hbm_model()
    print(f"  [OK] {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
