#!/usr/bin/env python3
"""Phase 13: ragged-batch CoW concurrency and throughput capacity study.

Emits ``bench/data/p13_ragged_concurrency.csv`` (+ ``.manifest.json``).

Model (fully disclosed in the manifest; nothing here is a GPU measurement):
- Geometry: Qwen2.5-7B-Instruct KV (32 layers, 4 KV heads, head_dim 128), so
  ``16384`` elements per token per K or V tensor. Bytes/elem come from the
  pinned ``kv_quant/layout.py`` equation (INT4 sym `0.53125`, INT4 asym
  `0.5625`, FP16 `2.0`), i.e. the same contract the VRAM cross-validation
  matched at R^2=1.0.
- CoW: the 24576-token shared system+repo prefix is stored ONCE at the
  session's quantized rate; each session pays only its divergent
  ``context_len - 24576`` tokens plus 4 FP16 sink tokens.
- Throughput is bandwidth-bound: ``bytes_per_us`` is taken from the committed
  `p7_ncu_bandwidth` fused row (dram_total_bytes / duration_us) at the matching
  seq_len; one decode step reads the shared prefix once plus every session's
  private tokens. ``tok/s = N * 1e6 / (step_bytes / bytes_per_us)``. This is an
  upper bound: it assumes ideal page sharing and no attention/softmax compute
  cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_quant.layout import bytes_per_element  # noqa: E402
from tasks.base import SweepManifest  # noqa: E402

DATA_DIR = Path("bench/data")
NCU_CSV = DATA_DIR / "p7_ncu_bandwidth.csv"
LAYERS, KV_HEADS, HEAD_DIM = 32, 4, 128
ELEMS_PER_TOKEN = LAYERS * KV_HEADS * HEAD_DIM  # per K or V tensor
SHARED_PREFIX_TOKENS = 24000  # exactly 375 blocks of 64 (mandate)
SINK_TOKENS = 4
BUDGETS_GIB = (24, 48, 80)
CONTEXTS = (32768, 65536)
GIB = 1024**3

CONFIGS = {
    # name: (k_bits, k_asym, v_bits, v_asym, ncu_config)
    "fp16": (16, False, 16, False, "k16v16"),
    "fp16_cow": (16, False, 16, False, "k16v16"),
    "k8v4_cow": (8, False, 4, True, "k8v4"),
    "k4v4_hadamard_cow": (4, False, 4, True, "k4v4"),
}


def bytes_per_token(k_bits: int, k_asym: bool, v_bits: int, v_asym: bool) -> float:
    return ELEMS_PER_TOKEN * (
        bytes_per_element(k_bits, 64, k_asym) + bytes_per_element(v_bits, 64, v_asym)
    )


def bandwidth_b_per_us(ncu: pd.DataFrame, ncu_config: str, seq_len: int) -> float:
    row = ncu[(ncu.config == ncu_config) & (ncu.kernel == "fused") & (ncu.seq_len == seq_len)]
    if row.empty:
        raise ValueError(f"no ncu row for {ncu_config} @ {seq_len}")
    return float(row.dram_total_bytes.iloc[0]) / float(row.duration_us.iloc[0])


def main() -> int:
    ncu = pd.read_csv(NCU_CSV)
    sink_bytes = SINK_TOKENS * ELEMS_PER_TOKEN * 2 * 2.0  # K+V FP16
    rows: list[dict] = []
    for budget in BUDGETS_GIB:
        budget_bytes = budget * GIB
        for ctx in CONTEXTS:
            for name, (kb, ka, vb, va, ncu_cfg) in CONFIGS.items():
                cow = name.endswith("_cow")
                bpt = bytes_per_token(kb, ka, vb, va)
                if cow:
                    shared_bytes = SHARED_PREFIX_TOKENS * bpt
                    private_bytes = (ctx - SHARED_PREFIX_TOKENS) * bpt + sink_bytes
                else:
                    shared_bytes = 0.0
                    private_bytes = ctx * bpt + sink_bytes
                if private_bytes <= 0 or budget_bytes <= shared_bytes:
                    continue
                sessions = int((budget_bytes - shared_bytes) // private_bytes)
                if sessions <= 0:
                    continue
                bw = bandwidth_b_per_us(ncu, ncu_cfg, ctx)
                step_bytes = shared_bytes + sessions * private_bytes
                modeled_tok_s = sessions * 1e6 / (step_bytes / bw)
                rows.append(
                    {
                        "budget_gib": budget,
                        "context_len": ctx,
                        "config": name,
                        "cow": cow,
                        "max_concurrent_sessions": sessions,
                        "bytes_per_token": round(bpt, 3),
                        "private_bytes_per_session": int(private_bytes),
                        "shared_prefix_bytes": int(shared_bytes),
                        "used_bytes_at_max": int(step_bytes),
                        "modeled_decode_tok_s": round(modeled_tok_s, 1),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["budget_gib", "context_len", "config"], kind="mergesort")
    csv = DATA_DIR / "p13_ragged_concurrency.csv"
    df.to_csv(csv, index=False)
    manifest = SweepManifest.capture(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        param_count_b=7.61,
        num_q_heads=28,
        num_kv_heads=KV_HEADS,
        rotary_dim=128,
        seed=42,
    )
    manifest.extra = {
        "description": "Ragged-batch CoW concurrency capacity and bandwidth-bound decode throughput",
        "mode": "capacity model + modeled throughput (no GPU on host)",
        "geometry": f"{LAYERS} layers x {KV_HEADS} KV heads x {HEAD_DIM} head_dim = "
        f"{ELEMS_PER_TOKEN} elems/token per K or V tensor",
        "shared_prefix_tokens": SHARED_PREFIX_TOKENS,
        "sink_tokens": SINK_TOKENS,
        "bytes_source": "kv_quant.layout.bytes_per_elem (golden-hashed contract)",
        "throughput_model": "bandwidth-bound: bytes_per_us = dram_total_bytes/duration_us "
        "from p7_ncu_bandwidth fused rows at the matching seq_len; "
        "tok/s = N*1e6/((shared + N*private)/bytes_per_us). Upper bound: ignores "
        "attention/softmax compute and assumes ideal CoW page sharing.",
        "source_csvs": [NCU_CSV.as_posix()],
    }
    manifest.with_csv_hash(csv).write(csv.with_suffix(csv.suffix + ".manifest.json"))
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
